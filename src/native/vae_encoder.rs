//! VAE encoder composition with bounded group-normalization reductions.

use candle_core::{D, DType, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};

use super::Architecture;
use super::conv::{BandedConv2d, conv2d};
use crate::{Result, pipeline::Checkpoint};

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "metal_ops.rs"]
pub(super) mod metal_ops;

const GROUPS: usize = 32;
const EPS: f64 = 1e-6;
const REDUCTION_CHUNK: usize = 256;

/// Sum a contiguous [batch, groups, elements] tensor in bounded stages.
/// A remaining prime larger than the bound has no equal-sized factorization;
/// use one direct sum in that case rather than looping indefinitely.
fn group_sum(xs: &Tensor) -> candle_core::Result<Tensor> {
    let (batch, groups, mut remaining) = xs.dims3()?;
    let mut sums = xs.clone();
    while remaining > REDUCTION_CHUNK {
        let factor = (2..=REDUCTION_CHUNK)
            .rev()
            .find(|factor| remaining.is_multiple_of(*factor));
        let Some(factor) = factor else {
            return sums.sum_keepdim(2);
        };
        remaining /= factor;
        sums = sums.reshape((batch, groups, remaining, factor))?.sum(3)?;
    }
    sums.sum_keepdim(2)
}

pub(super) struct StableGroupNorm {
    weight: Tensor,
    bias: Tensor,
    channels: usize,
}

impl StableGroupNorm {
    pub(super) fn new(channels: usize, vs: VarBuilder<'_>) -> candle_core::Result<Self> {
        if channels == 0 || !channels.is_multiple_of(GROUPS) {
            candle_core::bail!("VAE group normalization requires channels divisible by 32")
        }
        Ok(Self {
            weight: vs.get_with_hints(channels, "weight", nn::Init::Const(1.))?,
            bias: vs.get_with_hints(channels, "bias", nn::Init::Const(0.))?,
            channels,
        })
    }

    pub(super) fn forward_reference(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, channels, height, width) = input.dims4()?;
        if channels != self.channels || batch == 0 || height == 0 || width == 0 {
            candle_core::bail!("invalid VAE group normalization input shape")
        }
        let count = (channels / GROUPS)
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| candle_core::Error::Msg("VAE group size overflows".into()))?;
        let dtype = input.dtype();
        let stats_dtype = match dtype {
            DType::F16 | DType::BF16 => DType::F32,
            other => other,
        };
        let xs = input
            .reshape((batch, GROUPS, count))?
            .to_dtype(stats_dtype)?;
        let mean = (group_sum(&xs)? / count as f64)?;
        let centered = xs.broadcast_sub(&mean)?;
        let variance = (group_sum(&centered.sqr()?)? / count as f64)?;
        let normalized = centered.broadcast_div(&(variance + EPS)?.sqrt()?)?;
        normalized
            .to_dtype(dtype)?
            .reshape((batch, channels, height, width))?
            .broadcast_mul(&self.weight.reshape((1, channels, 1, 1))?)?
            .broadcast_add(&self.bias.reshape((1, channels, 1, 1))?)
    }

    pub(super) fn forward_silu(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        nn::ops::silu(&self.forward(input)?)
    }
}

impl Module for StableGroupNorm {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if input.device().is_metal()
            && input.dtype() == DType::F32
            && self.weight.dtype() == DType::F32
            && self.bias.dtype() == DType::F32
            && u32::try_from(input.elem_count()).is_ok()
        {
            let (batch, channels, height, width) = input.dims4()?;
            if channels != self.channels || batch == 0 || height == 0 || width == 0 {
                candle_core::bail!("invalid VAE group normalization input shape")
            }
            return metal_ops::group_norm(input, &self.weight, &self.bias);
        }
        self.forward_reference(input)
    }
}

pub(super) struct Resnet {
    norm1: StableGroupNorm,
    conv1: BandedConv2d,
    norm2: StableGroupNorm,
    conv2: BandedConv2d,
    shortcut: Option<BandedConv2d>,
}

impl Resnet {
    pub(super) fn new(
        vs: VarBuilder<'_>,
        input: usize,
        output: usize,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            norm1: StableGroupNorm::new(input, vs.pp("norm1"))?,
            conv1: conv2d(input, output, 3, 1, 1, vs.pp("conv1"))?,
            norm2: StableGroupNorm::new(output, vs.pp("norm2"))?,
            conv2: conv2d(output, output, 3, 1, 1, vs.pp("conv2"))?,
            shortcut: if input == output {
                None
            } else {
                Some(conv2d(input, output, 1, 1, 0, vs.pp("conv_shortcut"))?)
            },
        })
    }

    pub(super) fn forward_checked(
        &self,
        xs: &Tensor,
        checkpoint: Checkpoint<'_>,
    ) -> Result<Tensor> {
        checkpoint()?;
        let shortcut = match &self.shortcut {
            Some(conv) => conv.forward_checked(xs, checkpoint)?,
            None => xs.clone(),
        };
        let hidden = self
            .conv1
            .forward_checked(&self.norm1.forward_silu(xs)?, checkpoint)?;
        let hidden = self
            .conv2
            .forward_checked(&self.norm2.forward_silu(&hidden)?, checkpoint)?;
        Ok((shortcut + hidden)?)
    }
}

pub(super) struct DownBlock {
    resnets: Vec<Resnet>,
    downsample: Option<BandedConv2d>,
}

impl DownBlock {
    fn new(
        vs: VarBuilder<'_>,
        input: usize,
        output: usize,
        layers: usize,
        downsample: bool,
    ) -> candle_core::Result<Self> {
        let mut resnets = Vec::with_capacity(layers);
        for i in 0..layers {
            resnets.push(Resnet::new(
                vs.pp("resnets").pp(i),
                if i == 0 { input } else { output },
                output,
            )?);
        }
        let downsample = if downsample {
            Some(conv2d(
                output,
                output,
                3,
                2,
                0,
                vs.pp("downsamplers").pp(0).pp("conv"),
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsample,
        })
    }
}

impl DownBlock {
    pub(super) fn forward_checked(
        &self,
        input: &Tensor,
        checkpoint: Checkpoint<'_>,
    ) -> Result<Tensor> {
        let mut xs = input.clone();
        for resnet in &self.resnets {
            checkpoint()?;
            xs = resnet.forward_checked(&xs, checkpoint)?;
        }
        checkpoint()?;
        Ok(match &self.downsample {
            Some(conv) => conv.forward_checked(
                &xs.pad_with_zeros(D::Minus1, 0, 1)?
                    .pad_with_zeros(D::Minus2, 0, 1)?,
                checkpoint,
            )?,
            None => xs,
        })
    }
}

impl Module for DownBlock {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_checked(input, &|| Ok(()))
            .map_err(|e| candle_core::Error::Msg(e.to_string()))
    }
}

fn projection(vs: VarBuilder<'_>, channels: usize) -> candle_core::Result<nn::Linear> {
    match vs.get((channels, channels), "weight") {
        Ok(_) => nn::linear(channels, channels, vs),
        Err(_) => {
            let weight = vs
                .get((channels, channels, 1, 1), "weight")?
                .reshape((channels, channels))?;
            let bias = vs.get(channels, "bias")?;
            Ok(nn::Linear::new(weight, Some(bias)))
        }
    }
}

struct Attention {
    norm: StableGroupNorm,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    output: nn::Linear,
    channels: usize,
}

impl Attention {
    fn new(vs: VarBuilder<'_>, channels: usize) -> candle_core::Result<Self> {
        let (q, k, v, out) = if vs.contains_tensor("to_q.weight") {
            ("to_q", "to_k", "to_v", "to_out.0")
        } else {
            ("query", "key", "value", "proj_attn")
        };
        Ok(Self {
            norm: StableGroupNorm::new(channels, vs.pp("group_norm"))?,
            query: projection(vs.pp(q), channels)?,
            key: projection(vs.pp(k), channels)?,
            value: projection(vs.pp(v), channels)?,
            output: projection(vs.pp(out), channels)?,
            channels,
        })
    }

    #[cfg(test)]
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_checked(input, &|| Ok(()))
            .map_err(|error| candle_core::Error::Msg(error.to_string()))
    }

    fn forward_checked(&self, input: &Tensor, checkpoint: Checkpoint<'_>) -> Result<Tensor> {
        checkpoint()?;
        let (batch, channels, height, width) = input.dims4()?;
        if batch > 1 && input.device().is_metal() {
            // Candle 0.11's batched Metal attention changes entries after
            // the first even when normalization and projections agree.
            // Keep each image on the qualified single-image device path.
            if batch > crate::pipeline::MAX_BATCH {
                return Err(candle_core::Error::Msg(
                    "VAE attention batch exceeds the inference limit".into(),
                )
                .into());
            }
            let mut entries = Vec::new();
            entries.try_reserve_exact(batch).map_err(|error| {
                candle_core::Error::Msg(format!("cannot allocate VAE attention batch: {error}"))
            })?;
            for entry in 0..batch {
                checkpoint()?;
                entries.push(self.forward_checked(&input.narrow(0, entry, 1)?, checkpoint)?);
            }
            return Ok(Tensor::cat(&entries, 0)?);
        }
        let tokens = height
            .checked_mul(width)
            .ok_or_else(|| candle_core::Error::Msg("VAE attention shape overflows".into()))?;
        let xs = self
            .norm
            .forward(input)?
            .reshape((batch, channels, tokens))?
            .transpose(1, 2)?;
        // Candle 0.11's CPU linear path flattens batched inputs. A transposed
        // [B, tokens, channels] view has a different stride between images;
        // materialize it before projection so entries after the first use
        // their own pixels. Other device and batch-one paths stay unchanged.
        let xs = if batch > 1 && input.device().is_cpu() {
            xs.contiguous()?
        } else {
            xs
        };
        let query = self
            .query
            .forward(&xs)?
            .unsqueeze(1)?
            .to_dtype(DType::F32)?;
        let key = self.key.forward(&xs)?.unsqueeze(1)?.to_dtype(DType::F32)?;
        let value = self
            .value
            .forward(&xs)?
            .unsqueeze(1)?
            .to_dtype(DType::F32)?;
        let scale = (self.channels as f64).powf(-0.25);
        let output = super::attention::attend(
            &(query * scale)?,
            &(key.t()? * scale)?,
            &value,
            checkpoint,
            |scores| nn::ops::softmax(scores, D::Minus1),
        )?;
        let xs = output
            .to_dtype(input.dtype())?
            .transpose(1, 2)?
            .contiguous()?
            .flatten_from(D::Minus2)?;
        let xs = self
            .output
            .forward(&xs)?
            .t()?
            .reshape((batch, channels, height, width))?;
        Ok((xs + input)?)
    }
}

pub(super) struct MidBlock {
    first: Resnet,
    attention: Attention,
    second: Resnet,
}

impl MidBlock {
    pub(super) fn new(vs: VarBuilder<'_>, channels: usize) -> candle_core::Result<Self> {
        Ok(Self {
            first: Resnet::new(vs.pp("resnets").pp(0), channels, channels)?,
            attention: Attention::new(vs.pp("attentions").pp(0), channels)?,
            second: Resnet::new(vs.pp("resnets").pp(1), channels, channels)?,
        })
    }

    #[cfg(test)]
    pub(super) fn forward(
        &self,
        xs: &Tensor,
        _temb: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        self.forward_checked(xs, &|| Ok(()))
            .map_err(|error| candle_core::Error::Msg(error.to_string()))
    }

    pub(super) fn forward_checked(
        &self,
        xs: &Tensor,
        checkpoint: Checkpoint<'_>,
    ) -> Result<Tensor> {
        let xs = self.first.forward_checked(xs, checkpoint)?;
        let xs = self.attention.forward_checked(&xs, checkpoint)?;
        self.second.forward_checked(&xs, checkpoint)
    }
}

pub(super) struct VaeEncoder {
    pub(super) conv_in: BandedConv2d,
    pub(super) down: Vec<DownBlock>,
    pub(super) mid: MidBlock,
    pub(super) norm_out: StableGroupNorm,
    pub(super) conv_out: BandedConv2d,
    pub(super) quant_conv: BandedConv2d,
}

impl VaeEncoder {
    pub(super) fn new(vs: VarBuilder<'_>, arch: &Architecture) -> candle_core::Result<Self> {
        let channels = arch.vae_channels;
        let enc = vs.pp("encoder");
        let conv_in = conv2d(3, channels[0], 3, 1, 1, enc.pp("conv_in"))?;
        let mut down = Vec::with_capacity(channels.len());
        for (i, &output) in channels.iter().enumerate() {
            let input = if i == 0 { channels[0] } else { channels[i - 1] };
            down.push(DownBlock::new(
                enc.pp("down_blocks").pp(i),
                input,
                output,
                arch.vae_layers,
                i + 1 < channels.len(),
            )?);
        }
        let last = channels[3];
        Ok(Self {
            conv_in,
            down,
            mid: MidBlock::new(enc.pp("mid_block"), last)?,
            norm_out: StableGroupNorm::new(last, enc.pp("conv_norm_out"))?,
            conv_out: conv2d(last, 8, 3, 1, 1, enc.pp("conv_out"))?,
            quant_conv: conv2d(8, 8, 1, 1, 0, vs.pp("quant_conv"))?,
        })
    }

    pub(super) fn forward(&self, image: &Tensor, checkpoint: Checkpoint<'_>) -> Result<Tensor> {
        let mut xs = self.conv_in.forward_checked(image, checkpoint)?;
        for block in &self.down {
            xs = block.forward_checked(&xs, checkpoint)?;
        }
        checkpoint()?;
        xs = self.mid.forward_checked(&xs, checkpoint)?;
        xs = self.norm_out.forward_silu(&xs)?;
        self.quant_conv
            .forward_checked(&self.conv_out.forward_checked(&xs, checkpoint)?, checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    fn group_norm(bias: f32) -> candle_core::Result<StableGroupNorm> {
        let device = &Device::Cpu;
        let mut weights = HashMap::new();
        weights.insert("weight".into(), Tensor::ones(32, DType::F32, device)?);
        weights.insert("bias".into(), Tensor::from_vec(vec![bias; 32], 32, device)?);
        StableGroupNorm::new(32, VarBuilder::from_tensors(weights, DType::F32, device))
    }

    #[test]
    fn attention_preserves_distinct_batch_entries() -> candle_core::Result<()> {
        check_attention_batch_entries(&Device::Cpu)
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a real Metal device"]
    fn metal_attention_preserves_distinct_batch_entries() -> candle_core::Result<()> {
        check_attention_batch_entries(&Device::new_metal(0)?)
    }

    fn check_attention_batch_entries(device: &Device) -> candle_core::Result<()> {
        let values = |count: usize, multiplier: usize, modulus: usize, scale: f32| {
            (0..count)
                .map(|i| ((i * multiplier % modulus) as f32 - (modulus / 2) as f32) * scale)
                .collect::<Vec<_>>()
        };
        let linear = |multiplier| -> candle_core::Result<nn::Linear> {
            let weight = Tensor::from_vec(
                values(32 * 32, multiplier, 103, 1.0 / 512.0),
                (32, 32),
                device,
            )?;
            let bias = Tensor::from_vec(values(32, multiplier, 31, 1.0 / 256.0), 32, device)?;
            Ok(nn::Linear::new(weight, Some(bias)))
        };
        let attention = Attention {
            norm: StableGroupNorm::new(
                32,
                VarBuilder::from_tensors(
                    HashMap::from([
                        ("weight".into(), Tensor::ones(32, DType::F32, device)?),
                        (
                            "bias".into(),
                            Tensor::from_vec(vec![0.125f32; 32], 32, device)?,
                        ),
                    ]),
                    DType::F32,
                    device,
                ),
            )?,
            query: linear(37)?,
            key: linear(41)?,
            value: linear(43)?,
            output: linear(47)?,
            channels: 32,
        };
        let max_error = |a: &Tensor, b: &Tensor| -> candle_core::Result<f64> {
            (a.to_device(&Device::Cpu)?.to_dtype(DType::F64)?
                - b.to_device(&Device::Cpu)?.to_dtype(DType::F64)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f64>()
        };
        for (batch, height, width) in [(2, 5, 7), (4, 11, 13)] {
            let input = Tensor::from_vec(
                values(batch * 32 * height * width, 53, 211, 1.0 / 64.0),
                (batch, 32, height, width),
                device,
            )?;
            let normalized = attention.norm.forward(&input)?;
            let tokens = normalized
                .reshape((batch, 32, height * width))?
                .transpose(1, 2)?;
            // Probe Candle's raw strided projection as a diagnostic; the
            // behavioral assertion below exercises Attention::forward itself.
            let strided_query = attention.query.forward(&tokens)?;
            let batched_output = attention.forward(&input)?;
            for entry in 0..batch {
                let single = input.narrow(0, entry, 1)?.contiguous()?;
                let single_normalized = attention.norm.forward(&single)?;
                let single_tokens = single_normalized
                    .reshape((1, 32, height * width))?
                    .transpose(1, 2)?;
                let single_query = attention.query.forward(&single_tokens)?;
                let reference_query = single_tokens
                    .to_device(&Device::Cpu)?
                    .to_dtype(DType::F64)?
                    .contiguous()?
                    .reshape((height * width, 32))?
                    .matmul(
                        &attention
                            .query
                            .weight()
                            .to_device(&Device::Cpu)?
                            .to_dtype(DType::F64)?
                            .t()?,
                    )?
                    .broadcast_add(
                        &attention
                            .query
                            .bias()
                            .ok_or_else(|| candle_core::Error::Msg("missing query bias".into()))?
                            .to_device(&Device::Cpu)?
                            .to_dtype(DType::F64)?,
                    )?
                    .unsqueeze(0)?;
                let norm_error = max_error(&normalized.narrow(0, entry, 1)?, &single_normalized)?;
                let single_error = max_error(&single_query, &reference_query)?;
                let strided_error =
                    max_error(&strided_query.narrow(0, entry, 1)?, &reference_query)?;
                eprintln!(
                    "batch {batch} {height}x{width} entry {entry}: norm={norm_error}, single query vs f64={single_error}, raw strided query vs f64={strided_error}"
                );
                assert!(
                    norm_error <= 2e-5,
                    "batch normalization differs by {norm_error}"
                );
                assert!(
                    single_error <= 3e-6,
                    "single query F64 error {single_error}"
                );
                let output_error = max_error(
                    &batched_output.narrow(0, entry, 1)?,
                    &attention.forward(&single)?,
                )?;
                eprintln!(
                    "batch {batch} entry {entry}: attention output difference={output_error}"
                );
                assert!(
                    output_error <= 1e-5,
                    "batch attention differs by {output_error}"
                );
            }
        }
        Ok(())
    }

    // Compute the reference on CPU in f64, independently of Candle's
    // reductions. Transpose and narrow exercise noncontiguous and offset views.
    fn check_normalization_reference(device: &Device) -> candle_core::Result<()> {
        for (batch, channels, height, width, transposed, large_offset) in [
            (1, 32, 1, 1, false, false),
            (2, 64, 3, 7, true, false),
            (2, 32, 1, 257, false, false),
            (2, 32, 1, 4097, false, false),
            (1, 32, 128, 256, false, true),
        ] {
            let elements = (batch + 1) * channels * height * width;
            let values: Vec<f32> = (0..elements)
                .map(|i| {
                    if large_offset {
                        if i % 2 == 0 { 9.99 } else { 10.01 }
                    } else {
                        ((i * 37 % 101) as f32 - 50.0) * 0.03125
                    }
                })
                .collect();
            let shape = if transposed {
                (batch + 1, channels, width, height)
            } else {
                (batch + 1, channels, height, width)
            };
            let source = Tensor::from_vec(values, shape, device)?.narrow(0, 1, batch)?;
            let input = if transposed {
                source.transpose(2, 3)?
            } else {
                source
            };
            let scale: Vec<f32> = (0..channels)
                .map(|channel| 0.5 + (channel % 5) as f32 * 0.125)
                .collect();
            let bias: Vec<f32> = (0..channels)
                .map(|channel| (channel % 7) as f32 * 0.0625 - 0.125)
                .collect();
            let parameter = |values: &[f32]| -> candle_core::Result<Tensor> {
                let strided: Vec<f32> = values.iter().flat_map(|&value| [0.0, value]).collect();
                Tensor::from_vec(strided, (channels, 2), device)?
                    .narrow(1, 1, 1)?
                    .squeeze(1)
            };
            let weights = HashMap::from([
                ("weight".into(), parameter(&scale)?),
                ("bias".into(), parameter(&bias)?),
            ]);
            let norm = StableGroupNorm::new(
                channels,
                VarBuilder::from_tensors(weights, DType::F32, device),
            )?;
            let source = input
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let actual = norm
                .forward(&input)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let fused = norm
                .forward_silu(&input)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let per_channel = height * width;
            let per_group = channels / GROUPS * per_channel;
            let mut max_error = 0.0_f64;
            let mut max_silu_error = 0.0_f64;
            for (group_index, ((group, output), fused_output)) in source
                .chunks_exact(per_group)
                .zip(actual.chunks_exact(per_group))
                .zip(fused.chunks_exact(per_group))
                .enumerate()
            {
                let mean = group.iter().map(|&x| f64::from(x)).sum::<f64>() / per_group as f64;
                let variance = group
                    .iter()
                    .map(|&x| (f64::from(x) - mean).powi(2))
                    .sum::<f64>()
                    / per_group as f64;
                let denominator = (variance + EPS).sqrt();
                for (offset, ((&value, &actual), &fused_actual)) in
                    group.iter().zip(output).zip(fused_output).enumerate()
                {
                    let channel =
                        (group_index % GROUPS) * (channels / GROUPS) + offset / per_channel;
                    let (&scale, &bias) =
                        scale.get(channel).zip(bias.get(channel)).ok_or_else(|| {
                            candle_core::Error::Msg("reference channel missing".into())
                        })?;
                    let expected = (f64::from(value) - mean) / denominator * f64::from(scale)
                        + f64::from(bias);
                    assert!(actual.is_finite(), "nonfinite group normalization output");
                    assert!(fused_actual.is_finite(), "nonfinite norm and SiLU output");
                    max_error = max_error.max((f64::from(actual) - expected).abs());
                    let expected_silu = expected / (1.0 + (-expected).exp());
                    max_silu_error =
                        max_silu_error.max((f64::from(fused_actual) - expected_silu).abs());
                }
            }
            assert!(
                max_error <= 2e-5,
                "shape {batch}x{channels}x{height}x{width}, transposed={transposed}, offset={large_offset}: max error {max_error}"
            );
            assert!(
                max_silu_error <= 2e-5,
                "norm and SiLU shape {batch}x{channels}x{height}x{width}, transposed={transposed}, offset={large_offset}: max error {max_silu_error}"
            );
        }
        Ok(())
    }

    fn check_wide_magnitude_reference(device: &Device) -> candle_core::Result<()> {
        let values: Vec<f32> = (0..2 * 32 * 4096)
            .map(|i| match i % 4 {
                0 => 1e10,
                1 => -1e10,
                2 => 1e-10,
                _ => -1e-10,
            })
            .collect();
        let input = Tensor::from_vec(values.clone(), (2, 32, 1, 4096), device)?;
        let weights = HashMap::from([
            ("weight".into(), Tensor::ones(32, DType::F32, device)?),
            ("bias".into(), Tensor::zeros(32, DType::F32, device)?),
        ]);
        let norm = StableGroupNorm::new(32, VarBuilder::from_tensors(weights, DType::F32, device))?;
        let actual = norm
            .forward(&input)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let denominator = (0.5 * (1e10_f64.powi(2) + 1e-10_f64.powi(2)) + EPS).sqrt();
        let mut max_error = 0.0_f64;
        for (&actual, &input) in actual.iter().zip(&values) {
            assert!(actual.is_finite(), "nonfinite wide-magnitude normalization");
            max_error = max_error.max((f64::from(actual) - f64::from(input) / denominator).abs());
        }
        assert!(max_error <= 2e-5, "wide-magnitude max error {max_error}");
        Ok(())
    }

    #[test]
    fn group_normalization_matches_f64_reference_for_views_and_group_sizes()
    -> candle_core::Result<()> {
        check_normalization_reference(&Device::Cpu)?;
        check_wide_magnitude_reference(&Device::Cpu)
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a real Metal device"]
    fn metal_group_normalization_matches_f64_reference_for_views_and_group_sizes()
    -> candle_core::Result<()> {
        let device = &Device::new_metal(0)?;
        check_normalization_reference(device)?;
        check_wide_magnitude_reference(device)
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a real Metal device"]
    fn metal_half_group_normalization_keeps_existing_dtype_path() -> candle_core::Result<()> {
        let device = &Device::new_metal(0)?;
        let dtype = DType::F16;
        let input = Tensor::arange(0f32, 512f32, device)?
            .reshape((1, 64, 2, 4))?
            .to_dtype(dtype)?;
        let weights = HashMap::from([
            ("weight".into(), Tensor::ones(64, dtype, device)?),
            ("bias".into(), Tensor::zeros(64, dtype, device)?),
        ]);
        let norm = StableGroupNorm::new(64, VarBuilder::from_tensors(weights, dtype, device))?;
        let actual = norm
            .forward(&input)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = norm
            .forward_reference(&input)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, expected);
        assert_eq!(norm.forward(&input)?.dtype(), dtype);
        Ok(())
    }

    #[test]
    fn long_balanced_groups_match_closed_form_f64() -> candle_core::Result<()> {
        let device = &Device::Cpu;
        let low = 10.0_f32 - 0.01_f32;
        let high = 10.0_f32 + 0.01_f32;
        let mean = (f64::from(low) + f64::from(high)) / 2.0;
        let half_range = (f64::from(high) - f64::from(low)) / 2.0;
        let denominator = (half_range * half_range + 1e-6).sqrt();
        let expected_low = ((f64::from(low) - mean) / denominator) as f32;
        let expected_high = ((f64::from(high) - mean) / denominator) as f32;
        let values: Vec<f32> = (0..32 * 128 * 256)
            .map(|i| if i % 2 == 0 { low } else { high })
            .collect();
        let input = Tensor::from_vec(values, (1, 32, 128, 256), device)?;
        let actual = group_norm(0.0)?
            .forward(&input)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert!(actual.len() == 32 * 128 * 256);
        let mut max_error = 0.0_f32;
        let mut all_finite = true;
        for (i, value) in actual.iter().enumerate() {
            all_finite &= value.is_finite();
            let expected = if i % 2 == 0 {
                expected_low
            } else {
                expected_high
            };
            max_error = max_error.max((value - expected).abs());
        }
        assert!(
            all_finite,
            "group normalization produced non-finite values; max error {max_error}"
        );
        assert!(
            max_error <= 2e-5,
            "group normalization max error {max_error} exceeds 2e-5"
        );
        Ok(())
    }

    #[test]
    fn constant_groups_return_bias_and_reject_wrong_channels() -> candle_core::Result<()> {
        let device = &Device::Cpu;
        let norm = group_norm(1.25)?;
        let input = Tensor::from_vec(vec![7.0_f32; 32 * 4], (1, 32, 2, 2), device)?;
        let actual = norm.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
        assert!(actual.iter().all(|&value| value == 1.25));
        let wrong_channels = Tensor::zeros((1, 64, 2, 2), DType::F32, device)?;
        assert!(norm.forward(&wrong_channels).is_err());
        Ok(())
    }
}
