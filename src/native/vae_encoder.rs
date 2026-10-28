//! VAE encoder composition with bounded group-normalization reductions.

use candle_core::{D, DType, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};

use super::Architecture;
use super::conv::{BandedConv2d, conv2d};
use crate::{Result, pipeline::Checkpoint};

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
}

impl Module for StableGroupNorm {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
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

    pub(super) fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let shortcut = match &self.shortcut {
            Some(conv) => conv.forward(xs)?,
            None => xs.clone(),
        };
        let hidden = self
            .conv1
            .forward(&nn::ops::silu(&self.norm1.forward(xs)?)?)?;
        let hidden = self
            .conv2
            .forward(&nn::ops::silu(&self.norm2.forward(&hidden)?)?)?;
        shortcut + hidden
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
            xs = resnet.forward(&xs)?;
        }
        checkpoint()?;
        Ok(match &self.downsample {
            Some(conv) => conv.forward(&xs.pad_with_zeros(D::Minus1, 0, 1)?.pad_with_zeros(
                D::Minus2,
                0,
                1,
            )?)?,
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

    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let (batch, channels, height, width) = input.dims4()?;
        let tokens = height
            .checked_mul(width)
            .ok_or_else(|| candle_core::Error::Msg("VAE attention shape overflows".into()))?;
        let xs = self
            .norm
            .forward(input)?
            .reshape((batch, channels, tokens))?
            .transpose(1, 2)?;
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
        let scores = (query * scale)?.matmul(&(key.t()? * scale)?)?;
        let probs = nn::ops::softmax(&scores, D::Minus1)?;
        let xs = probs
            .matmul(&value)?
            .to_dtype(input.dtype())?
            .transpose(1, 2)?
            .contiguous()?
            .flatten_from(D::Minus2)?;
        let xs = self
            .output
            .forward(&xs)?
            .t()?
            .reshape((batch, channels, height, width))?;
        xs + input
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

    pub(super) fn forward(
        &self,
        xs: &Tensor,
        _temb: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        self.second
            .forward(&self.attention.forward(&self.first.forward(xs)?)?)
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
        let mut xs = self.conv_in.forward(image)?;
        for block in &self.down {
            xs = block.forward_checked(&xs, checkpoint)?;
        }
        checkpoint()?;
        xs = self.mid.forward(&xs, None)?;
        xs = nn::ops::silu(&self.norm_out.forward(&xs)?)?;
        Ok(self.quant_conv.forward(&self.conv_out.forward(&xs)?)?)
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
