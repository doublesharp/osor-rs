//! Native SDXL layers on Candle 0.11. Block orchestration was adapted from
//! Candle's stable-diffusion UNet and VAE (candle-transformers 0.9.1, MIT/Apache-2.0):
//! <https://github.com/huggingface/candle/tree/0.9.1/candle-transformers/src/models/stable_diffusion>
//! SDXL's additional text/time embedding is applied before every UNet block.
//!
//! Native attention slices query rows to bound each F32 score/probability
//! tensor to 256 MiB. This excludes weights, other activations, and allocator
//! caches. Cancellation checkpoints run between transformer operations,
//! attention slices, and convolution bands. Individual device kernels and
//! opaque Candle blocks still complete before cancellation can be observed.

mod attention;
mod columns;
mod conv;
mod cross_blocks;
#[cfg(all(test, feature = "metal", target_os = "macos"))]
mod metal_bench;
#[cfg(test)]
mod precision_tests;
mod resnet;
mod spatial;
mod transformer;
mod unet_blocks;
mod unet_conv;
mod vae_decoder;
mod vae_encoder;

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};
#[cfg(test)]
use candle_transformers::models::stable_diffusion::unet_2d_blocks::{DownBlock2D, UpBlock2D};
use candle_transformers::models::stable_diffusion::{
    embeddings::{TimestepEmbedding, Timesteps},
    unet_2d_blocks::{DownBlock2DConfig, UpBlock2DConfig},
};

use crate::{
    Error, Result,
    pipeline::{Checkpoint, IMAGE_AXIS_MULTIPLE, MAX_BATCH, MAX_IMAGE_AXIS, MIN_IMAGE_AXIS},
};
use cross_blocks::{CrossConfig, CrossDown, CrossMid, CrossUp};
use unet_blocks::{PlainDown, PlainUp};
use unet_conv::UnetConv2d;
use vae_decoder::VaeDecoder;
use vae_encoder::VaeEncoder;

const MIN_LATENT_AXIS: usize = MIN_IMAGE_AXIS / IMAGE_AXIS_MULTIPLE;
const MAX_LATENT_AXIS: usize = MAX_IMAGE_AXIS / IMAGE_AXIS_MULTIPLE;

#[derive(Clone, Copy)]
struct BlockSpec {
    channels: usize,
    depth: Option<usize>,
    heads: usize,
}

#[derive(Clone)]
struct Architecture {
    blocks: [BlockSpec; 3],
    context: usize,
    pooled: usize,
    time_projection: usize,
    layers: usize,
    vae_channels: [usize; 4],
    vae_layers: usize,
}

impl Architecture {
    fn sdxl() -> Self {
        Self {
            blocks: [
                BlockSpec {
                    channels: 320,
                    depth: None,
                    heads: 5,
                },
                BlockSpec {
                    channels: 640,
                    depth: Some(2),
                    heads: 10,
                },
                BlockSpec {
                    channels: 1280,
                    depth: Some(10),
                    heads: 20,
                },
            ],
            context: 2048,
            pooled: 1280,
            time_projection: 256,
            layers: 2,
            vae_channels: [128, 256, 512, 512],
            vae_layers: 2,
        }
    }
}

#[derive(Debug)]
enum Down {
    Plain(PlainDown),
    Cross(CrossDown),
}

#[derive(Debug)]
enum Up {
    Plain(PlainUp),
    Cross(CrossUp),
}

struct Unet {
    conv_in: UnetConv2d,
    time_proj: Timesteps,
    time_embedding: TimestepEmbedding,
    add_time_proj: Timesteps,
    add_embedding: TimestepEmbedding,
    down: Vec<Down>,
    mid: CrossMid,
    up: Vec<Up>,
    norm_out: nn::GroupNorm,
    conv_out: UnetConv2d,
    pooled: Tensor,
    arch: Architecture,
}

impl Unet {
    /// `context` is the fixed `[1, tokens, context]` prompt sequence and
    /// `pooled` the fixed `[1, pooled]` embedding; both are bound at load.
    fn new(
        vs: VarBuilder<'_>,
        arch: Architecture,
        context: &Tensor,
        pooled: &Tensor,
    ) -> candle_core::Result<Self> {
        let (context_batch, tokens, context_width) = context.dims3()?;
        if context_batch != 1
            || tokens == 0
            || tokens > 1024
            || context_width != arch.context
            || pooled.dims() != [1, arch.pooled]
        {
            candle_core::bail!(
                "fixed conditioning must be [1, 1..=1024, {}] and [1, {}]",
                arch.context,
                arch.pooled
            )
        }
        let blocks = arch.blocks;
        let base = blocks[0].channels;
        let time_dim = base * 4;
        let conv_in = UnetConv2d::new(9, base, 3, 1, 1, vs.pp("conv_in"))?;
        let time_proj = Timesteps::new(base, true, 0.);
        let time_embedding = TimestepEmbedding::new(vs.pp("time_embedding"), base, time_dim)?;
        let add_time_proj = Timesteps::new(arch.time_projection, true, 0.);
        let add_embedding = TimestepEmbedding::new(
            vs.pp("add_embedding"),
            arch.pooled + 6 * arch.time_projection,
            time_dim,
        )?;
        let mut down = Vec::with_capacity(blocks.len());
        for (i, spec) in blocks.iter().enumerate() {
            let in_channels = if i == 0 { base } else { blocks[i - 1].channels };
            let cfg = DownBlock2DConfig {
                num_layers: arch.layers,
                resnet_eps: 1e-5,
                resnet_groups: 32,
                add_downsample: i + 1 < blocks.len(),
                downsample_padding: 1,
                ..Default::default()
            };
            let path = vs.pp("down_blocks").pp(i);
            let block = match spec.depth {
                Some(depth) => Down::Cross(CrossDown::new(
                    path,
                    in_channels,
                    CrossConfig {
                        layers: arch.layers,
                        channels: spec.channels,
                        time_dim,
                        heads: spec.heads,
                        depth,
                    },
                    i + 1 < blocks.len(),
                    context,
                )?),
                None => Down::Plain(PlainDown::new(
                    path,
                    in_channels,
                    spec.channels,
                    Some(time_dim),
                    cfg,
                )?),
            };
            down.push(block);
        }
        let last = blocks[2];
        let mid_depth = last.depth.ok_or_else(|| {
            candle_core::Error::Msg("UNet middle block requires cross attention".into())
        })?;
        let mid = CrossMid::new(
            vs.pp("mid_block"),
            CrossConfig {
                layers: 1,
                channels: last.channels,
                time_dim,
                heads: last.heads,
                depth: mid_depth,
            },
            context,
        )?;
        let mut up = Vec::with_capacity(blocks.len());
        for (i, spec) in blocks.iter().rev().enumerate() {
            let prev = if i == 0 {
                last.channels
            } else {
                blocks[blocks.len() - i].channels
            };
            let in_channels = if i + 1 == blocks.len() {
                base
            } else {
                blocks[blocks.len() - i - 2].channels
            };
            let cfg = UpBlock2DConfig {
                num_layers: arch.layers + 1,
                resnet_eps: 1e-5,
                resnet_groups: 32,
                add_upsample: i + 1 < blocks.len(),
                ..Default::default()
            };
            let path = vs.pp("up_blocks").pp(i);
            let block = match spec.depth {
                Some(depth) => Up::Cross(CrossUp::new(
                    path,
                    in_channels,
                    prev,
                    CrossConfig {
                        layers: arch.layers + 1,
                        channels: spec.channels,
                        time_dim,
                        heads: spec.heads,
                        depth,
                    },
                    i + 1 < blocks.len(),
                    context,
                )?),
                None => Up::Plain(PlainUp::new(
                    path,
                    in_channels,
                    prev,
                    spec.channels,
                    Some(time_dim),
                    cfg,
                )?),
            };
            up.push(block);
        }
        let norm_out = nn::group_norm(32, base, 1e-5, vs.pp("conv_norm_out"))?;
        let conv_out = UnetConv2d::new(base, 5, 3, 1, 1, vs.pp("conv_out"))?;
        Ok(Self {
            conv_in,
            time_proj,
            time_embedding,
            add_time_proj,
            add_embedding,
            down,
            mid,
            up,
            norm_out,
            conv_out,
            pooled: pooled.clone(),
            arch,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        timestep: usize,
        time_ids: &Tensor,
        checkpoint: Checkpoint<'_>,
    ) -> Result<Tensor> {
        let (batch, channels, h, w) = input.dims4().map_err(|_| {
            Error::InvalidInput("UNet input must have shape [batch,9,height,width]".into())
        })?;
        if batch == 0
            || batch > MAX_BATCH
            || channels != 9
            || !(MIN_LATENT_AXIS..=MAX_LATENT_AXIS).contains(&h)
            || !(MIN_LATENT_AXIS..=MAX_LATENT_AXIS).contains(&w)
        {
            return Err(Error::InvalidInput(format!(
                "UNet expects batch 1..={MAX_BATCH}, 9 channels and spatial dimensions {MIN_LATENT_AXIS}..={MAX_LATENT_AXIS}"
            )));
        }
        if time_ids.dims() != [batch, 6] || !input.device().same_device(time_ids.device()) {
            return Err(Error::InvalidInput(
                "time_ids must have shape [batch, 6] on the UNet input device".into(),
            ));
        }
        if self.pooled.dtype() != input.dtype() || !input.device().same_device(self.pooled.device())
        {
            return Err(Error::InvalidInput(
                "UNet input must match the model dtype and device".into(),
            ));
        }
        let pooled = if batch == 1 {
            self.pooled.clone()
        } else {
            self.pooled.repeat((batch, 1))?
        };
        // Diffusers computes sinusoidal projections in float32, then casts to the model dtype.
        let timesteps = (Tensor::ones(batch, DType::F32, input.device())? * timestep as f64)?;
        let emb = self.time_embedding.forward(
            &self
                .time_proj
                .forward(&timesteps)?
                .to_dtype(input.dtype())?,
        )?;
        let ids = time_ids.to_dtype(DType::F32)?.reshape((batch * 6,))?;
        let ids = self
            .add_time_proj
            .forward(&ids)?
            .reshape((batch, 6 * self.arch.time_projection))?
            .to_dtype(input.dtype())?;
        let added = self
            .add_embedding
            .forward(&Tensor::cat(&[&pooled, &ids], 1)?)?;
        let emb = (&emb + &added)?;
        let mut xs = self.conv_in.forward_checked(input, checkpoint)?;
        let mut residuals = vec![xs.clone()];
        for block in &self.down {
            checkpoint()?;
            let (next, saved) = match block {
                Down::Plain(b) => b.forward_checked(&xs, Some(&emb), checkpoint)?,
                Down::Cross(b) => b.forward_checked(&xs, Some(&emb), checkpoint)?,
            };
            xs = next;
            residuals.extend(saved);
        }
        checkpoint()?;
        xs = self.mid.forward_checked(&xs, Some(&emb), checkpoint)?;
        for (index, block) in self.up.iter().enumerate() {
            checkpoint()?;
            let count = match block {
                Up::Plain(b) => b.residual_count(),
                Up::Cross(b) => b.residual_count(),
            };
            let start = residuals.len().checked_sub(count).ok_or_else(|| {
                Error::InvalidArtifact("UNet residual topology is inconsistent".into())
            })?;
            let saved = residuals.split_off(start);
            // The next skip connection determines the upsampled size. Downsampling
            // rounds odd axes up, so doubling the current size may be too large.
            let upsample_size = if index + 1 < self.up.len() {
                let next = residuals.last().ok_or_else(|| {
                    Error::InvalidArtifact("UNet residual topology is inconsistent".into())
                })?;
                let (_, _, height, width) = next.dims4()?;
                Some((height, width))
            } else {
                None
            };
            xs = match block {
                Up::Plain(b) => {
                    b.forward_checked(&xs, &saved, Some(&emb), upsample_size, checkpoint)?
                }
                Up::Cross(b) => {
                    b.forward_checked(&xs, &saved, Some(&emb), upsample_size, checkpoint)?
                }
            };
        }
        if !residuals.is_empty() {
            return Err(Error::InvalidArtifact(
                "UNet residual topology left unused tensors".into(),
            ));
        }
        self.conv_out
            .forward_checked(&nn::ops::silu(&self.norm_out.forward(&xs)?)?, checkpoint)
    }
}

/// SDXL inpainting UNet and VAE, constructed solely from supplied model tensors.
pub struct NativeModel {
    unet: Unet,
    encoder: VaeEncoder,
    decoder: VaeDecoder,
    dtype: DType,
    device: Device,
}

impl NativeModel {
    /// Construct the fixed released architecture; every requested weight is required and shape checked.
    ///
    /// `sequence` (`[1, 77, 2048]`) and `pooled` (`[1, 1280]`) are the bundle's
    /// fixed prompt embeddings. Cross-attention keys and values are computed
    /// from them once, and the corresponding projection weights are released.
    pub fn new(
        unet: HashMap<String, Tensor>,
        vae: HashMap<String, Tensor>,
        sequence: &Tensor,
        pooled: &Tensor,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::InvalidInput(
                "native model dtype must be F32, F16, or BF16".into(),
            ));
        }
        let arch = Architecture::sdxl();
        if sequence.dims() != [1, 77, arch.context] || pooled.dims() != [1, arch.pooled] {
            return Err(Error::InvalidInput(format!(
                "prompt embeddings must have shapes [1, 77, {}] and [1, {}]",
                arch.context, arch.pooled
            )));
        }
        let sequence = sequence.to_device(device)?.to_dtype(dtype)?;
        let pooled = pooled.to_device(device)?.to_dtype(dtype)?;
        let unet_vs = VarBuilder::from_tensors(unet, dtype, device);
        let vae_vs = VarBuilder::from_tensors(vae, dtype, device);
        Self::build(unet_vs, vae_vs, dtype, device, arch, &sequence, &pooled).map_err(|e| {
            Error::InvalidArtifact(format!("native model tensor missing or incompatible: {e}"))
        })
    }

    fn build(
        unet_vs: VarBuilder<'_>,
        vae_vs: VarBuilder<'_>,
        dtype: DType,
        device: &Device,
        arch: Architecture,
        sequence: &Tensor,
        pooled: &Tensor,
    ) -> candle_core::Result<Self> {
        let unet = Unet::new(unet_vs, arch.clone(), sequence, pooled)?;
        let encoder = VaeEncoder::new(vae_vs.clone(), &arch)?;
        let decoder = VaeDecoder::new(vae_vs, &arch)?;
        Ok(Self {
            unet,
            encoder,
            decoder,
            dtype,
            device: device.clone(),
        })
    }

    fn validate_image(
        &self,
        tensor: &Tensor,
        channels: usize,
        min: usize,
        max: usize,
        divisible: usize,
        name: &str,
    ) -> Result<()> {
        let (batch, actual, h, w) = tensor
            .dims4()
            .map_err(|_| Error::InvalidInput(format!("{name} must be rank four")))?;
        if batch == 0
            || batch > MAX_BATCH
            || actual != channels
            || h < min
            || w < min
            || h > max
            || w > max
            || !h.is_multiple_of(divisible)
            || !w.is_multiple_of(divisible)
        {
            return Err(Error::InvalidInput(format!(
                "invalid {name} shape; batch must be 1..=4, channels {channels}, dimensions {min}..={max} divisible by {divisible}"
            )));
        }
        if tensor.dtype() != self.dtype || !tensor.device().same_device(&self.device) {
            return Err(Error::InvalidInput(format!(
                "{name} dtype and device must match the model"
            )));
        }
        Ok(())
    }

    /// Return the unmodified eight-channel mean/log-variance posterior parameters.
    pub fn encode_moments(&self, image: &Tensor) -> Result<Tensor> {
        self.encode_moments_checked(image, &|| Ok(()))
    }

    pub(crate) fn encode_moments_checked(
        &self,
        image: &Tensor,
        checkpoint: Checkpoint<'_>,
    ) -> Result<Tensor> {
        self.validate_image(
            image,
            3,
            MIN_IMAGE_AXIS,
            MAX_IMAGE_AXIS,
            IMAGE_AXIS_MULTIPLE,
            "image",
        )?;
        self.encoder.forward(image, checkpoint)
    }

    /// Run the five-channel SDXL UNet with the bound prompt conditioning and
    /// per-call SDXL time IDs.
    pub fn denoise(&self, input: &Tensor, timestep: usize, time_ids: &Tensor) -> Result<Tensor> {
        self.denoise_checked(input, timestep, time_ids, &|| Ok(()))
    }

    pub(crate) fn denoise_checked(
        &self,
        input: &Tensor,
        timestep: usize,
        time_ids: &Tensor,
        checkpoint: Checkpoint<'_>,
    ) -> Result<Tensor> {
        self.validate_image(input, 9, MIN_LATENT_AXIS, MAX_LATENT_AXIS, 1, "UNet input")?;
        self.unet.forward(input, timestep, time_ids, checkpoint)
    }

    /// Decode unscaled four-channel VAE latents into the raw RGB output.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        self.decode_checked(latents, &|| Ok(()))
    }

    pub(crate) fn decode_checked(
        &self,
        latents: &Tensor,
        checkpoint: Checkpoint<'_>,
    ) -> Result<Tensor> {
        self.validate_image(latents, 4, 1, MAX_LATENT_AXIS, 1, "latents")?;
        self.decoder.forward(latents, checkpoint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tiny {
        unet: nn::VarMap,
        vae: nn::VarMap,
    }

    impl Tiny {
        fn new() -> Self {
            Self {
                unet: nn::VarMap::new(),
                vae: nn::VarMap::new(),
            }
        }

        /// Build a model; repeated builds share the same random weights.
        fn model(&self, sequence: &Tensor, pooled: &Tensor) -> Result<NativeModel> {
            let device = Device::Cpu;
            let arch = Architecture {
                blocks: [
                    BlockSpec {
                        channels: 64,
                        depth: None,
                        heads: 1,
                    },
                    BlockSpec {
                        channels: 64,
                        depth: Some(1),
                        heads: 1,
                    },
                    BlockSpec {
                        channels: 64,
                        depth: Some(1),
                        heads: 1,
                    },
                ],
                context: 8,
                pooled: 8,
                time_projection: 8,
                layers: 1,
                vae_channels: [32, 32, 32, 32],
                vae_layers: 1,
            };
            Ok(NativeModel::build(
                VarBuilder::from_varmap(&self.unet, DType::F32, &device),
                VarBuilder::from_varmap(&self.vae, DType::F32, &device),
                DType::F32,
                &device,
                arch,
                sequence,
                pooled,
            )?)
        }
    }

    fn conditioning() -> Result<(Tensor, Tensor)> {
        let device = &Device::Cpu;
        Ok((
            Tensor::randn(0f32, 1f32, (1, 2, 8), device)?,
            Tensor::randn(0f32, 1f32, (1, 8), device)?,
        ))
    }

    fn tiny_model() -> Result<NativeModel> {
        let (sequence, pooled) = conditioning()?;
        Tiny::new().model(&sequence, &pooled)
    }

    fn differs(a: &Tensor, b: &Tensor) -> Result<bool> {
        Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()? > 1e-7)
    }

    #[test]
    fn unet_input_convolution_tracks_f64_reference() -> Result<()> {
        let mut tiny = Tiny::new();
        let (sequence, pooled) = conditioning()?;
        let model = tiny.model(&sequence, &pooled)?;
        let device = &Device::Cpu;
        let values = |count: usize, multiplier: usize, modulus: usize| {
            (0..count)
                .map(|i| ((i * multiplier % modulus) as f32 - (modulus / 2) as f32) / 37.)
                .collect::<Vec<_>>()
        };
        let weight = Tensor::from_vec(values(64 * 9 * 9, 61, 101), (64, 9, 3, 3), device)?;
        let bias = Tensor::zeros(64, DType::F32, device)?;
        tiny.unet.set_one("conv_in.weight", &weight)?;
        tiny.unet.set_one("conv_in.bias", &bias)?;
        let mut failures = Vec::new();
        for batch in [1, 2] {
            let input =
                Tensor::from_vec(values(batch * 9 * 5 * 7, 43, 103), (batch, 9, 5, 7), device)?;
            let reference = nn::Conv2d::new(
                weight.to_dtype(DType::F64)?,
                Some(bias.to_dtype(DType::F64)?),
                nn::Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
            )
            .forward(&input.to_dtype(DType::F64)?)?;
            let actual = model.unet.conv_in.forward(&input)?.to_dtype(DType::F64)?;
            let max_error = (&actual - &reference)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f64>()?;
            eprintln!("batch {batch} maximum F64 error: {max_error}");
            if max_error > 6.2e-6 {
                failures.push((batch, max_error));
            }
        }
        assert!(failures.is_empty(), "F64 convolution errors: {failures:?}");
        Ok(())
    }

    #[test]
    fn plain_unet_blocks_match_candle_for_one_and_two_images() -> Result<()> {
        let tiny = Tiny::new();
        let (sequence, pooled) = conditioning()?;
        let model = tiny.model(&sequence, &pooled)?;
        let device = &Device::Cpu;
        let vs = VarBuilder::from_varmap(&tiny.unet, DType::F32, device);
        let down_reference = DownBlock2D::new(
            vs.pp("down_blocks.0"),
            64,
            64,
            Some(256),
            DownBlock2DConfig {
                num_layers: 1,
                resnet_eps: 1e-5,
                ..Default::default()
            },
        )?;
        let up_reference = UpBlock2D::new(
            vs.pp("up_blocks.2"),
            64,
            64,
            64,
            Some(256),
            UpBlock2DConfig {
                num_layers: 2,
                resnet_eps: 1e-5,
                add_upsample: false,
                ..Default::default()
            },
        )?;
        let Some(Down::Plain(down)) = model.unet.down.first() else {
            return Err(Error::InvalidArtifact("missing plain down block".into()));
        };
        let Some(Up::Plain(up)) = model.unet.up.last() else {
            return Err(Error::InvalidArtifact("missing plain up block".into()));
        };
        let max_error = |a: &Tensor, b: &Tensor| -> Result<f32> {
            Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
        };
        for batch in [1, 2] {
            let input = Tensor::randn(0f32, 1f32, (batch, 64, 5, 7), device)?;
            let temb = Tensor::randn(0f32, 1f32, (batch, 256), device)?;
            let (actual, saved) = down.forward(&input, Some(&temb))?;
            let (expected, reference_saved) = down_reference.forward(&input, Some(&temb))?;
            assert_eq!(actual.dims(), &[batch, 64, 3, 4]);
            assert_eq!(saved.len(), 2);
            assert_eq!(saved.len(), reference_saved.len());
            assert!(max_error(&actual, &expected)? <= 3e-5);
            for (actual, expected) in saved.iter().zip(&reference_saved) {
                assert!(max_error(actual, expected)? <= 3e-5);
            }
            let skips = vec![input.clone(), input.clone()];
            let actual = up.forward(&input, &skips, Some(&temb), None)?;
            let expected = up_reference.forward(&input, &skips, Some(&temb), None)?;
            assert_eq!(actual.dims(), &[batch, 64, 5, 7]);
            assert!(max_error(&actual, &expected)? <= 3e-5);
        }
        Ok(())
    }

    #[test]
    fn tiny_complete_graph_uses_all_conditioning_and_preserves_shapes() -> Result<()> {
        let tiny = Tiny::new();
        let (sequence, pooled) = conditioning()?;
        let m = tiny.model(&sequence, &pooled)?;
        let device = &Device::Cpu;
        let image = Tensor::randn(0f32, 1f32, (1, 3, 32, 32), device)?;
        let moments = m.encode_moments(&image)?;
        assert_eq!(moments.dims(), &[1, 8, 4, 4]);
        let latent = Tensor::randn(0f32, 1f32, (1, 4, 4, 4), device)?;
        assert_eq!(m.decode(&latent)?.dims(), &[1, 3, 32, 32]);
        let input = Tensor::randn(0f32, 1f32, (1, 9, 4, 4), device)?;
        let ids = Tensor::new(&[[32f32, 32., 0., 0., 32., 32.]], device)?;
        let original = m.denoise(&input, 400, &ids)?;
        assert_eq!(original.dims(), &[1, 5, 4, 4]);
        assert!(differs(&original, &m.denoise(&input, 401, &ids)?)?);
        assert!(differs(&original, &m.denoise(&input, 400, &(&ids + 1.)?)?)?);
        let rebuilt = tiny.model(&sequence, &pooled)?;
        assert!(!differs(&original, &rebuilt.denoise(&input, 400, &ids)?)?);
        let other_sequence = tiny.model(&(&sequence * 2.)?, &pooled)?;
        assert!(differs(
            &original,
            &other_sequence.denoise(&input, 400, &ids)?
        )?);
        let other_pooled = tiny.model(&sequence, &(&pooled * 2.)?)?;
        assert!(differs(
            &original,
            &other_pooled.denoise(&input, 400, &ids)?
        )?);
        // Batch entries see the same bound prompt as a batch-one call. Batched
        // matrix products may round differently, so this is not bitwise.
        let batch = Tensor::cat(&[&input, &input], 0)?;
        let batch_ids = Tensor::cat(&[&ids, &ids], 0)?;
        let batched = m.denoise(&batch, 400, &batch_ids)?;
        for index in 0..2 {
            let difference = (&original - batched.narrow(0, index, 1)?)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f32>()?;
            assert!(
                difference <= 1e-5,
                "batch entry {index} differs by {difference}"
            );
        }
        Ok(())
    }

    #[test]
    fn checkpoints_run_between_blocks_and_stop_each_stage() -> Result<()> {
        use std::cell::Cell;
        let m = tiny_model()?;
        let device = &Device::Cpu;
        let image = Tensor::randn(0f32, 1f32, (1, 3, 32, 32), device)?;
        let latent = Tensor::randn(0f32, 1f32, (1, 4, 4, 4), device)?;
        let input = Tensor::randn(0f32, 1f32, (1, 9, 4, 4), device)?;
        let ids = Tensor::new(&[[32f32, 32., 0., 0., 32., 32.]], device)?;
        let count = Cell::new(0_usize);
        let counting = || {
            count.set(count.get() + 1);
            Ok(())
        };
        m.encode_moments_checked(&image, &counting)?;
        let encode = count.take();
        m.denoise_checked(&input, 400, &ids, &counting)?;
        let denoise = count.take();
        m.decode_checked(&latent, &counting)?;
        let decode = count.take();
        // Four encoder down blocks, three UNet down and up blocks plus the
        // middle, and four decoder up blocks each give several checkpoints.
        assert!(encode >= 8, "encoder checkpoints {encode}");
        assert!(denoise >= 7, "UNet checkpoints {denoise}");
        assert!(decode >= 12, "decoder checkpoints {decode}");
        let calls = Cell::new(0_usize);
        let cancel_second = || {
            calls.set(calls.get() + 1);
            if calls.get() >= 2 {
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            m.encode_moments_checked(&image, &cancel_second),
            Err(Error::Cancelled)
        ));
        calls.set(0);
        assert!(matches!(
            m.denoise_checked(&input, 400, &ids, &cancel_second),
            Err(Error::Cancelled)
        ));
        calls.set(0);
        assert!(matches!(
            m.decode_checked(&latent, &cancel_second),
            Err(Error::Cancelled)
        ));
        Ok(())
    }

    #[test]
    fn tiny_complete_graph_preserves_nonsquare_odd_latent_shape() -> Result<()> {
        let m = tiny_model()?;
        let device = &Device::Cpu;
        let image = Tensor::randn(0f32, 1f32, (1, 3, 40, 56), device)?;
        let moments = m.encode_moments(&image)?;
        assert_eq!(moments.dims(), &[1, 8, 5, 7]);
        let latent = moments.narrow(1, 0, 4)?.contiguous()?;
        assert_eq!(m.decode(&latent)?.dims(), &[1, 3, 40, 56]);
        let input = Tensor::randn(0f32, 1f32, (1, 9, 5, 7), device)?;
        let ids = Tensor::new(&[[40f32, 56., 0., 0., 40., 56.]], device)?;
        assert_eq!(m.denoise(&input, 400, &ids)?.dims(), &[1, 5, 5, 7]);
        Ok(())
    }

    #[test]
    fn vae_accepts_diffusers_attention_names_and_rejects_missing_projection() -> Result<()> {
        let device = Device::Cpu;
        let variables = nn::VarMap::new();
        let arch = Architecture {
            blocks: [BlockSpec {
                channels: 32,
                depth: None,
                heads: 1,
            }; 3],
            context: 8,
            pooled: 8,
            time_projection: 8,
            layers: 1,
            vae_channels: [32, 32, 32, 32],
            vae_layers: 1,
        };
        let reference = VaeEncoder::new(
            VarBuilder::from_varmap(&variables, DType::F32, &device),
            &arch,
        )?;
        let guard = variables
            .data()
            .lock()
            .map_err(|_| Error::InvalidArtifact("test variable lock poisoned".into()))?;
        let mut renamed = HashMap::with_capacity(guard.len());
        for (name, var) in guard.iter() {
            let name = name
                .replace(".query.", ".to_q.")
                .replace(".key.", ".to_k.")
                .replace(".value.", ".to_v.")
                .replace(".proj_attn.", ".to_out.0.");
            renamed.insert(name, var.as_tensor().clone());
        }
        drop(guard);
        let converted = VaeEncoder::new(
            VarBuilder::from_tensors(renamed.clone(), DType::F32, &device),
            &arch,
        )?;
        let image = Tensor::randn(0f32, 1f32, (1, 3, 32, 32), &device)?;
        assert!(!differs(
            &reference.forward(&image, &|| Ok(()))?,
            &converted.forward(&image, &|| Ok(()))?
        )?);
        renamed.remove("encoder.mid_block.attentions.0.to_v.weight");
        assert!(
            VaeEncoder::new(
                VarBuilder::from_tensors(renamed, DType::F32, &device),
                &arch
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_bad_shapes_and_missing_weights() -> Result<()> {
        let device = &Device::Cpu;
        let sequence = Tensor::zeros((1, 77, 2048), DType::F32, device)?;
        let pooled = Tensor::zeros((1, 1280), DType::F32, device)?;
        assert!(matches!(
            NativeModel::new(
                HashMap::new(),
                HashMap::new(),
                &sequence,
                &pooled,
                DType::F32,
                device
            ),
            Err(Error::InvalidArtifact(_))
        ));
        assert!(matches!(
            NativeModel::new(
                HashMap::new(),
                HashMap::new(),
                &sequence.narrow(1, 0, 76)?,
                &pooled,
                DType::F32,
                device
            ),
            Err(Error::InvalidInput(_))
        ));
        let m = tiny_model()?;
        let image = Tensor::zeros((1, 3, 31, 32), DType::F32, device)?;
        assert!(matches!(
            m.encode_moments(&image),
            Err(Error::InvalidInput(_))
        ));
        let latents = Tensor::zeros((1, 3, 4, 4), DType::F32, device)?;
        assert!(matches!(m.decode(&latents), Err(Error::InvalidInput(_))));
        let input = Tensor::zeros((1, 9, 3, 4), DType::F32, device)?;
        let ids = Tensor::zeros((1, 6), DType::F32, device)?;
        assert!(matches!(
            m.denoise(&input, 400, &ids),
            Err(Error::InvalidInput(_))
        ));
        let input = Tensor::zeros((1, 9, 4, 4), DType::F32, device)?;
        let invalid_ids = Tensor::zeros((2, 6), DType::F32, device)?;
        assert!(matches!(
            m.denoise(&input, 400, &invalid_ids),
            Err(Error::InvalidInput(_))
        ));
        Ok(())
    }
}
