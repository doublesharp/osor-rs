//! VAE decoder composition with bounded group-normalization reductions and
//! convolution scratch. Block topology and tensor names follow Candle's
//! `AutoEncoderKL` decoder (candle-transformers 0.9.1, MIT/Apache-2.0).

use candle_core::Tensor;
use candle_nn::{self as nn, Module, VarBuilder};

use super::Architecture;
use super::conv::{BandedConv2d, conv2d};
use super::vae_encoder::{MidBlock, Resnet, StableGroupNorm};
use crate::Result;

struct UpBlock {
    resnets: Vec<Resnet>,
    upsample: Option<BandedConv2d>,
}

impl UpBlock {
    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let mut xs = input.clone();
        for resnet in &self.resnets {
            xs = resnet.forward(&xs)?;
        }
        match &self.upsample {
            Some(conv) => {
                let (_, _, height, width) = xs.dims4()?;
                let height = height
                    .checked_mul(2)
                    .ok_or_else(|| candle_core::Error::Msg("upsample height overflow".into()))?;
                let width = width
                    .checked_mul(2)
                    .ok_or_else(|| candle_core::Error::Msg("upsample width overflow".into()))?;
                conv.forward(&xs.upsample_nearest2d(height, width)?)
            }
            None => Ok(xs),
        }
    }
}

pub(super) struct VaeDecoder {
    post_quant_conv: BandedConv2d,
    conv_in: BandedConv2d,
    mid: MidBlock,
    up: Vec<UpBlock>,
    norm_out: StableGroupNorm,
    conv_out: BandedConv2d,
}

impl VaeDecoder {
    pub(super) fn new(vs: VarBuilder<'_>, arch: &Architecture) -> candle_core::Result<Self> {
        let channels = arch.vae_channels;
        let dec = vs.pp("decoder");
        let reversed: Vec<usize> = channels.iter().rev().copied().collect();
        let last = reversed[0];
        let mut up = Vec::with_capacity(reversed.len());
        for (i, &output) in reversed.iter().enumerate() {
            let input = if i == 0 { last } else { reversed[i - 1] };
            let path = dec.pp("up_blocks").pp(i);
            let mut resnets = Vec::with_capacity(arch.vae_layers + 1);
            for j in 0..=arch.vae_layers {
                resnets.push(Resnet::new(
                    path.pp("resnets").pp(j),
                    if j == 0 { input } else { output },
                    output,
                )?);
            }
            let upsample = if i + 1 < reversed.len() {
                Some(conv2d(
                    output,
                    output,
                    3,
                    1,
                    1,
                    path.pp("upsamplers").pp(0).pp("conv"),
                )?)
            } else {
                None
            };
            up.push(UpBlock { resnets, upsample });
        }
        Ok(Self {
            post_quant_conv: conv2d(4, 4, 1, 1, 0, vs.pp("post_quant_conv"))?,
            conv_in: conv2d(4, last, 3, 1, 1, dec.pp("conv_in"))?,
            mid: MidBlock::new(dec.pp("mid_block"), last)?,
            up,
            norm_out: StableGroupNorm::new(channels[0], dec.pp("conv_norm_out"))?,
            conv_out: conv2d(channels[0], 3, 3, 1, 1, dec.pp("conv_out"))?,
        })
    }

    /// Decode unscaled latents into raw RGB in approximately [-1, 1].
    pub(super) fn forward(&self, latents: &Tensor) -> Result<Tensor> {
        let xs = self
            .conv_in
            .forward(&self.post_quant_conv.forward(latents)?)?;
        let mut xs = self.mid.forward(&xs, None)?;
        for block in &self.up {
            xs = block.forward(&xs)?;
        }
        let xs = nn::ops::silu(&self.norm_out.forward(&xs)?)?;
        Ok(self.conv_out.forward(&xs)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::BlockSpec;
    use candle_core::{DType, Device};
    use candle_transformers::models::stable_diffusion::vae::{AutoEncoderKL, AutoEncoderKLConfig};

    #[test]
    fn decoder_matches_candle_autoencoder_with_single_pass_groups() -> Result<()> {
        let device = &Device::Cpu;
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
        let variables = nn::VarMap::new();
        let candle = AutoEncoderKL::new(
            VarBuilder::from_varmap(&variables, DType::F32, device),
            3,
            3,
            AutoEncoderKLConfig {
                block_out_channels: arch.vae_channels.to_vec(),
                layers_per_block: arch.vae_layers,
                latent_channels: 4,
                norm_num_groups: 32,
                use_quant_conv: true,
                use_post_quant_conv: true,
            },
        )?;
        let ours = VaeDecoder::new(
            VarBuilder::from_varmap(&variables, DType::F32, device),
            &arch,
        )?;
        // A 2x2 latent decodes to 16x16; with one channel per group every
        // normalization reduces at most 256 elements in a single pass.
        let latents = Tensor::randn(0f32, 1f32, (2, 4, 2, 2), device)?;
        let expected = candle.decode(&latents)?;
        let actual = ours.forward(&latents)?;
        assert_eq!(actual.dims(), &[2, 3, 16, 16]);
        let difference = (actual - expected)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            difference <= 1e-6,
            "decoder differs from Candle by {difference}"
        );
        Ok(())
    }

    #[test]
    fn decoder_rejects_a_missing_upsampler() -> Result<()> {
        let device = &Device::Cpu;
        let arch = Architecture::sdxl();
        let variables = nn::VarMap::new();
        let _ = VaeDecoder::new(
            VarBuilder::from_varmap(&variables, DType::F32, device),
            &Architecture {
                vae_channels: [32, 32, 32, 32],
                vae_layers: 1,
                ..arch.clone()
            },
        )?;
        let mut tensors = std::collections::HashMap::new();
        for (name, var) in variables
            .data()
            .lock()
            .map_err(|_| crate::Error::InvalidArtifact("test variable lock poisoned".into()))?
            .iter()
        {
            if !name.starts_with("decoder.up_blocks.1.upsamplers.0.conv") {
                tensors.insert(name.clone(), var.as_tensor().clone());
            }
        }
        assert!(
            VaeDecoder::new(
                VarBuilder::from_tensors(tensors, DType::F32, device),
                &Architecture {
                    vae_channels: [32, 32, 32, 32],
                    vae_layers: 1,
                    ..arch
                },
            )
            .is_err()
        );
        Ok(())
    }
}
