//! Fixed SDXL Phase II graph. All learned tensors are loaded once onto the
//! selected Burn device. The bridge validates public input shapes and devices.
//!
//! Bridge contract: `denoise` returns `crate::Result<Tensor<4>>` so that an
//! inconsistent internal skip topology is reported instead of panicking.
mod layers;
#[cfg(test)]
mod operator_tests;
#[cfg(test)]
mod precision_tests;
mod vae;

use crate::{Error, Result};
use burn::tensor::{Device, Tensor, activation};
use layers::{Conv, GroupNorm, Resnet, TimeEmbedding, Transformer, upsample};
use vae::Vae;

struct DownBlock {
    resnets: Vec<Resnet>,
    attention: Vec<Transformer>,
    downsample: Option<Conv>,
}
struct UpBlock {
    resnets: Vec<Resnet>,
    attention: Vec<Transformer>,
    upsample: Option<Conv>,
}
struct MidBlock {
    first: Resnet,
    attention: Transformer,
    second: Resnet,
}
struct Unet {
    input: Conv,
    time: TimeEmbedding,
    added: TimeEmbedding,
    down: Vec<DownBlock>,
    mid: MidBlock,
    up: Vec<UpBlock>,
    norm: GroupNorm,
    output: Conv,
    pooled: Tensor<2>,
    device: Device,
}

impl Unet {
    /// `sequence` `[1, 77, 2048]` and `pooled` `[1, 1280]` are the fixed prompt.
    fn new(
        w: &mut super::Weights,
        device: &Device,
        sequence: &Tensor<3>,
        pooled: Tensor<2>,
    ) -> Result<Self> {
        let channels = [320, 640, 1280];
        let depths = [0, 2, 10];
        let heads = [5, 10, 20];
        let input = Conv::new(w, "conv_in", 9, 320, 3, 1, 1, false)?;
        let time = TimeEmbedding::new(w, "time_embedding", 320, 320, 1280, device)?;
        let added = TimeEmbedding::new(w, "add_embedding", 256, 2816, 1280, device)?;
        let mut down = Vec::new();
        for i in 0..3 {
            let out = channels[i];
            let input_channels = if i == 0 { 320 } else { channels[i - 1] };
            let p = format!("down_blocks.{i}");
            let mut resnets = Vec::new();
            let mut attention = Vec::new();
            for j in 0..2 {
                let r_in = if j == 0 { input_channels } else { out };
                resnets.push(Resnet::new(
                    w,
                    &format!("{p}.resnets.{j}"),
                    r_in,
                    out,
                    Some(1280),
                    1e-5,
                )?);
                if depths[i] != 0 {
                    attention.push(Transformer::new(
                        w,
                        &format!("{p}.attentions.{j}"),
                        out,
                        sequence,
                        heads[i],
                        depths[i],
                    )?);
                }
            }
            let downsample = if i < 2 {
                Some(Conv::new(
                    w,
                    &format!("{p}.downsamplers.0.conv"),
                    out,
                    out,
                    3,
                    2,
                    1,
                    false,
                )?)
            } else {
                None
            };
            down.push(DownBlock {
                resnets,
                attention,
                downsample,
            });
        }
        let mid = MidBlock {
            first: Resnet::new(w, "mid_block.resnets.0", 1280, 1280, Some(1280), 1e-5)?,
            attention: Transformer::new(w, "mid_block.attentions.0", 1280, sequence, 20, 10)?,
            second: Resnet::new(w, "mid_block.resnets.1", 1280, 1280, Some(1280), 1e-5)?,
        };
        let mut up = Vec::new();
        for i in 0..3 {
            let index = 2 - i;
            let out = channels[index];
            let previous = if i == 0 { 1280 } else { channels[index + 1] };
            let input_channels = if i == 2 { 320 } else { channels[index - 1] };
            let p = format!("up_blocks.{i}");
            let mut resnets = Vec::new();
            let mut attention = Vec::new();
            for j in 0..3 {
                let skip = if j == 2 { input_channels } else { out };
                let r_in = if j == 0 { previous } else { out };
                resnets.push(Resnet::new(
                    w,
                    &format!("{p}.resnets.{j}"),
                    r_in + skip,
                    out,
                    Some(1280),
                    1e-5,
                )?);
                if depths[index] != 0 {
                    attention.push(Transformer::new(
                        w,
                        &format!("{p}.attentions.{j}"),
                        out,
                        sequence,
                        heads[index],
                        depths[index],
                    )?);
                }
            }
            let upsample = if i < 2 {
                Some(Conv::new(
                    w,
                    &format!("{p}.upsamplers.0.conv"),
                    out,
                    out,
                    3,
                    1,
                    1,
                    false,
                )?)
            } else {
                None
            };
            up.push(UpBlock {
                resnets,
                attention,
                upsample,
            });
        }
        let norm = GroupNorm::new(w, "conv_norm_out", 320, 1e-5)?;
        let output = Conv::new(w, "conv_out", 320, 5, 3, 1, 1, false)?;
        Ok(Self {
            input,
            time,
            added,
            down,
            mid,
            up,
            norm,
            output,
            pooled,
            device: device.clone(),
        })
    }
    fn forward(&self, input: Tensor<4>, timestep: usize, time_ids: Tensor<2>) -> Result<Tensor<4>> {
        let batch = input.dims()[0];
        let t = Tensor::<1>::full(
            [batch],
            timestep as f32,
            (&self.device, burn::tensor::DType::F32),
        );
        let normal = self.time.forward(self.time.project(t));
        let ids = self
            .added
            .project(time_ids.reshape([batch * 6]))
            .reshape([batch, 6 * 256]);
        let embedding = normal
            + self
                .added
                .forward(Tensor::cat(vec![self.pooled.clone(), ids], 1));
        let mut x = self.input.forward(input);
        let mut skips = vec![x.clone()];
        for block in &self.down {
            for (i, resnet) in block.resnets.iter().enumerate() {
                x = resnet.forward(x, Some(&embedding));
                if let Some(attention) = block.attention.get(i) {
                    x = attention.forward(x);
                }
                skips.push(x.clone());
            }
            if let Some(conv) = &block.downsample {
                x = conv.forward(x);
                skips.push(x.clone());
            }
        }
        x = self.mid.first.forward(x, Some(&embedding));
        x = self.mid.attention.forward(x);
        x = self.mid.second.forward(x, Some(&embedding));
        for block in &self.up {
            for (i, resnet) in block.resnets.iter().enumerate() {
                let skip = skips.pop().ok_or_else(|| {
                    Error::InvalidArtifact("WGPU UNet skip topology is inconsistent".into())
                })?;
                x = resnet.forward(Tensor::cat(vec![x, skip], 1), Some(&embedding));
                if let Some(attention) = block.attention.get(i) {
                    x = attention.forward(x);
                }
            }
            if let Some(conv) = &block.upsample {
                let next = skips.last().ok_or_else(|| {
                    Error::InvalidArtifact("WGPU UNet upsample target is missing".into())
                })?;
                let [_, _, h, w] = next.dims();
                x = conv.forward(upsample(x, [h, w]));
            }
        }
        if !skips.is_empty() {
            return Err(Error::InvalidArtifact(
                "WGPU UNet has unused skip tensors".into(),
            ));
        }
        Ok(self.output.forward(activation::silu(self.norm.forward(x))))
    }
}

/// Inference-only SDXL UNet and VAE with resident supplied model weights.
pub struct Graph {
    unet: Unet,
    vae: Vae,
}
impl Graph {
    /// Consume and shape-check every required checkpoint tensor and bind the
    /// fixed prompt conditioning.
    pub fn new(
        unet: &mut super::Weights,
        vae: &mut super::Weights,
        device: &Device,
        sequence: &Tensor<3>,
        pooled: Tensor<2>,
    ) -> Result<Self> {
        Ok(Self {
            unet: Unet::new(unet, device, sequence, pooled)?,
            vae: Vae::new(vae)?,
        })
    }
    /// Return raw posterior mean and log-variance channels.
    pub fn encode_moments(&self, image: Tensor<4>) -> Tensor<4> {
        self.vae.encode(image)
    }
    /// Return four denoising channels followed by the alpha logit.
    pub fn denoise(
        &self,
        input: Tensor<4>,
        timestep: usize,
        time_ids: Tensor<2>,
    ) -> Result<Tensor<4>> {
        self.unet.forward(input, timestep, time_ids)
    }
    /// Decode unscaled VAE latents into raw RGB values.
    pub fn decode(&self, latents: Tensor<4>) -> Tensor<4> {
        self.vae.decode(latents)
    }
}
