//! SDXL cross-attention block orchestration, using Candle's public ResNet layers.
use candle_core::{Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};
use candle_transformers::models::stable_diffusion::resnet::{ResnetBlock2D, ResnetBlock2DConfig};

use super::spatial::SpatialTransformer;

#[derive(Clone, Copy)]
pub(super) struct CrossConfig {
    pub layers: usize,
    pub channels: usize,
    pub time_dim: usize,
    pub context: usize,
    pub heads: usize,
    pub depth: usize,
}

fn resnet(
    vs: VarBuilder<'_>,
    input: usize,
    output: usize,
    time_dim: usize,
) -> Result<ResnetBlock2D> {
    ResnetBlock2D::new(
        vs,
        input,
        ResnetBlock2DConfig {
            out_channels: Some(output),
            temb_channels: Some(time_dim),
            eps: 1e-5,
            groups: 32,
            ..Default::default()
        },
    )
}

fn attention(vs: VarBuilder<'_>, cfg: CrossConfig) -> Result<SpatialTransformer> {
    SpatialTransformer::new(vs, cfg.channels, cfg.heads, cfg.context, cfg.depth)
}

#[derive(Debug)]
pub(super) struct CrossDown {
    layers: Vec<(ResnetBlock2D, SpatialTransformer)>,
    downsample: Option<nn::Conv2d>,
}

impl CrossDown {
    pub fn new(
        vs: VarBuilder<'_>,
        input: usize,
        cfg: CrossConfig,
        downsample: bool,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let channels = if i == 0 { input } else { cfg.channels };
            layers.push((
                resnet(vs.pp("resnets").pp(i), channels, cfg.channels, cfg.time_dim)?,
                attention(vs.pp("attentions").pp(i), cfg)?,
            ));
        }
        let downsample = if downsample {
            Some(nn::conv2d(
                cfg.channels,
                cfg.channels,
                3,
                nn::Conv2dConfig {
                    stride: 2,
                    padding: 1,
                    ..Default::default()
                },
                vs.pp("downsamplers.0.conv"),
            )?)
        } else {
            None
        };
        Ok(Self { layers, downsample })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        temb: Option<&Tensor>,
        context: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut xs = input.clone();
        let mut saved =
            Vec::with_capacity(self.layers.len() + usize::from(self.downsample.is_some()));
        for (resnet, attention) in &self.layers {
            xs = attention.forward(&resnet.forward(&xs, temb)?, context)?;
            saved.push(xs.clone());
        }
        if let Some(conv) = &self.downsample {
            xs = conv.forward(&xs)?;
            saved.push(xs.clone());
        }
        Ok((xs, saved))
    }
}

#[derive(Debug)]
pub(super) struct CrossUp {
    layers: Vec<(ResnetBlock2D, SpatialTransformer)>,
    upsample: Option<nn::Conv2d>,
}

impl CrossUp {
    pub fn new(
        vs: VarBuilder<'_>,
        input: usize,
        previous: usize,
        cfg: CrossConfig,
        upsample: bool,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let skip = if i + 1 == cfg.layers {
                input
            } else {
                cfg.channels
            };
            let channels = if i == 0 { previous } else { cfg.channels };
            layers.push((
                resnet(
                    vs.pp("resnets").pp(i),
                    channels + skip,
                    cfg.channels,
                    cfg.time_dim,
                )?,
                attention(vs.pp("attentions").pp(i), cfg)?,
            ));
        }
        let upsample = if upsample {
            Some(nn::conv2d(
                cfg.channels,
                cfg.channels,
                3,
                nn::Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vs.pp("upsamplers.0.conv"),
            )?)
        } else {
            None
        };
        Ok(Self { layers, upsample })
    }

    pub fn residual_count(&self) -> usize {
        self.layers.len()
    }

    pub fn forward(
        &self,
        input: &Tensor,
        saved: &[Tensor],
        temb: Option<&Tensor>,
        size: Option<(usize, usize)>,
        context: Option<&Tensor>,
    ) -> Result<Tensor> {
        if saved.len() != self.layers.len() {
            candle_core::bail!("cross-attention up block received an invalid number of residuals")
        }
        let mut xs = input.clone();
        for ((resnet, attention), skip) in self.layers.iter().zip(saved.iter().rev()) {
            xs = Tensor::cat(&[&xs, skip], 1)?.contiguous()?;
            xs = attention.forward(&resnet.forward(&xs, temb)?, context)?;
        }
        if let Some(conv) = &self.upsample {
            let (height, width) = match size {
                Some(size) => size,
                None => {
                    let (_, _, h, w) = xs.dims4()?;
                    let height = h.checked_mul(2).ok_or_else(|| {
                        candle_core::Error::Msg("upsample height overflow".into())
                    })?;
                    let width = w
                        .checked_mul(2)
                        .ok_or_else(|| candle_core::Error::Msg("upsample width overflow".into()))?;
                    (height, width)
                }
            };
            xs = conv.forward(&xs.upsample_nearest2d(height, width)?)?;
        }
        Ok(xs)
    }
}

struct MidLayers {
    initial: ResnetBlock2D,
    attention: SpatialTransformer,
    final_resnet: ResnetBlock2D,
}

pub(super) struct CrossMid(MidLayers);
impl CrossMid {
    pub fn new(vs: VarBuilder<'_>, cfg: CrossConfig) -> Result<Self> {
        Ok(Self(MidLayers {
            initial: resnet(vs.pp("resnets.0"), cfg.channels, cfg.channels, cfg.time_dim)?,
            attention: attention(vs.pp("attentions.0"), cfg)?,
            final_resnet: resnet(vs.pp("resnets.1"), cfg.channels, cfg.channels, cfg.time_dim)?,
        }))
    }
    pub fn forward(
        &self,
        xs: &Tensor,
        temb: Option<&Tensor>,
        context: Option<&Tensor>,
    ) -> Result<Tensor> {
        let xs = self.0.initial.forward(xs, temb)?;
        let xs = self.0.attention.forward(&xs, context)?;
        self.0.final_resnet.forward(&xs, temb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_transformers::models::stable_diffusion::unet_2d_blocks::{
        CrossAttnDownBlock2D, CrossAttnDownBlock2DConfig, CrossAttnUpBlock2D,
        CrossAttnUpBlock2DConfig, DownBlock2DConfig, UNetMidBlock2DCrossAttn,
        UNetMidBlock2DCrossAttnConfig, UpBlock2DConfig,
    };
    use std::collections::BTreeMap;

    fn shapes(map: &nn::VarMap) -> Result<BTreeMap<String, Vec<usize>>> {
        let variables = map
            .data()
            .lock()
            .map_err(|_| candle_core::Error::Msg("variable lock poisoned".into()))?;
        Ok(variables
            .iter()
            .map(|(name, value)| (name.clone(), value.shape().dims().to_vec()))
            .collect())
    }

    #[test]
    fn cross_blocks_preserve_candle_tensor_names_and_shapes() -> Result<()> {
        let device = &Device::Cpu;
        for depth in [2, 10] {
            let ours = nn::VarMap::new();
            let candle = nn::VarMap::new();
            let a = VarBuilder::from_varmap(&ours, DType::F32, device);
            let b = VarBuilder::from_varmap(&candle, DType::F32, device);
            let cfg = CrossConfig {
                layers: 2,
                channels: 64,
                time_dim: 128,
                context: 8,
                heads: 2,
                depth,
            };
            let _ = CrossDown::new(a.pp("down"), 32, cfg, true)?;
            let _ = CrossAttnDownBlock2D::new(
                b.pp("down"),
                32,
                64,
                Some(128),
                false,
                CrossAttnDownBlock2DConfig {
                    downblock: DownBlock2DConfig {
                        num_layers: 2,
                        resnet_eps: 1e-5,
                        ..Default::default()
                    },
                    attn_num_head_channels: 2,
                    cross_attention_dim: 8,
                    use_linear_projection: true,
                    transformer_layers_per_block: depth,
                    ..Default::default()
                },
            )?;
            let up_cfg = CrossConfig { layers: 3, ..cfg };
            let _ = CrossUp::new(a.pp("up"), 32, 64, up_cfg, true)?;
            let _ = CrossAttnUpBlock2D::new(
                b.pp("up"),
                32,
                64,
                64,
                Some(128),
                false,
                CrossAttnUpBlock2DConfig {
                    upblock: UpBlock2DConfig {
                        num_layers: 3,
                        resnet_eps: 1e-5,
                        ..Default::default()
                    },
                    attn_num_head_channels: 2,
                    cross_attention_dim: 8,
                    use_linear_projection: true,
                    transformer_layers_per_block: depth,
                    ..Default::default()
                },
            )?;
            let _ = CrossMid::new(a.pp("mid"), cfg)?;
            let _ = UNetMidBlock2DCrossAttn::new(
                b.pp("mid"),
                64,
                Some(128),
                false,
                UNetMidBlock2DCrossAttnConfig {
                    resnet_eps: 1e-5,
                    cross_attn_dim: 8,
                    attn_num_head_channels: 2,
                    use_linear_projection: true,
                    transformer_layers_per_block: depth,
                    ..Default::default()
                },
            )?;
            assert_eq!(
                shapes(&ours)?,
                shapes(&candle)?,
                "parameter mismatch at depth {depth}"
            );
        }
        Ok(())
    }
}
