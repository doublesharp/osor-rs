//! SDXL cross-attention block orchestration.
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::stable_diffusion::resnet::ResnetBlock2DConfig;

use super::{resnet::Resnet, spatial::SpatialTransformer, unet_conv::UnetConv2d};

#[derive(Clone, Copy)]
pub(super) struct CrossConfig {
    pub layers: usize,
    pub channels: usize,
    pub time_dim: usize,
    pub heads: usize,
    pub depth: usize,
}

fn resnet(vs: VarBuilder<'_>, input: usize, output: usize, time_dim: usize) -> Result<Resnet> {
    Resnet::new(
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

fn attention(vs: VarBuilder<'_>, cfg: CrossConfig, context: &Tensor) -> Result<SpatialTransformer> {
    SpatialTransformer::new(vs, cfg.channels, cfg.heads, context, cfg.depth)
}

#[derive(Debug)]
pub(super) struct CrossDown {
    layers: Vec<(Resnet, SpatialTransformer)>,
    downsample: Option<UnetConv2d>,
}

impl CrossDown {
    pub fn new(
        vs: VarBuilder<'_>,
        input: usize,
        cfg: CrossConfig,
        downsample: bool,
        context: &Tensor,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let channels = if i == 0 { input } else { cfg.channels };
            layers.push((
                resnet(vs.pp("resnets").pp(i), channels, cfg.channels, cfg.time_dim)?,
                attention(vs.pp("attentions").pp(i), cfg, context)?,
            ));
        }
        let downsample = if downsample {
            Some(UnetConv2d::new(
                cfg.channels,
                cfg.channels,
                3,
                2,
                1,
                vs.pp("downsamplers.0.conv"),
            )?)
        } else {
            None
        };
        Ok(Self { layers, downsample })
    }

    pub fn forward(&self, input: &Tensor, temb: Option<&Tensor>) -> Result<(Tensor, Vec<Tensor>)> {
        let mut xs = input.clone();
        let mut saved =
            Vec::with_capacity(self.layers.len() + usize::from(self.downsample.is_some()));
        for (resnet, attention) in &self.layers {
            xs = attention.forward(&resnet.forward(&xs, temb)?)?;
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
    layers: Vec<(Resnet, SpatialTransformer)>,
    upsample: Option<UnetConv2d>,
}

impl CrossUp {
    pub fn new(
        vs: VarBuilder<'_>,
        input: usize,
        previous: usize,
        cfg: CrossConfig,
        upsample: bool,
        context: &Tensor,
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
                attention(vs.pp("attentions").pp(i), cfg, context)?,
            ));
        }
        let upsample = if upsample {
            Some(UnetConv2d::new(
                cfg.channels,
                cfg.channels,
                3,
                1,
                1,
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
    ) -> Result<Tensor> {
        if saved.len() != self.layers.len() {
            candle_core::bail!("cross-attention up block received an invalid number of residuals")
        }
        let mut xs = input.clone();
        for ((resnet, attention), skip) in self.layers.iter().zip(saved.iter().rev()) {
            xs = Tensor::cat(&[&xs, skip], 1)?.contiguous()?;
            xs = attention.forward(&resnet.forward(&xs, temb)?)?;
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
    initial: Resnet,
    attention: SpatialTransformer,
    final_resnet: Resnet,
}

pub(super) struct CrossMid(MidLayers);
impl CrossMid {
    pub fn new(vs: VarBuilder<'_>, cfg: CrossConfig, context: &Tensor) -> Result<Self> {
        Ok(Self(MidLayers {
            initial: resnet(vs.pp("resnets.0"), cfg.channels, cfg.channels, cfg.time_dim)?,
            attention: attention(vs.pp("attentions.0"), cfg, context)?,
            final_resnet: resnet(vs.pp("resnets.1"), cfg.channels, cfg.channels, cfg.time_dim)?,
        }))
    }
    pub fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        let xs = self.0.initial.forward(xs, temb)?;
        let xs = self.0.attention.forward(&xs)?;
        self.0.final_resnet.forward(&xs, temb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn as nn;
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
                heads: 2,
                depth,
            };
            let context = Tensor::zeros((1, 2, 8), DType::F32, device)?;
            let _ = CrossDown::new(a.pp("down"), 32, cfg, true, &context)?;
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
            let _ = CrossUp::new(a.pp("up"), 32, 64, up_cfg, true, &context)?;
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
            let _ = CrossMid::new(a.pp("mid"), cfg, &context)?;
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

    #[test]
    fn cross_up_matches_candle_for_batched_and_odd_upsamples() -> Result<()> {
        let device = &Device::Cpu;
        let mut variables = nn::VarMap::new();
        let vs = VarBuilder::from_varmap(&variables, DType::F32, device);
        let values = |count: usize, seed: usize| {
            (0..count)
                .map(|i| (((i * 43 + seed) % 103) as f32 - 51.) / 37.)
                .collect::<Vec<_>>()
        };
        let context = Tensor::from_vec(values(24, 3), (1, 3, 8), device)?;
        let cfg = CrossConfig {
            layers: 2,
            channels: 64,
            time_dim: 128,
            heads: 2,
            depth: 1,
        };
        // Populate the shared variable map before assigning deterministic weights.
        let _ = CrossUp::new(vs.clone(), 64, 64, cfg, true, &context)?;
        let candle = CrossAttnUpBlock2D::new(
            vs,
            64,
            64,
            64,
            Some(128),
            false,
            CrossAttnUpBlock2DConfig {
                upblock: UpBlock2DConfig {
                    num_layers: 2,
                    resnet_eps: 1e-5,
                    ..Default::default()
                },
                attn_num_head_channels: 2,
                cross_attention_dim: 8,
                use_linear_projection: true,
                transformer_layers_per_block: 1,
                ..Default::default()
            },
        )?;
        {
            let tensors = variables
                .data()
                .lock()
                .map_err(|_| candle_core::Error::Msg("variable lock poisoned".into()))?;
            for (name, variable) in tensors.iter() {
                if name.ends_with(".weight") && variable.dims().len() > 1 {
                    let fan_in = variable.dims().iter().skip(1).product::<usize>();
                    let scale = 1. / (fan_in as f32).sqrt();
                    let mut state = name
                        .bytes()
                        .fold(17u32, |s, b| s.wrapping_mul(31).wrapping_add(u32::from(b)));
                    let weights = (0..variable.elem_count())
                        .map(|_| {
                            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                            (((state >> 16) as f32 / 32_768.) - 1.) * scale
                        })
                        .collect::<Vec<_>>();
                    variable.set(&Tensor::from_vec(
                        weights,
                        variable.shape().clone(),
                        device,
                    )?)?;
                } else if name.ends_with(".bias") {
                    variable.set(&Tensor::zeros(
                        variable.shape().clone(),
                        DType::F32,
                        device,
                    )?)?;
                }
            }
        }
        // Candle 0.11 uses approximate GELU; OSOR's transformer deliberately
        // uses the reference's exact GELU. Suppress only that feed-forward
        // output so this comparison isolates the surrounding block behavior.
        for layer in 0..2 {
            let prefix = format!("attentions.{layer}.transformer_blocks.0.ff.net.2");
            variables.set_one(
                format!("{prefix}.weight"),
                Tensor::zeros((64, 256), DType::F32, device)?,
            )?;
            variables.set_one(
                format!("{prefix}.bias"),
                Tensor::zeros(64, DType::F32, device)?,
            )?;
        }
        // The fixed attention keys and values must use the deterministic weights.
        let ours = CrossUp::new(
            VarBuilder::from_varmap(&variables, DType::F32, device),
            64,
            64,
            cfg,
            true,
            &context,
        )?;
        for batch in [1, 2] {
            let input =
                Tensor::from_vec(values(batch * 64 * 3 * 5, 11), (batch, 64, 3, 5), device)?;
            let skips = vec![
                Tensor::from_vec(values(batch * 64 * 3 * 5, 17), (batch, 64, 3, 5), device)?,
                Tensor::from_vec(values(batch * 64 * 3 * 5, 29), (batch, 64, 3, 5), device)?,
            ];
            let temb = Tensor::from_vec(values(batch * 128, 23), (batch, 128), device)?;
            let full_context = context.repeat((batch, 1, 1))?;
            for size in [None, Some((5, 9))] {
                let actual = ours.forward(&input, &skips, Some(&temb), size)?;
                let expected =
                    candle.forward(&input, &skips, Some(&temb), size, Some(&full_context))?;
                let (height, width) = size.unwrap_or((6, 10));
                assert_eq!(actual.dims(), &[batch, 64, height, width]);
                let error = (actual - expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f32>()?;
                assert!(error <= 3e-5, "batch {batch}, size {size:?}: error {error}");
            }
            assert!(
                ours.forward(&input, std::slice::from_ref(&input), Some(&temb), None)
                    .is_err()
            );
        }
        Ok(())
    }
}
