//! CPU F32 variants of Candle's plain SDXL down and up blocks.
//!
//! Adapted from candle-transformers 0.11.0 at
//! <https://github.com/huggingface/candle/blob/31f35b147389700ed2a178ee66a91c3cc25cc80d/candle-transformers/src/models/stable_diffusion/unet_2d_blocks.rs>.
//! Used under the MIT option; see `LICENSE-CANDLE-MIT`.

use crate::pipeline::Checkpoint;
use candle_core::{D, DType, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::{
    resnet::ResnetBlock2DConfig,
    unet_2d_blocks::{DownBlock2D, DownBlock2DConfig, UpBlock2D, UpBlock2DConfig},
};

use super::{resnet::Resnet, unet_conv::UnetConv2d};

#[derive(Debug)]
pub(super) enum PlainDown {
    Cpu(CpuDown),
    Candle(DownBlock2D),
}

impl PlainDown {
    pub(super) fn new(
        vs: VarBuilder<'_>,
        input: usize,
        output: usize,
        time: Option<usize>,
        cfg: DownBlock2DConfig,
    ) -> Result<Self> {
        if !vs.device().is_cpu() || vs.dtype() != DType::F32 {
            return Ok(Self::Candle(DownBlock2D::new(
                vs, input, output, time, cfg,
            )?));
        }
        let resnet_cfg = ResnetBlock2DConfig {
            out_channels: Some(output),
            eps: cfg.resnet_eps,
            output_scale_factor: cfg.output_scale_factor,
            temb_channels: time,
            ..Default::default()
        };
        let mut resnets = Vec::with_capacity(cfg.num_layers);
        for index in 0..cfg.num_layers {
            resnets.push(Resnet::new(
                vs.pp("resnets").pp(index),
                if index == 0 { input } else { output },
                resnet_cfg,
            )?);
        }
        let downsample = if cfg.add_downsample {
            Some(UnetConv2d::new(
                output,
                output,
                3,
                2,
                cfg.downsample_padding,
                vs.pp("downsamplers.0.conv"),
            )?)
        } else {
            None
        };
        Ok(Self::Cpu(CpuDown {
            resnets,
            downsample,
            padding: cfg.downsample_padding,
        }))
    }

    #[cfg(test)]
    pub(super) fn forward(
        &self,
        input: &Tensor,
        temb: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        self.forward_checked(input, temb, &|| Ok(()))
            .map_err(|error| candle_core::Error::Msg(error.to_string()))
    }

    pub(super) fn forward_checked(
        &self,
        input: &Tensor,
        temb: Option<&Tensor>,
        checkpoint: Checkpoint<'_>,
    ) -> crate::Result<(Tensor, Vec<Tensor>)> {
        checkpoint()?;
        match self {
            Self::Candle(block) => Ok(block.forward(input, temb)?),
            Self::Cpu(block) => {
                let mut xs = input.clone();
                let mut saved = Vec::with_capacity(
                    block.resnets.len() + usize::from(block.downsample.is_some()),
                );
                for resnet in &block.resnets {
                    checkpoint()?;
                    xs = resnet.forward_checked(&xs, temb, checkpoint)?;
                    saved.push(xs.clone());
                }
                if let Some(conv) = &block.downsample {
                    if block.padding == 0 {
                        xs = xs
                            .pad_with_zeros(D::Minus1, 0, 1)?
                            .pad_with_zeros(D::Minus2, 0, 1)?;
                    }
                    xs = conv.forward_checked(&xs, checkpoint)?;
                    saved.push(xs.clone());
                }
                Ok((xs, saved))
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct CpuDown {
    resnets: Vec<Resnet>,
    downsample: Option<UnetConv2d>,
    padding: usize,
}

#[derive(Debug)]
pub(super) enum PlainUp {
    Cpu(CpuUp),
    Candle(UpBlock2D),
}

impl PlainUp {
    pub(super) fn new(
        vs: VarBuilder<'_>,
        input: usize,
        previous: usize,
        output: usize,
        time: Option<usize>,
        cfg: UpBlock2DConfig,
    ) -> Result<Self> {
        if !vs.device().is_cpu() || vs.dtype() != DType::F32 {
            return Ok(Self::Candle(UpBlock2D::new(
                vs, input, previous, output, time, cfg,
            )?));
        }
        let resnet_cfg = ResnetBlock2DConfig {
            out_channels: Some(output),
            eps: cfg.resnet_eps,
            output_scale_factor: cfg.output_scale_factor,
            temb_channels: time,
            ..Default::default()
        };
        let mut resnets = Vec::with_capacity(cfg.num_layers);
        for index in 0..cfg.num_layers {
            let skip = if index + 1 == cfg.num_layers {
                input
            } else {
                output
            };
            let channels = if index == 0 { previous } else { output };
            resnets.push(Resnet::new(
                vs.pp("resnets").pp(index),
                channels + skip,
                resnet_cfg,
            )?);
        }
        let upsample = if cfg.add_upsample {
            Some(UnetConv2d::new(
                output,
                output,
                3,
                1,
                1,
                vs.pp("upsamplers.0.conv"),
            )?)
        } else {
            None
        };
        Ok(Self::Cpu(CpuUp { resnets, upsample }))
    }

    pub(super) fn residual_count(&self) -> usize {
        match self {
            Self::Cpu(block) => block.resnets.len(),
            Self::Candle(block) => block.resnets.len(),
        }
    }

    #[cfg(test)]
    pub(super) fn forward(
        &self,
        input: &Tensor,
        saved: &[Tensor],
        temb: Option<&Tensor>,
        size: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        self.forward_checked(input, saved, temb, size, &|| Ok(()))
            .map_err(|error| candle_core::Error::Msg(error.to_string()))
    }

    pub(super) fn forward_checked(
        &self,
        input: &Tensor,
        saved: &[Tensor],
        temb: Option<&Tensor>,
        size: Option<(usize, usize)>,
        checkpoint: Checkpoint<'_>,
    ) -> crate::Result<Tensor> {
        checkpoint()?;
        match self {
            Self::Candle(block) => Ok(block.forward(input, saved, temb, size)?),
            Self::Cpu(block) => {
                if saved.len() != block.resnets.len() {
                    return Err(candle_core::Error::Msg(
                        "plain up block received an invalid number of residuals".into(),
                    )
                    .into());
                }
                let mut xs = input.clone();
                for (resnet, skip) in block.resnets.iter().zip(saved.iter().rev()) {
                    checkpoint()?;
                    xs = resnet.forward_checked(
                        &Tensor::cat(&[&xs, skip], 1)?.contiguous()?,
                        temb,
                        checkpoint,
                    )?;
                }
                if let Some(conv) = &block.upsample {
                    let (height, width) = match size {
                        Some(size) => size,
                        None => {
                            let (_, _, height, width) = xs.dims4()?;
                            (
                                height.checked_mul(2).ok_or_else(|| {
                                    candle_core::Error::Msg("upsample height overflow".into())
                                })?,
                                width.checked_mul(2).ok_or_else(|| {
                                    candle_core::Error::Msg("upsample width overflow".into())
                                })?,
                            )
                        }
                    };
                    xs =
                        conv.forward_checked(&xs.upsample_nearest2d(height, width)?, checkpoint)?;
                }
                Ok(xs)
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct CpuUp {
    resnets: Vec<Resnet>,
    upsample: Option<UnetConv2d>,
}
