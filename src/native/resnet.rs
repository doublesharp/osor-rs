//! CPU F32 variant of Candle's SDXL ResNet block.
//!
//! Adapted from candle-transformers 0.11.0 at
//! <https://github.com/huggingface/candle/blob/31f35b147389700ed2a178ee66a91c3cc25cc80d/candle-transformers/src/models/stable_diffusion/resnet.rs>.
//! Used under the MIT option; see `LICENSE-CANDLE-MIT`.

use candle_core::{D, DType, Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};
use candle_transformers::models::stable_diffusion::resnet::{ResnetBlock2D, ResnetBlock2DConfig};

use super::unet_conv::UnetConv2d;

#[derive(Debug)]
pub(super) enum Resnet {
    Cpu(Box<CpuResnet>),
    Candle(Box<ResnetBlock2D>),
}

impl Resnet {
    pub(super) fn new(vs: VarBuilder<'_>, input: usize, cfg: ResnetBlock2DConfig) -> Result<Self> {
        if vs.device().is_cpu() && vs.dtype() == DType::F32 {
            Ok(Self::Cpu(Box::new(CpuResnet::new(vs, input, cfg)?)))
        } else {
            Ok(Self::Candle(Box::new(ResnetBlock2D::new(vs, input, cfg)?)))
        }
    }

    pub(super) fn forward_checked(
        &self,
        input: &Tensor,
        temb: Option<&Tensor>,
        checkpoint: crate::pipeline::Checkpoint<'_>,
    ) -> crate::Result<Tensor> {
        checkpoint()?;
        match self {
            Self::Cpu(block) => block.forward_checked(input, temb, checkpoint),
            Self::Candle(block) => Ok(block.forward(input, temb)?),
        }
    }
}

#[derive(Debug)]
pub(super) struct CpuResnet {
    norm1: nn::GroupNorm,
    conv1: UnetConv2d,
    norm2: nn::GroupNorm,
    conv2: UnetConv2d,
    time_emb_proj: Option<nn::Linear>,
    shortcut: Option<UnetConv2d>,
    output_scale_factor: f64,
}

impl CpuResnet {
    fn new(vs: VarBuilder<'_>, input: usize, cfg: ResnetBlock2DConfig) -> Result<Self> {
        let output = cfg.out_channels.unwrap_or(input);
        let norm1 = nn::group_norm(cfg.groups, input, cfg.eps, vs.pp("norm1"))?;
        let conv1 = UnetConv2d::new(input, output, 3, 1, 1, vs.pp("conv1"))?;
        let norm2 = nn::group_norm(
            cfg.groups_out.unwrap_or(cfg.groups),
            output,
            cfg.eps,
            vs.pp("norm2"),
        )?;
        let conv2 = UnetConv2d::new(output, output, 3, 1, 1, vs.pp("conv2"))?;
        let shortcut = if cfg.use_in_shortcut.unwrap_or(input != output) {
            Some(UnetConv2d::new(
                input,
                output,
                1,
                1,
                0,
                vs.pp("conv_shortcut"),
            )?)
        } else {
            None
        };
        let time_emb_proj = cfg
            .temb_channels
            .map(|channels| nn::linear(channels, output, vs.pp("time_emb_proj")))
            .transpose()?;
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            time_emb_proj,
            shortcut,
            output_scale_factor: cfg.output_scale_factor,
        })
    }

    fn forward_checked(
        &self,
        input: &Tensor,
        temb: Option<&Tensor>,
        checkpoint: crate::pipeline::Checkpoint<'_>,
    ) -> crate::Result<Tensor> {
        let shortcut = match &self.shortcut {
            Some(conv) => conv.forward_checked(input, checkpoint)?,
            None => input.clone(),
        };
        let hidden = self
            .conv1
            .forward_checked(&nn::ops::silu(&self.norm1.forward(input)?)?, checkpoint)?;
        checkpoint()?;
        let hidden = match (temb, &self.time_emb_proj) {
            (Some(temb), Some(projection)) => projection
                .forward(&nn::ops::silu(temb)?)?
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?
                .broadcast_add(&hidden)?,
            _ => hidden,
        };
        let hidden = self
            .conv2
            .forward_checked(&nn::ops::silu(&self.norm2.forward(&hidden)?)?, checkpoint)?;
        Ok(((shortcut + hidden)? / self.output_scale_factor)?)
    }
}
