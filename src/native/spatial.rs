//! Spatial and sequence transformer layers for the fixed SDXL architecture.
use super::transformer::FeedForward;
use candle_core::{Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};
use candle_transformers::models::stable_diffusion::attention::CrossAttention;

#[derive(Debug)]
struct TransformerBlock {
    self_attention: CrossAttention,
    cross_attention: CrossAttention,
    feed_forward: FeedForward,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    norm3: nn::LayerNorm,
}
impl TransformerBlock {
    fn new(vs: VarBuilder<'_>, channels: usize, heads: usize, context: usize) -> Result<Self> {
        let head_dim = channels / heads;
        Ok(Self {
            self_attention: CrossAttention::new(
                vs.pp("attn1"),
                channels,
                None,
                heads,
                head_dim,
                None,
                false,
            )?,
            cross_attention: CrossAttention::new(
                vs.pp("attn2"),
                channels,
                Some(context),
                heads,
                head_dim,
                None,
                false,
            )?,
            feed_forward: FeedForward::new(vs.pp("ff"), channels)?,
            norm1: nn::layer_norm(channels, 1e-5, vs.pp("norm1"))?,
            norm2: nn::layer_norm(channels, 1e-5, vs.pp("norm2"))?,
            norm3: nn::layer_norm(channels, 1e-5, vs.pp("norm3"))?,
        })
    }
    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let xs = (xs
            + self
                .self_attention
                .forward(&self.norm1.forward(xs)?, None)?)?;
        let xs = (&xs
            + self
                .cross_attention
                .forward(&self.norm2.forward(&xs)?, context)?)?;
        &xs + self.feed_forward.forward(&self.norm3.forward(&xs)?)?
    }
}

#[derive(Debug)]
pub(super) struct SpatialTransformer {
    norm: nn::GroupNorm,
    projection_in: nn::Linear,
    blocks: Vec<TransformerBlock>,
    projection_out: nn::Linear,
}
impl SpatialTransformer {
    pub fn new(
        vs: VarBuilder<'_>,
        channels: usize,
        heads: usize,
        context: usize,
        depth: usize,
    ) -> Result<Self> {
        if heads == 0 || !channels.is_multiple_of(heads) {
            candle_core::bail!("transformer channels must be divisible by a nonzero head count")
        }
        let norm = nn::group_norm(32, channels, 1e-6, vs.pp("norm"))?;
        let projection_in = nn::linear(channels, channels, vs.pp("proj_in"))?;
        let mut blocks = Vec::with_capacity(depth);
        for index in 0..depth {
            blocks.push(TransformerBlock::new(
                vs.pp("transformer_blocks").pp(index),
                channels,
                heads,
                context,
            )?);
        }
        let projection_out = nn::linear(channels, channels, vs.pp("proj_out"))?;
        Ok(Self {
            norm,
            projection_in,
            blocks,
            projection_out,
        })
    }
    pub fn forward(&self, input: &Tensor, context: Option<&Tensor>) -> Result<Tensor> {
        let (batch, channels, height, width) = input.dims4()?;
        let positions = height.checked_mul(width).ok_or_else(|| {
            candle_core::Error::Msg("spatial transformer dimensions overflow".into())
        })?;
        let xs = self
            .norm
            .forward(input)?
            .transpose(1, 2)?
            .t()?
            .reshape((batch, positions, channels))?;
        let mut xs = self.projection_in.forward(&xs)?;
        for block in &self.blocks {
            xs = block.forward(&xs, context)?;
        }
        let xs = self
            .projection_out
            .forward(&xs)?
            .reshape((batch, height, width, channels))?
            .t()?
            .transpose(1, 2)?;
        xs + input
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    #[test]
    fn spatial_transformer_uses_exact_geglu() -> Result<()> {
        let device = &Device::Cpu;
        let variables = nn::VarMap::new();
        let _ = SpatialTransformer::new(
            VarBuilder::from_varmap(&variables, DType::F32, device),
            32,
            1,
            8,
            1,
        )?;
        let guard = variables
            .data()
            .lock()
            .map_err(|_| candle_core::Error::Msg("variable lock poisoned".into()))?;
        let mut weights = HashMap::new();
        for (name, var) in guard.iter() {
            weights.insert(
                name.clone(),
                Tensor::zeros(var.shape().clone(), DType::F32, device)?,
            );
        }
        drop(guard);
        weights.insert(
            "proj_out.weight".into(),
            Tensor::eye(32, DType::F32, device)?,
        );
        let bias: Vec<f32> = (0..256)
            .map(|i| match i {
                0 => 1.,
                128 => 2.,
                _ => 0.,
            })
            .collect();
        weights.insert(
            "transformer_blocks.0.ff.net.0.proj.bias".into(),
            Tensor::new(bias, device)?,
        );
        let output_weights: Vec<f32> = (0..(32 * 128))
            .map(|i| if i == 0 { 1. } else { 0. })
            .collect();
        weights.insert(
            "transformer_blocks.0.ff.net.2.weight".into(),
            Tensor::from_vec(output_weights, (32, 128), device)?,
        );
        let transformer = SpatialTransformer::new(
            VarBuilder::from_tensors(weights, DType::F32, device),
            32,
            1,
            8,
            1,
        )?;
        let input = Tensor::zeros((1, 32, 1, 1), DType::F32, device)?;
        let context = Tensor::zeros((1, 1, 8), DType::F32, device)?;
        let output = transformer
            .forward(&input, Some(&context))?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(
            (output - 1.954_499_7).abs() < 1e-6,
            "spatial transformer output {output} differs from PyTorch"
        );
        Ok(())
    }
}
