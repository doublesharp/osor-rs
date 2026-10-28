//! Spatial and sequence transformer layers for the fixed SDXL architecture.
//! Attention follows Candle's stable-diffusion `CrossAttention`
//! (candle-transformers 0.9.1, MIT/Apache-2.0) operation for operation.
//! Keys and values for the fixed prompt are computed once at construction,
//! so the `attn2.to_k` and `attn2.to_v` weights are not retained.
use super::transformer::FeedForward;
use candle_core::{DType, Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};

fn split_heads(xs: &Tensor, heads: usize) -> Result<Tensor> {
    let (batch, tokens, width) = xs.dims3()?;
    xs.reshape((batch, tokens, heads, width / heads))?
        .transpose(1, 2)?
        .reshape((batch * heads, tokens, width / heads))
}

fn merge_heads(xs: &Tensor, heads: usize) -> Result<Tensor> {
    let (batch_heads, tokens, head_width) = xs.dims3()?;
    xs.reshape((batch_heads / heads, heads, tokens, head_width))?
        .transpose(1, 2)?
        .reshape((batch_heads / heads, tokens, head_width * heads))
}

/// Keys transposed to `[batch*heads, head_width, tokens]` and scaled, in F32.
fn scaled_keys_t(keys: &Tensor, scale: f64) -> Result<Tensor> {
    keys.to_dtype(DType::F32)?.t()? * scale
}

/// Softmax attention computed in F32, returning the query dtype.
fn attend(query: &Tensor, keys_t: &Tensor, values: &Tensor, heads: usize) -> Result<Tensor> {
    let dtype = query.dtype();
    let scores = query.to_dtype(DType::F32)?.matmul(keys_t)?;
    let probabilities = nn::ops::softmax_last_dim(&scores)?;
    merge_heads(&probabilities.matmul(values)?.to_dtype(dtype)?, heads)
}

fn head_scale(channels: usize, heads: usize) -> Result<f64> {
    if heads == 0 || !channels.is_multiple_of(heads) {
        candle_core::bail!("attention channels must be divisible by a nonzero head count")
    }
    Ok(1.0 / f64::sqrt((channels / heads) as f64))
}

#[derive(Debug)]
struct SelfAttention {
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    output: nn::Linear,
    heads: usize,
    scale: f64,
}
impl SelfAttention {
    fn new(vs: VarBuilder<'_>, channels: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            query: nn::linear_no_bias(channels, channels, vs.pp("to_q"))?,
            key: nn::linear_no_bias(channels, channels, vs.pp("to_k"))?,
            value: nn::linear_no_bias(channels, channels, vs.pp("to_v"))?,
            output: nn::linear(channels, channels, vs.pp("to_out.0"))?,
            heads,
            scale: head_scale(channels, heads)?,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let query = split_heads(&self.query.forward(xs)?, self.heads)?;
        let context = xs.contiguous()?;
        let keys = split_heads(&self.key.forward(&context)?, self.heads)?;
        let values = split_heads(&self.value.forward(&context)?, self.heads)?;
        let xs = attend(
            &query,
            &scaled_keys_t(&keys, self.scale)?,
            &values.to_dtype(DType::F32)?,
            self.heads,
        )?;
        self.output.forward(&xs)
    }
}

/// Cross-attention against a fixed `[1, tokens, context]` sequence.
#[derive(Debug)]
struct FixedCrossAttention {
    query: nn::Linear,
    keys_t: Tensor,
    values: Tensor,
    output: nn::Linear,
    heads: usize,
}
impl FixedCrossAttention {
    fn new(vs: VarBuilder<'_>, channels: usize, heads: usize, context: &Tensor) -> Result<Self> {
        let (batch, _, width) = context.dims3()?;
        if batch != 1 {
            candle_core::bail!("fixed cross-attention context must have batch one")
        }
        let scale = head_scale(channels, heads)?;
        let query = nn::linear_no_bias(channels, channels, vs.pp("to_q"))?;
        let key = nn::linear_no_bias(width, channels, vs.pp("to_k"))?;
        let value = nn::linear_no_bias(width, channels, vs.pp("to_v"))?;
        let output = nn::linear(channels, channels, vs.pp("to_out.0"))?;
        let context = context.contiguous()?;
        let keys = split_heads(&key.forward(&context)?, heads)?;
        let values = split_heads(&value.forward(&context)?, heads)?;
        Ok(Self {
            query,
            keys_t: scaled_keys_t(&keys, scale)?,
            values: values.to_dtype(DType::F32)?,
            output,
            heads,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let batch = xs.dim(0)?;
        let query = split_heads(&self.query.forward(xs)?, self.heads)?;
        let xs = if batch == 1 {
            attend(&query, &self.keys_t, &self.values, self.heads)?
        } else {
            attend(
                &query,
                &self.keys_t.repeat((batch, 1, 1))?,
                &self.values.repeat((batch, 1, 1))?,
                self.heads,
            )?
        };
        self.output.forward(&xs)
    }
}

#[derive(Debug)]
struct TransformerBlock {
    self_attention: SelfAttention,
    cross_attention: FixedCrossAttention,
    feed_forward: FeedForward,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    norm3: nn::LayerNorm,
}
impl TransformerBlock {
    fn new(vs: VarBuilder<'_>, channels: usize, heads: usize, context: &Tensor) -> Result<Self> {
        Ok(Self {
            self_attention: SelfAttention::new(vs.pp("attn1"), channels, heads)?,
            cross_attention: FixedCrossAttention::new(vs.pp("attn2"), channels, heads, context)?,
            feed_forward: FeedForward::new(vs.pp("ff"), channels)?,
            norm1: nn::layer_norm(channels, 1e-5, vs.pp("norm1"))?,
            norm2: nn::layer_norm(channels, 1e-5, vs.pp("norm2"))?,
            norm3: nn::layer_norm(channels, 1e-5, vs.pp("norm3"))?,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = (xs + self.self_attention.forward(&self.norm1.forward(xs)?)?)?;
        let xs = (&xs + self.cross_attention.forward(&self.norm2.forward(&xs)?)?)?;
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
        context: &Tensor,
        depth: usize,
    ) -> Result<Self> {
        head_scale(channels, heads)?;
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
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
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
            xs = block.forward(&xs)?;
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
    use candle_transformers::models::stable_diffusion::attention::CrossAttention;
    use std::collections::HashMap;

    fn identical(a: &Tensor, b: &Tensor) -> Result<bool> {
        Ok(a.dims() == b.dims()
            && (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()? == 0.0)
    }

    // F64 projections, scores, softmax and output projection from the defining
    // attention equations. No fixed-prompt cache or forced F32 intermediates.
    fn reference_attention_f64(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        input: &Tensor,
        context: &Tensor,
        heads: usize,
    ) -> Result<Tensor> {
        let project = |name: &str, xs: &Tensor| -> Result<Tensor> {
            let (batch, tokens, width) = xs.dims3()?;
            let key = format!("{prefix}.{name}");
            let weight = weights.get(&format!("{key}.weight")).ok_or_else(|| {
                candle_core::Error::Msg(format!("missing reference weight {key}"))
            })?;
            let output = xs
                .to_dtype(DType::F64)?
                .contiguous()?
                .reshape((batch * tokens, width))?
                .matmul(&weight.to_dtype(DType::F64)?.t()?)?;
            let output = match weights.get(&format!("{key}.bias")) {
                Some(bias) => output.broadcast_add(&bias.to_dtype(DType::F64)?)?,
                None => output,
            };
            output.reshape((batch, tokens, ()))
        };
        let query = split_heads(&project("to_q", input)?, heads)?;
        let key = split_heads(&project("to_k", context)?, heads)?;
        let value = split_heads(&project("to_v", context)?, heads)?;
        let scale = 1.0 / (query.dim(2)? as f64).sqrt();
        let scores = (query.matmul(&key.t()?)? * scale)?;
        let output = merge_heads(&nn::ops::softmax_last_dim(&scores)?.matmul(&value)?, heads)?;
        project("to_out.0", &output)
    }

    #[test]
    fn cached_and_self_attention_match_candle_and_f64() -> Result<()> {
        let device = &Device::Cpu;
        let (channels, heads, width) = (16, 4, 12);
        // Dyadic inputs and projection weights are exactly representable,
        // making the cache equivalence check independent of BLAS tiling.
        let values = |count: usize, salt: usize| {
            (0..count)
                .map(|i| (((i * 7 + salt) % 11) as f32 - 5.) / 16.)
                .collect::<Vec<_>>()
        };
        let mut weights = HashMap::new();
        for (prefix, context_width) in [("cross", width), ("self", channels)] {
            for (index, (name, input_width)) in [
                ("to_q", channels),
                ("to_k", context_width),
                ("to_v", context_width),
                ("to_out.0", channels),
            ]
            .into_iter()
            .enumerate()
            {
                weights.insert(
                    format!("{prefix}.{name}.weight"),
                    Tensor::from_vec(
                        values(channels * input_width, index),
                        (channels, input_width),
                        device,
                    )?,
                );
            }
            weights.insert(
                format!("{prefix}.to_out.0.bias"),
                Tensor::from_vec(values(channels, 3), channels, device)?,
            );
        }
        let vs = VarBuilder::from_tensors(weights.clone(), DType::F32, device);
        let context = Tensor::from_vec(values(5 * width, 5), (1, 5, width), device)?;
        let candle_cross = CrossAttention::new(
            vs.pp("cross"),
            channels,
            Some(width),
            heads,
            channels / heads,
            None,
            false,
        )?;
        let candle_self = CrossAttention::new(
            vs.pp("self"),
            channels,
            None,
            heads,
            channels / heads,
            None,
            false,
        )?;
        let cross = FixedCrossAttention::new(vs.pp("cross"), channels, heads, &context)?;
        let self_attention = SelfAttention::new(vs.pp("self"), channels, heads)?;
        for batch in [1, 2] {
            let xs = Tensor::from_vec(
                values(batch * 7 * channels, 2),
                (batch, 7, channels),
                device,
            )?;
            let batch_context = context.repeat((batch, 1, 1))?;
            let cached = cross.forward(&xs)?;
            let direct_cross = candle_cross.forward(&xs, Some(&batch_context))?;
            let own_self = self_attention.forward(&xs)?;
            let direct_self = candle_self.forward(&xs, None)?;
            assert!(identical(&cached, &direct_cross)?);
            assert!(identical(&own_self, &direct_self)?);
            let reference_cross =
                reference_attention_f64(&weights, "cross", &xs, &batch_context, heads)?;
            let reference_self = reference_attention_f64(&weights, "self", &xs, &xs, heads)?;
            for (actual, reference) in [
                (&cached, &reference_cross),
                (&direct_cross, &reference_cross),
                (&own_self, &reference_self),
                (&direct_self, &reference_self),
            ] {
                let error = (actual.to_dtype(DType::F64)? - reference)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_scalar::<f64>()?;
                assert!(error <= 1e-5, "batch {batch} F64 attention error {error}");
            }
        }
        Ok(())
    }

    #[test]
    fn spatial_transformer_uses_exact_geglu() -> Result<()> {
        let device = &Device::Cpu;
        let variables = nn::VarMap::new();
        let context = Tensor::zeros((1, 1, 8), DType::F32, device)?;
        let _ = SpatialTransformer::new(
            VarBuilder::from_varmap(&variables, DType::F32, device),
            32,
            1,
            &context,
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
            &context,
            1,
        )?;
        let input = Tensor::zeros((1, 32, 1, 1), DType::F32, device)?;
        let output = transformer.forward(&input)?.sum_all()?.to_scalar::<f32>()?;
        assert!(
            (output - 1.954_499_7).abs() < 1e-6,
            "spatial transformer output {output} differs from PyTorch"
        );
        Ok(())
    }
}
