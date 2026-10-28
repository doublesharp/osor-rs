//! Inference-only layers backed by supplied checkpoint tensors.
use super::super::Weights;
use crate::Result;
use burn::tensor::{
    Device, FloatDType, Int, Tensor, activation, module,
    ops::{ConvOptions, InterpolateMode, InterpolateOptions},
};

pub(super) struct Conv {
    weight: Tensor<4>,
    bias: Tensor<1>,
    stride: usize,
    padding: usize,
    asymmetric: bool,
}
impl Conv {
    #[allow(clippy::too_many_arguments)] // Explicit convolution geometry and checkpoint dimensions.
    pub fn new(
        w: &mut Weights,
        p: &str,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        asymmetric: bool,
    ) -> Result<Self> {
        Ok(Self {
            weight: w.take(&format!("{p}.weight"), [output, input, kernel, kernel])?,
            bias: w.take(&format!("{p}.bias"), [output])?,
            stride,
            padding,
            asymmetric,
        })
    }
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let padding = if self.asymmetric {
            [(0, 1), (0, 1)]
        } else {
            [(self.padding, self.padding); 2]
        };
        module::conv2d(
            x,
            self.weight.clone(),
            Some(self.bias.clone()),
            ConvOptions::new_with_padding([self.stride; 2], padding, [1; 2], 1),
        )
    }
}
pub(super) struct Linear {
    weight: Tensor<2>,
    bias: Option<Tensor<1>>,
}
impl Linear {
    pub fn new(w: &mut Weights, p: &str, input: usize, output: usize, bias: bool) -> Result<Self> {
        Ok(Self {
            weight: w.take(&format!("{p}.weight"), [output, input])?.transpose(),
            bias: if bias {
                Some(w.take(&format!("{p}.bias"), [output])?)
            } else {
                None
            },
        })
    }
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        module::linear(x, self.weight.clone(), self.bias.clone())
    }
}
pub(super) struct GroupNorm {
    weight: Tensor<1>,
    bias: Tensor<1>,
    eps: f64,
}
impl GroupNorm {
    fn mean(mut x: Tensor<3>) -> Tensor<3> {
        let [batch, groups, elements] = x.dims();
        if elements <= 256 {
            return x.mean_dim(2);
        }
        // Bound each first-pass accumulation. Direct million-element means
        // lose enough F32 precision to alter the trained VAE's normalization.
        let mut remaining = elements;
        while remaining > 256 {
            let mut block = 256;
            while block > 1 && !remaining.is_multiple_of(block) {
                block -= 1;
            }
            if block == 1 {
                break;
            }
            remaining /= block;
            x = x
                .reshape([batch, groups, remaining, block])
                .sum_dim(3)
                .reshape([batch, groups, remaining]);
        }
        x.sum_dim(2) / elements as f64
    }

    pub fn new(w: &mut Weights, p: &str, channels: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            weight: w.take(&format!("{p}.weight"), [channels])?,
            bias: w.take(&format!("{p}.bias"), [channels])?,
            eps,
        })
    }
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [b, c, h, w] = x.dims();
        let x = x.reshape([b, 32, c / 32 * h * w]);
        let mean = Self::mean(x.clone());
        let centered = x - mean;
        let variance = Self::mean(centered.clone().square());
        (centered / (variance + self.eps).sqrt()).reshape([b, c, h, w])
            * self.weight.clone().reshape([1, c, 1, 1])
            + self.bias.clone().reshape([1, c, 1, 1])
    }
}
pub(super) struct LayerNorm {
    weight: Tensor<1>,
    bias: Tensor<1>,
}
impl LayerNorm {
    pub fn new(w: &mut Weights, p: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            weight: w.take(&format!("{p}.weight"), [channels])?,
            bias: w.take(&format!("{p}.bias"), [channels])?,
        })
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        module::layer_norm(x, self.weight.clone(), Some(self.bias.clone()), 1e-5)
    }
}
pub(super) fn upsample(x: Tensor<4>, size: [usize; 2]) -> Tensor<4> {
    module::interpolate(
        x,
        InterpolateOptions::new(InterpolateMode::Nearest).with_output_size(size),
    )
}
pub(super) struct Resnet {
    norm1: GroupNorm,
    conv1: Conv,
    time: Option<Linear>,
    norm2: GroupNorm,
    conv2: Conv,
    shortcut: Option<Conv>,
}
impl Resnet {
    pub fn new(
        w: &mut Weights,
        p: &str,
        input: usize,
        output: usize,
        time: Option<usize>,
        eps: f64,
    ) -> Result<Self> {
        Ok(Self {
            norm1: GroupNorm::new(w, &format!("{p}.norm1"), input, eps)?,
            conv1: Conv::new(w, &format!("{p}.conv1"), input, output, 3, 1, 1, false)?,
            time: time
                .map(|d| Linear::new(w, &format!("{p}.time_emb_proj"), d, output, true))
                .transpose()?,
            norm2: GroupNorm::new(w, &format!("{p}.norm2"), output, eps)?,
            conv2: Conv::new(w, &format!("{p}.conv2"), output, output, 3, 1, 1, false)?,
            shortcut: if input != output {
                Some(Conv::new(
                    w,
                    &format!("{p}.conv_shortcut"),
                    input,
                    output,
                    1,
                    1,
                    0,
                    false,
                )?)
            } else {
                None
            },
        })
    }
    pub fn forward(&self, x: Tensor<4>, time: Option<&Tensor<2>>) -> Tensor<4> {
        let residual = match &self.shortcut {
            Some(c) => c.forward(x.clone()),
            None => x.clone(),
        };
        let mut y = self.conv1.forward(activation::silu(self.norm1.forward(x)));
        if let (Some(proj), Some(time)) = (&self.time, time) {
            let [b, c] = [time.dims()[0], y.dims()[1]];
            y = y + proj
                .forward(activation::silu(time.clone()))
                .reshape([b, c, 1, 1]);
        }
        residual + self.conv2.forward(activation::silu(self.norm2.forward(y)))
    }
}
/// Default bound on one attention score slice: 64 Mi F32 elements (256 MiB).
const MAX_SCORE_ELEMENTS: usize = 64 * 1024 * 1024;

/// Scaled-dot-product attention over `q [b,h,n,d]`, `k_t [b,h,d,t]` and
/// `v [b,h,t,d]`. Query rows are sliced only when the full score matrix would
/// exceed `max_score_elements`.
fn attend(
    q: Tensor<4>,
    k_t: Tensor<4>,
    v: Tensor<4>,
    head_width: usize,
    max_score_elements: usize,
) -> Tensor<3> {
    let [b, heads, n, d] = q.dims();
    let tokens = k_t.dims()[3];
    let row_elements = b.saturating_mul(heads).saturating_mul(tokens).max(1);
    let rows = (max_score_elements / row_elements).clamp(1, n.max(1));
    let scale = (head_width as f64).sqrt();
    let y = if rows >= n {
        activation::softmax(q.matmul(k_t) / scale, 3).matmul(v)
    } else {
        let mut pieces = Vec::with_capacity(n.div_ceil(rows));
        for start in (0..n).step_by(rows) {
            let end = (start + rows).min(n);
            let scores = q.clone().slice_dim(2, start..end).matmul(k_t.clone()) / scale;
            pieces.push(activation::softmax(scores, 3).matmul(v.clone()));
        }
        Tensor::cat(pieces, 2)
    };
    y.permute([0, 2, 1, 3]).reshape([b, n, heads * d])
}

pub(super) struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    max_score_elements: usize,
}
impl Attention {
    pub fn new(
        w: &mut Weights,
        p: &str,
        width: usize,
        context: usize,
        heads: usize,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            q: Linear::new(w, &format!("{p}.to_q"), width, width, bias)?,
            k: Linear::new(w, &format!("{p}.to_k"), context, width, bias)?,
            v: Linear::new(w, &format!("{p}.to_v"), context, width, bias)?,
            out: Linear::new(w, &format!("{p}.to_out.0"), width, width, true)?,
            heads,
            max_score_elements: MAX_SCORE_ELEMENTS,
        })
    }
    #[cfg(test)]
    pub fn with_score_limit(mut self, elements: usize) -> Self {
        self.max_score_elements = elements.max(1);
        self
    }
    pub fn forward(&self, x: Tensor<3>, context: Tensor<3>) -> Tensor<3> {
        let [b, n, c] = x.dims();
        let tokens = context.dims()[1];
        let d = c / self.heads;
        let q = self
            .q
            .forward(x)
            .reshape([b, n, self.heads, d])
            .permute([0, 2, 1, 3]);
        let k = self
            .k
            .forward(context.clone())
            .reshape([b, tokens, self.heads, d])
            .permute([0, 2, 3, 1]);
        let v = self
            .v
            .forward(context)
            .reshape([b, tokens, self.heads, d])
            .permute([0, 2, 1, 3]);
        self.out
            .forward(attend(q, k, v, d, self.max_score_elements))
    }
}

/// Cross-attention against the bundle's fixed prompt. Keys and values are
/// computed once at construction; `to_k` and `to_v` weights are released.
pub(super) struct FixedCrossAttention {
    q: Linear,
    k_t: Tensor<4>,
    v: Tensor<4>,
    out: Linear,
    heads: usize,
}
impl FixedCrossAttention {
    pub fn new(
        w: &mut Weights,
        p: &str,
        width: usize,
        context: &Tensor<3>,
        heads: usize,
    ) -> Result<Self> {
        let [b, tokens, context_width] = context.dims();
        let d = width / heads;
        let key = Linear::new(w, &format!("{p}.to_k"), context_width, width, false)?;
        let value = Linear::new(w, &format!("{p}.to_v"), context_width, width, false)?;
        Ok(Self {
            q: Linear::new(w, &format!("{p}.to_q"), width, width, false)?,
            k_t: key
                .forward(context.clone())
                .reshape([b, tokens, heads, d])
                .permute([0, 2, 3, 1]),
            v: value
                .forward(context.clone())
                .reshape([b, tokens, heads, d])
                .permute([0, 2, 1, 3]),
            out: Linear::new(w, &format!("{p}.to_out.0"), width, width, true)?,
            heads,
        })
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, n, c] = x.dims();
        let d = c / self.heads;
        let q = self
            .q
            .forward(x)
            .reshape([b, n, self.heads, d])
            .permute([0, 2, 1, 3]);
        self.out.forward(attend(
            q,
            self.k_t.clone(),
            self.v.clone(),
            d,
            MAX_SCORE_ELEMENTS,
        ))
    }
}
pub(super) struct TransformerBlock {
    norm1: LayerNorm,
    attn1: Attention,
    norm2: LayerNorm,
    attn2: FixedCrossAttention,
    norm3: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
    width: usize,
}
impl TransformerBlock {
    pub fn new(
        w: &mut Weights,
        p: &str,
        width: usize,
        context: &Tensor<3>,
        heads: usize,
    ) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::new(w, &format!("{p}.norm1"), width)?,
            attn1: Attention::new(w, &format!("{p}.attn1"), width, width, heads, false)?,
            norm2: LayerNorm::new(w, &format!("{p}.norm2"), width)?,
            attn2: FixedCrossAttention::new(w, &format!("{p}.attn2"), width, context, heads)?,
            norm3: LayerNorm::new(w, &format!("{p}.norm3"), width)?,
            ff_in: Linear::new(w, &format!("{p}.ff.net.0.proj"), width, width * 8, true)?,
            ff_out: Linear::new(w, &format!("{p}.ff.net.2"), width * 4, width, true)?,
            width,
        })
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let n = self.norm1.forward(x.clone());
        let x = x + self.attn1.forward(n.clone(), n);
        let n = self.norm2.forward(x.clone());
        let x = x + self.attn2.forward(n);
        let ff = self.ff_in.forward(self.norm3.forward(x.clone()));
        let a = ff.clone().slice_dim(2, 0..self.width * 4);
        let gate = ff.slice_dim(2, self.width * 4..self.width * 8);
        x + self.ff_out.forward(a * activation::gelu(gate))
    }
}
pub(super) struct Transformer {
    norm: GroupNorm,
    proj_in: Linear,
    blocks: Vec<TransformerBlock>,
    proj_out: Linear,
}
impl Transformer {
    pub fn new(
        w: &mut Weights,
        p: &str,
        width: usize,
        context: &Tensor<3>,
        heads: usize,
        depth: usize,
    ) -> Result<Self> {
        let norm = GroupNorm::new(w, &format!("{p}.norm"), width, 1e-6)?;
        let proj_in = Linear::new(w, &format!("{p}.proj_in"), width, width, true)?;
        let blocks = (0..depth)
            .map(|i| {
                TransformerBlock::new(
                    w,
                    &format!("{p}.transformer_blocks.{i}"),
                    width,
                    context,
                    heads,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let proj_out = Linear::new(w, &format!("{p}.proj_out"), width, width, true)?;
        Ok(Self {
            norm,
            proj_in,
            blocks,
            proj_out,
        })
    }
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [b, c, h, w] = x.dims();
        let mut y = self
            .norm
            .forward(x.clone())
            .reshape([b, c, h * w])
            .swap_dims(1, 2);
        y = self.proj_in.forward(y);
        for block in &self.blocks {
            y = block.forward(y);
        }
        x + self
            .proj_out
            .forward(y)
            .swap_dims(1, 2)
            .reshape([b, c, h, w])
    }
}
pub(super) struct TimeEmbedding {
    frequencies: Tensor<2>,
    first: Linear,
    second: Linear,
}
impl TimeEmbedding {
    pub fn new(
        w: &mut Weights,
        p: &str,
        projection: usize,
        input: usize,
        output: usize,
        device: &Device,
    ) -> Result<Self> {
        let half = projection / 2;
        let frequencies = (Tensor::<1, Int>::arange(0..half as i64, device)
            .float()
            .cast(FloatDType::F32)
            * -10000_f64.ln()
            / half as f64)
            .exp()
            .reshape([1, half]);
        Ok(Self {
            frequencies,
            first: Linear::new(w, &format!("{p}.linear_1"), input, output, true)?,
            second: Linear::new(w, &format!("{p}.linear_2"), output, output, true)?,
        })
    }
    pub fn project(&self, x: Tensor<1>) -> Tensor<2> {
        let phase = x.unsqueeze_dim::<2>(1) * self.frequencies.clone();
        Tensor::cat(vec![phase.clone().cos(), phase.sin()], 1)
    }
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.second.forward(activation::silu(self.first.forward(x)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Error, Result};

    // A real adapter is deliberately required; normal tests do not initialize a GPU.
    #[test]
    #[ignore = "requires a real WGPU adapter"]
    fn gpu_nearest_odd_geometry_and_erf_gelu() -> Result<()> {
        let (device, _) = super::super::super::acquire_device(0)?;
        let input = Tensor::<4>::from_floats([[[[1., 2., 3.], [4., 5., 6.]]]], &device);
        let resized = upsample(input, [3, 5]);
        let values = resized
            .try_into_data()
            .map_err(|e| Error::Unavailable(format!("WGPU resize readback failed: {e}")))?
            .try_into_vec::<f32>()
            .map_err(|e| Error::Unavailable(format!("WGPU resize dtype failed: {e}")))?;
        assert_eq!(
            values,
            vec![1., 1., 2., 2., 3., 1., 1., 2., 2., 3., 4., 4., 5., 5., 6.]
        );
        let values = activation::gelu(Tensor::<1>::from_floats([-1., 0., 1.], &device))
            .try_into_data()
            .map_err(|e| Error::Unavailable(format!("WGPU GELU readback failed: {e}")))?
            .try_into_vec::<f32>()
            .map_err(|e| Error::Unavailable(format!("WGPU GELU dtype failed: {e}")))?;
        for (actual, expected) in values.iter().zip([-0.15865526_f32, 0., 0.8413447]) {
            assert!((actual - expected).abs() < 1e-5);
        }
        Ok(())
    }
}
