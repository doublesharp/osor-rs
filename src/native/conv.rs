//! VAE convolutions with bounded im2col scratch.
//!
//! Candle lowers each 2D convolution to an im2col buffer holding
//! `batch * out_h * out_w * c_in * k * k` elements. A full-resolution VAE
//! convolution can need several GiB, which Metal rounds up to a power-of-two
//! allocation. Large convolutions are split into bands of output rows. Each
//! output element keeps the same input window and reduction length.

use candle_core::{Module, Result, Tensor};
use candle_nn::{self as nn, VarBuilder};

/// Default im2col scratch bound: 32 Mi elements, 128 MiB in F32.
pub(super) const MAX_SCRATCH_ELEMENTS: usize = 32 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(super) struct BandedConv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    padding: usize,
    stride: usize,
    kernel: usize,
    max_scratch_elements: usize,
}

impl BandedConv2d {
    fn checked_scratch(
        batch: usize,
        rows: usize,
        width: usize,
        channels: usize,
        kernel: usize,
    ) -> Result<usize> {
        [rows, width, channels, kernel, kernel]
            .iter()
            .try_fold(batch, |count, &dim| count.checked_mul(dim))
            .ok_or_else(|| candle_core::Error::Msg("convolution scratch size overflows".into()))
    }

    #[cfg(test)]
    fn with_scratch_limit(mut self, elements: usize) -> Self {
        self.max_scratch_elements = elements.max(1);
        self
    }
}

impl Module for BandedConv2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, channels, height, width) = xs.dims4()?;
        let padded_height = height + 2 * self.padding;
        let padded_width = width + 2 * self.padding;
        if padded_height < self.kernel || padded_width < self.kernel {
            candle_core::bail!("convolution input is smaller than its kernel")
        }
        let out_height = (padded_height - self.kernel) / self.stride + 1;
        let out_width = (padded_width - self.kernel) / self.stride + 1;
        let row_scratch = Self::checked_scratch(batch, 1, out_width, channels, self.kernel)?;
        let rows_per_band = (self.max_scratch_elements / row_scratch).max(1);
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if xs.device().is_metal()
            && xs.dtype() == candle_core::DType::F32
            && self.weight.dtype() == candle_core::DType::F32
        {
            // Candle's Metal im2col and NCHW copy dominate its convolution
            // time; channel-major columns feed its fast GEMM directly.
            let output = super::columns::conv2d(
                xs,
                &self.weight,
                self.stride,
                self.padding,
                self.max_scratch_elements,
            )?;
            return self.add_bias(output);
        }
        let output = if rows_per_band >= out_height {
            xs.conv2d(&self.weight, self.padding, self.stride, 1, 1)?
        } else {
            let padded = if self.padding == 0 {
                xs.clone()
            } else {
                xs.pad_with_zeros(2, self.padding, self.padding)?
                    .pad_with_zeros(3, self.padding, self.padding)?
            };
            let mut bands = Vec::with_capacity(out_height.div_ceil(rows_per_band));
            for start in (0..out_height).step_by(rows_per_band) {
                let rows = rows_per_band.min(out_height - start);
                let input_rows = (rows - 1) * self.stride + self.kernel;
                let band = padded
                    .narrow(2, start * self.stride, input_rows)?
                    .contiguous()?;
                bands.push(band.conv2d(&self.weight, 0, self.stride, 1, 1)?);
            }
            Tensor::cat(&bands, 2)?
        };
        self.add_bias(output)
    }
}

impl BandedConv2d {
    fn add_bias(&self, output: Tensor) -> Result<Tensor> {
        match &self.bias {
            Some(bias) => {
                let channels = bias.dims1()?;
                output.broadcast_add(&bias.reshape((1, channels, 1, 1))?)
            }
            None => Ok(output),
        }
    }
}

/// Load a square, ungrouped, undilated convolution like `candle_nn::conv2d`.
pub(super) fn conv2d(
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    vs: VarBuilder<'_>,
) -> Result<BandedConv2d> {
    if kernel == 0 || stride == 0 {
        candle_core::bail!("convolution kernel and stride must be positive")
    }
    let weight = vs.get_with_hints(
        (output, input, kernel, kernel),
        "weight",
        nn::init::DEFAULT_KAIMING_NORMAL,
    )?;
    let bound = 1. / (input as f64).sqrt();
    let bias = vs.get_with_hints(
        output,
        "bias",
        nn::Init::Uniform {
            lo: -bound,
            up: bound,
        },
    )?;
    Ok(BandedConv2d {
        weight,
        bias: Some(bias),
        padding,
        stride,
        kernel,
        max_scratch_elements: MAX_SCRATCH_ELEMENTS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn exact(a: &Tensor, b: &Tensor) -> Result<bool> {
        Ok(a.dims() == b.dims()
            && (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()? == 0.0)
    }

    #[test]
    fn banded_convolution_matches_direct_convolution_exactly() -> Result<()> {
        let device = &Device::Cpu;
        let variables = nn::VarMap::new();
        let vs = VarBuilder::from_varmap(&variables, DType::F32, device);
        // (input height, width, kernel, stride, padding, scratch rows)
        for (index, (height, width, kernel, stride, padding, rows)) in [
            (37, 20, 3, 1, 1, 4),
            (37, 20, 3, 1, 1, 1),
            (33, 17, 3, 2, 0, 3),
            (19, 11, 1, 1, 0, 5),
        ]
        .into_iter()
        .enumerate()
        {
            let conv = conv2d(6, 5, kernel, stride, padding, vs.pp(index))?;
            let input = Tensor::randn(0f32, 1f32, (2, 6, height, width), device)?;
            let out_width = (width + 2 * padding - kernel) / stride + 1;
            let banded = conv
                .clone()
                .with_scratch_limit(2 * out_width * 6 * kernel * kernel * rows)
                .forward(&input)?;
            let direct = nn::Conv2d::new(
                conv.weight.clone(),
                conv.bias.clone(),
                nn::Conv2dConfig {
                    padding,
                    stride,
                    ..Default::default()
                },
            )
            .forward(&input)?;
            assert!(
                exact(&banded, &direct)?,
                "banded convolution differs for case {index}"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_inputs_smaller_than_the_kernel() -> Result<()> {
        let device = &Device::Cpu;
        let variables = nn::VarMap::new();
        let conv = conv2d(
            1,
            1,
            3,
            2,
            0,
            VarBuilder::from_varmap(&variables, DType::F32, device),
        )?;
        assert!(
            conv.forward(&Tensor::zeros((1, 1, 2, 8), DType::F32, device)?)
                .is_err()
        );
        Ok(())
    }
}
