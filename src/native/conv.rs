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
        self.forward_checked(xs, &|| Ok(()))
            .map_err(|error| candle_core::Error::Msg(error.to_string()))
    }
}

impl BandedConv2d {
    pub(super) fn forward_checked(
        &self,
        xs: &Tensor,
        checkpoint: crate::pipeline::Checkpoint<'_>,
    ) -> crate::Result<Tensor> {
        checkpoint()?;
        let (batch, channels, height, width) = xs.dims4()?;
        let padded_height = height + 2 * self.padding;
        let padded_width = width + 2 * self.padding;
        if padded_height < self.kernel || padded_width < self.kernel {
            return Err(candle_core::Error::Msg(
                "convolution input is smaller than its kernel".into(),
            )
            .into());
        }
        let out_height = (padded_height - self.kernel) / self.stride + 1;
        let out_width = (padded_width - self.kernel) / self.stride + 1;
        let row_scratch = Self::checked_scratch(batch, 1, out_width, channels, self.kernel)?;
        let rows_per_band = (self.max_scratch_elements / row_scratch).max(1);
        if (xs.device().is_metal() || xs.device().is_cpu())
            && xs.dtype() == candle_core::DType::F32
            && self.weight.dtype() == candle_core::DType::F32
        {
            // Candle's Metal im2col and NCHW copy dominate its convolution
            // time; channel-major columns feed its fast GEMM directly.
            let output = super::columns::conv2d_checked(
                xs,
                &self.weight,
                self.stride,
                self.padding,
                self.max_scratch_elements,
                checkpoint,
            )?;
            return Ok(self.add_bias(output)?);
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
                checkpoint()?;
                let rows = rows_per_band.min(out_height - start);
                let input_rows = (rows - 1) * self.stride + self.kernel;
                let band = padded
                    .narrow(2, start * self.stride, input_rows)?
                    .contiguous()?;
                bands.push(band.conv2d(&self.weight, 0, self.stride, 1, 1)?);
            }
            Tensor::cat(&bands, 2)?
        };
        checkpoint()?;
        Ok(self.add_bias(output)?)
    }
}

impl BandedConv2d {
    fn add_bias(&self, output: Tensor) -> Result<Tensor> {
        match &self.bias {
            Some(bias) => {
                let channels = bias.dims1()?;
                #[cfg(all(feature = "metal", target_os = "macos"))]
                if output.device().is_metal()
                    && output.dtype() == candle_core::DType::F32
                    && bias.dtype() == candle_core::DType::F32
                {
                    let (_, output_channels, height, width) = output.dims4()?;
                    let spatial = height.checked_mul(width).ok_or_else(|| {
                        candle_core::Error::Msg("convolution output size overflows".into())
                    })?;
                    if channels == output_channels
                        && spatial > 0
                        && output.elem_count() > 0
                        && spatial.is_multiple_of(4)
                        && u32::try_from(output.elem_count() / 4).is_ok()
                    {
                        let contiguous = output.contiguous()?;
                        if contiguous.layout().start_offset().is_multiple_of(4) {
                            return super::vae_encoder::metal_ops::channel_bias(&contiguous, bias);
                        }
                    }
                }
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

    #[test]
    fn checked_generic_convolution_cancels_after_the_first_band() -> crate::Result<()> {
        use std::cell::Cell;
        let device = &Device::Cpu;
        let variables = nn::VarMap::new();
        let conv = conv2d(
            2,
            3,
            3,
            1,
            1,
            VarBuilder::from_varmap(&variables, DType::F64, device),
        )?
        .with_scratch_limit(1);
        let input = Tensor::ones((1, 2, 8, 8), DType::F64, device)?;
        let calls = Cell::new(0);
        let result = conv.forward_checked(&input, &|| {
            calls.set(calls.get() + 1);
            if calls.get() == 3 {
                Err(crate::Error::Cancelled)
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(crate::Error::Cancelled)));
        assert_eq!(calls.get(), 3);
        Ok(())
    }

    fn check_bias_reference(device: &Device) -> Result<()> {
        let input = Tensor::arange(0f32, 72f32, device)?
            .reshape((3, 3, 4, 2))?
            .narrow(0, 1, 2)?
            .transpose(2, 3)?;
        let source = input
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let conv = BandedConv2d {
            weight: Tensor::zeros((3, 3, 1, 1), DType::F32, device)?,
            bias: Some(Tensor::from_vec(vec![1.0f32, -2.0, 0.5], 3, device)?),
            padding: 0,
            stride: 1,
            kernel: 1,
            max_scratch_elements: 1,
        };
        let actual = conv
            .add_bias(input)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let biases = [1.0f32, -2.0, 0.5];
        let mut expected = Vec::new();
        for image in source.chunks_exact(24) {
            for (plane, &bias) in image.chunks_exact(8).zip(&biases) {
                expected.extend(plane.iter().map(|value| value + bias));
            }
        }
        assert_eq!(actual, expected);
        // A non-four-element plane and a non-vector-aligned offset both use
        // Candle's broadcast fallback on Metal.
        for (height, width, offset) in [(1, 5, 0), (2, 4, 1)] {
            let count = 3 * height * width;
            let source = Tensor::arange(0f32, (count + offset) as f32, device)?
                .narrow(0, offset, count)?
                .reshape((1, 3, height, width))?;
            let actual = conv
                .add_bias(source.clone())?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let source = source
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut expected = Vec::new();
            for (plane, &bias) in source.chunks_exact(height * width).zip(&biases) {
                expected.extend(plane.iter().map(|value| value + bias));
            }
            assert_eq!(actual, expected);
        }
        Ok(())
    }

    #[test]
    fn channel_bias_matches_explicit_reference_for_batch_and_views() -> Result<()> {
        check_bias_reference(&Device::Cpu)
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a real Metal device"]
    fn metal_channel_bias_matches_explicit_reference_for_batch_and_views() -> Result<()> {
        check_bias_reference(&Device::new_metal(0)?)
    }

    fn exact(a: &Tensor, b: &Tensor) -> Result<bool> {
        Ok(a.dims() == b.dims()
            && (a - b)?
                .abs()?
                .flatten_all()?
                .to_dtype(DType::F64)?
                .max(0)?
                .to_scalar::<f64>()?
                == 0.0)
    }

    #[test]
    fn banded_convolution_is_band_independent_and_matches_candle() -> Result<()> {
        let device = &Device::Cpu;
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
            let out_width = (width + 2 * padding - kernel) / stride + 1;
            let limit = 2 * out_width * 6 * kernel * kernel * rows;
            for dtype in [DType::F32, DType::F64] {
                // F32 runs the channel-major column path; F64 runs the generic
                // banded fallback used for other dtypes and devices.
                // Small dyadic coefficients make every product and sum
                // exact in F32 and F64, including Accelerate. This isolates
                // band geometry from backend-dependent reduction rounding.
                let weight = Tensor::from_vec(
                    (0..5 * 6 * kernel * kernel)
                        .map(|i| ((i * 7 % 11) as f32 - 5.) / 32.)
                        .collect::<Vec<_>>(),
                    (5, 6, kernel, kernel),
                    device,
                )?
                .to_dtype(dtype)?;
                let bias =
                    Tensor::from_vec(vec![-0.125f32, 0.0625, 0., 0.125, -0.0625], 5, device)?
                        .to_dtype(dtype)?;
                let vs = VarBuilder::from_tensors(
                    std::collections::HashMap::from([
                        (format!("{index}.weight"), weight),
                        (format!("{index}.bias"), bias),
                    ]),
                    dtype,
                    device,
                );
                let conv = conv2d(6, 5, kernel, stride, padding, vs.pp(index))?;
                let input = Tensor::from_vec(
                    (0..2 * 6 * height * width)
                        .map(|i| ((i * 13 % 41) as f32 - 20.) / 16.)
                        .collect::<Vec<_>>(),
                    (2, 6, height, width),
                    device,
                )?
                .to_dtype(dtype)?;
                let banded = conv.clone().with_scratch_limit(limit).forward(&input)?;
                let one_band = conv
                    .clone()
                    .with_scratch_limit(usize::MAX)
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
                    exact(&banded, &one_band)?,
                    "{dtype:?} banding changes case {index}"
                );
                if dtype == DType::F64 {
                    assert!(
                        exact(&banded, &direct)?,
                        "F64 fallback differs from Candle for case {index}"
                    );
                } else {
                    let difference = (banded - direct)?
                        .abs()?
                        .flatten_all()?
                        .max(0)?
                        .to_scalar::<f32>()?;
                    assert!(
                        difference <= 1e-5,
                        "F32 case {index} differs by {difference}"
                    );
                }
            }
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
