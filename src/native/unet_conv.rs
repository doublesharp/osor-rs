//! UNet convolution selection. Only CPU F32 uses channel-major columns.

use candle_core::{DType, Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};

use super::conv::MAX_SCRATCH_ELEMENTS;

#[derive(Debug)]
pub(super) struct UnetConv2d(nn::Conv2d);

impl UnetConv2d {
    pub(super) fn new(
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        vs: VarBuilder<'_>,
    ) -> Result<Self> {
        Ok(Self(nn::conv2d(
            input,
            output,
            kernel,
            nn::Conv2dConfig {
                stride,
                padding,
                ..Default::default()
            },
            vs,
        )?))
    }
}

impl Module for UnetConv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if !input.device().is_cpu()
            || input.dtype() != DType::F32
            || self.0.weight().dtype() != DType::F32
        {
            return self.0.forward(input);
        }
        let config = self.0.config();
        let output = super::columns::conv2d(
            input,
            self.0.weight(),
            config.stride,
            config.padding,
            MAX_SCRATCH_ELEMENTS,
        )?;
        match self.0.bias() {
            Some(bias) => output.broadcast_add(&bias.reshape((1, bias.dims1()?, 1, 1))?),
            None => Ok(output),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    #[test]
    fn convolution_matches_f64_for_batch_sampling_shortcuts_and_views() -> Result<()> {
        let device = &Device::Cpu;
        let values = |count: usize, seed: usize| {
            (0..count)
                .map(|i| (((i * 41 + seed) % 101) as f32 - 50.) / 53.)
                .collect::<Vec<_>>()
        };
        for (batch, channels, height, width, outputs, kernel, stride, padding) in [
            (1, 9, 5, 7, 5, 3, 1, 1),
            (2, 4, 7, 5, 3, 3, 2, 1),
            (2, 6, 5, 7, 5, 1, 1, 0),
        ] {
            let input = Tensor::from_vec(
                values(batch * channels * height * width, 9),
                (batch, channels, width, height),
                device,
            )?
            .transpose(2, 3)?;
            let weight = Tensor::from_vec(
                values(outputs * channels * kernel * kernel, 7),
                (outputs, channels, kernel, kernel),
                device,
            )?;
            let bias = Tensor::from_vec(values(outputs, 31), outputs, device)?;
            let tensors = HashMap::from([
                ("weight".to_owned(), weight.clone()),
                ("bias".to_owned(), bias.clone()),
            ]);
            let conv = UnetConv2d::new(
                channels,
                outputs,
                kernel,
                stride,
                padding,
                VarBuilder::from_tensors(tensors, DType::F32, device),
            )?;
            let reference = nn::Conv2d::new(
                weight.to_dtype(DType::F64)?,
                Some(bias.to_dtype(DType::F64)?),
                nn::Conv2dConfig {
                    stride,
                    padding,
                    ..Default::default()
                },
            )
            .forward(&input.to_dtype(DType::F64)?)?;
            let actual = conv.forward(&input)?.to_dtype(DType::F64)?;
            assert_eq!(actual.dims(), reference.dims());
            let error = (actual - reference)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_scalar::<f64>()?;
            assert!(
                error <= 2e-5,
                "batch {batch}, kernel {kernel}, stride {stride}: {error}"
            );
        }
        Ok(())
    }

    #[test]
    fn non_f32_convolution_keeps_candle_execution() -> Result<()> {
        let device = &Device::Cpu;
        let weight = Tensor::from_vec(vec![0.25f64; 2 * 3 * 3 * 3], (2, 3, 3, 3), device)?;
        let bias = Tensor::new(&[0.5f64, -0.25], device)?;
        let input = Tensor::arange(0f64, 3. * 5. * 7., device)?.reshape((1, 3, 5, 7))?;
        let config = nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let reference = nn::Conv2d::new(weight.clone(), Some(bias.clone()), config);
        let conv = UnetConv2d::new(
            3,
            2,
            3,
            1,
            1,
            VarBuilder::from_tensors(
                HashMap::from([("weight".to_owned(), weight), ("bias".to_owned(), bias)]),
                DType::F64,
                device,
            ),
        )?;
        let difference = (conv.forward(&input)? - reference.forward(&input)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f64>()?;
        assert_eq!(difference, 0.);
        Ok(())
    }
}
