//! Input geometry helpers, separate from the tensor inference API.
//!
//! [`resize_reference_inputs`] processes raw RGB8 and grayscale mask bytes
//! according to the upstream CLI; [`reference_output_rgb8`] restores the
//! decoded result with optional predicted-alpha paste-back. These helpers are
//! separate from the normalized NCHW F32 tensor inference API.
//!
//! As an alternative that keeps source pixels unresampled,
//! [`pad_to_model_size`] pads to the model's size rules and
//! [`Padding::crop_result`] removes the padding from the output. Either choice
//! changes results near the edges; applications pick and validate the policy.

use candle_core::{DType, Tensor};

mod bytes;
pub use bytes::{
    ImageSize, MAX_PREPROCESS_AXIS, MAX_PREPROCESS_PIXELS, ReferenceInputs, quantize_alpha_f32,
    quantize_rgb_chw_f32, reference_output_rgb8, resize_alpha8, resize_mask8,
    resize_reference_inputs, resize_rgb8,
};

use crate::{
    Error, InferenceResult, Result,
    pipeline::{IMAGE_AXIS_MULTIPLE, MAX_IMAGE_AXIS, MIN_IMAGE_AXIS},
};

/// Original and padded dimensions recorded by [`pad_to_model_size`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Padding {
    /// Source height before padding.
    pub height: usize,
    /// Source width before padding.
    pub width: usize,
    /// Padded height passed to the model.
    pub padded_height: usize,
    /// Padded width passed to the model.
    pub padded_width: usize,
}

fn padded_axis(axis: usize, name: &str) -> Result<usize> {
    if axis == 0 || axis > MAX_IMAGE_AXIS {
        return Err(Error::InvalidInput(format!(
            "{name} must be in 1..={MAX_IMAGE_AXIS} before padding; resize or crop larger images"
        )));
    }
    Ok(axis
        .next_multiple_of(IMAGE_AXIS_MULTIPLE)
        .max(MIN_IMAGE_AXIS))
}

/// Pad an F32 `[B, 3, H, W]` image and `[B, 1, H, W]` mask on the bottom and
/// right to the model's size rules: multiples of eight, at least 32, at most
/// 1024 per axis. Edge pixels are repeated in both tensors, so a masked object
/// touching the border stays masked in the padding.
pub fn pad_to_model_size(image: &Tensor, mask: &Tensor) -> Result<(Tensor, Tensor, Padding)> {
    let (batch, channels, height, width) = image
        .dims4()
        .map_err(|_| Error::InvalidInput("image must have shape [B,3,H,W]".into()))?;
    if channels != 3
        || mask.dims() != [batch, 1, height, width]
        || image.dtype() != DType::F32
        || mask.dtype() != DType::F32
        || !image.device().same_device(mask.device())
    {
        return Err(Error::InvalidInput(
            "pad expects an F32 [B,3,H,W] image and matching [B,1,H,W] mask on one device".into(),
        ));
    }
    let padding = Padding {
        height,
        width,
        padded_height: padded_axis(height, "height")?,
        padded_width: padded_axis(width, "width")?,
    };
    let pad = |tensor: &Tensor| -> Result<Tensor> {
        Ok(tensor
            .pad_with_same(2, 0, padding.padded_height - height)?
            .pad_with_same(3, 0, padding.padded_width - width)?)
    };
    Ok((pad(image)?, pad(mask)?, padding))
}

impl Padding {
    /// Whether any padding was added.
    pub fn is_padded(&self) -> bool {
        self.height != self.padded_height || self.width != self.padded_width
    }

    /// Crop a `[B, C, padded_height, padded_width]` tensor to `[B, C, height, width]`.
    pub fn crop(&self, tensor: &Tensor) -> Result<Tensor> {
        let (_, _, height, width) = tensor.dims4()?;
        if (height, width) != (self.padded_height, self.padded_width) {
            return Err(Error::InvalidInput(format!(
                "expected padded size {}x{}, got {height}x{width}",
                self.padded_height, self.padded_width
            )));
        }
        Ok(tensor
            .narrow(2, 0, self.height)?
            .narrow(3, 0, self.width)?
            .contiguous()?)
    }

    /// Crop image and alpha to the source size, and latent alpha to the
    /// latent cells that cover it (`ceil(height / 8)` by `ceil(width / 8)`).
    pub fn crop_result(&self, result: InferenceResult) -> Result<InferenceResult> {
        let latent = |axis: usize| axis.div_ceil(IMAGE_AXIS_MULTIPLE);
        let (_, _, latent_height, latent_width) = result.latent_alpha.dims4()?;
        if (latent_height, latent_width) != (latent(self.padded_height), latent(self.padded_width))
        {
            return Err(Error::InvalidInput(
                "latent alpha does not match the padded size".into(),
            ));
        }
        Ok(InferenceResult {
            image: self.crop(&result.image)?,
            alpha: self.crop(&result.alpha)?,
            latent_alpha: result
                .latent_alpha
                .narrow(2, 0, latent(self.height))?
                .narrow(3, 0, latent(self.width))?
                .contiguous()?,
        })
    }
}

/// Preserve the reference's short-side resize rule and round both axes up to 16.
///
/// Smaller images are not scaled up to `resolution`. This plans dimensions;
/// use [`resize_reference_inputs`] for byte resampling and check inference
/// admission separately.
pub fn reference_resize_dimensions(width: u32, height: u32, resolution: u32) -> Result<(u32, u32)> {
    if width == 0 || height == 0 || resolution == 0 {
        return Err(Error::InvalidInput(
            "image dimensions and resolution must be positive".into(),
        ));
    }
    let (width, height, resolution) = (u64::from(width), u64::from(height), u64::from(resolution));
    let (width, height) = if width.min(height) < resolution {
        (width, height)
    } else if width < height {
        (
            resolution,
            height
                .checked_mul(resolution)
                .ok_or_else(|| Error::InvalidInput("resize dimensions overflow".into()))?
                / width,
        )
    } else {
        (
            width
                .checked_mul(resolution)
                .ok_or_else(|| Error::InvalidInput("resize dimensions overflow".into()))?
                / height,
            resolution,
        )
    };
    let round = |value: u64| -> Result<u32> {
        let rounded = value
            .checked_add(15)
            .map(|value| value / 16 * 16)
            .ok_or_else(|| Error::InvalidInput("rounded resize dimension overflows".into()))?;
        u32::try_from(rounded)
            .map_err(|_| Error::InvalidInput("resize dimensions exceed u32".into()))
    };
    Ok((round(width)?, round(height)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn ramp(shape: (usize, usize, usize, usize)) -> Result<Tensor> {
        let count = shape.0 * shape.1 * shape.2 * shape.3;
        let values: Vec<f32> = (0..count).map(|i| (i % 251) as f32 / 251.0).collect();
        Ok(Tensor::from_vec(values, shape, &Device::Cpu)?)
    }

    #[test]
    fn odd_sizes_pad_with_edges_and_crop_back_exactly() -> Result<()> {
        let image = ramp((2, 3, 47, 101))?;
        let mask = ramp((2, 1, 47, 101))?;
        let (padded_image, padded_mask, padding) = pad_to_model_size(&image, &mask)?;
        assert_eq!(
            padding,
            Padding {
                height: 47,
                width: 101,
                padded_height: 48,
                padded_width: 104,
            }
        );
        assert!(padding.is_padded());
        assert_eq!(padded_image.dims(), &[2, 3, 48, 104]);
        assert_eq!(padded_mask.dims(), &[2, 1, 48, 104]);
        // The padded row and columns repeat the last source row and column.
        let last_row = image.narrow(2, 46, 1)?.narrow(3, 0, 101)?;
        let padded_row = padded_image.narrow(2, 47, 1)?.narrow(3, 0, 101)?;
        assert_eq!(
            last_row.flatten_all()?.to_vec1::<f32>()?,
            padded_row.flatten_all()?.to_vec1::<f32>()?
        );
        let last_column = mask.narrow(3, 100, 1)?;
        let padded_column = padded_mask.narrow(3, 103, 1)?.narrow(2, 0, 47)?;
        assert_eq!(
            last_column.flatten_all()?.to_vec1::<f32>()?,
            padded_column.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            padding
                .crop(&padded_image)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            image.flatten_all()?.to_vec1::<f32>()?
        );
        let result = padding.crop_result(InferenceResult {
            image: padded_image.clone(),
            alpha: padded_mask.clone(),
            latent_alpha: Tensor::zeros((2, 1, 6, 13), DType::F32, &Device::Cpu)?,
        })?;
        assert_eq!(result.image.dims(), &[2, 3, 47, 101]);
        assert_eq!(result.alpha.dims(), &[2, 1, 47, 101]);
        assert_eq!(result.latent_alpha.dims(), &[2, 1, 6, 13]);
        Ok(())
    }

    #[test]
    fn small_inputs_pad_to_the_minimum_and_aligned_inputs_are_unchanged() -> Result<()> {
        let (image, _, padding) = pad_to_model_size(&ramp((1, 3, 20, 9))?, &ramp((1, 1, 20, 9))?)?;
        assert_eq!(image.dims(), &[1, 3, 32, 32]);
        assert_eq!((padding.padded_height, padding.padded_width), (32, 32));
        let source = ramp((1, 3, 64, 40))?;
        let (image, _, padding) = pad_to_model_size(&source, &ramp((1, 1, 64, 40))?)?;
        assert!(!padding.is_padded());
        assert_eq!(
            image.flatten_all()?.to_vec1::<f32>()?,
            source.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn rejects_oversized_mismatched_or_non_f32_inputs() -> Result<()> {
        let device = &Device::Cpu;
        let mask = Tensor::zeros((1, 1, 1025, 8), DType::F32, device)?;
        let image = Tensor::zeros((1, 3, 1025, 8), DType::F32, device)?;
        assert!(matches!(
            pad_to_model_size(&image, &mask),
            Err(Error::InvalidInput(_))
        ));
        let image = Tensor::zeros((1, 3, 16, 16), DType::F32, device)?;
        let mask = Tensor::zeros((1, 1, 16, 15), DType::F32, device)?;
        assert!(matches!(
            pad_to_model_size(&image, &mask),
            Err(Error::InvalidInput(_))
        ));
        let mask = Tensor::zeros((1, 1, 16, 16), DType::F64, device)?;
        assert!(matches!(
            pad_to_model_size(&image, &mask),
            Err(Error::InvalidInput(_))
        ));
        let (_, _, padding) = pad_to_model_size(&image, &mask.to_dtype(DType::F32)?)?;
        assert!(
            padding
                .crop(&Tensor::zeros((1, 3, 16, 16), DType::F32, device)?)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn reference_aspect_rounding_and_no_short_side_upscale() -> Result<()> {
        assert_eq!(reference_resize_dimensions(1920, 1080, 512)?, (912, 512));
        assert_eq!(reference_resize_dimensions(1080, 1920, 512)?, (512, 912));
        assert_eq!(reference_resize_dimensions(640, 480, 512)?, (640, 480));
        assert_eq!(reference_resize_dimensions(640, 480, 256)?, (352, 256));
        assert_eq!(reference_resize_dimensions(101, 2000, 512)?, (112, 2000));
        assert!(reference_resize_dimensions(0, 1, 512).is_err());
        assert!(reference_resize_dimensions(u32::MAX, 1, 512).is_err());
        Ok(())
    }
}
