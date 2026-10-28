//! Byte processing follows the OSOR reference script and Pillow 13.0.0.
//!
//! The resize/composite implementation is adapted from Pillow's `Resample.c`,
//! `Geometry.c`, `Paste.c`, and `ImagingUtils.h`. The PyPI 13.0.0 source
//! distribution has SHA-256
//! `cdcd1ba258178f7356d74170c9a81ba4eab6a5d55748bc402a5a2a6fe6be3510`.
//! See `LICENSE-PILLOW` here for the original copyright and permission notice.

use crate::{Error, Result};

#[path = "resample.rs"]
mod resample;

/// Maximum width or height accepted by the byte preprocessing helpers.
pub const MAX_PREPROCESS_AXIS: u32 = 16_384;
/// Maximum pixels in any input, output, or intermediate byte image.
pub const MAX_PREPROCESS_PIXELS: usize = 16_777_216;

/// Image width and height, in pixels. Helpers validate both on use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
}

/// Byte images resized to the dimensions selected by the reference CLI.
#[derive(Debug, PartialEq, Eq)]
pub struct ReferenceInputs {
    /// Shared output size, in width/height order.
    pub size: ImageSize,
    /// Tightly packed, row-major, interleaved RGB8 bytes.
    pub image_rgb8: Vec<u8>,
    /// Tightly packed, row-major grayscale mask bytes.
    pub mask8: Vec<u8>,
}

/// Resize tightly packed RGB8 with Pillow 13's bicubic filter.
///
/// This preserves byte processing separately from the normalized tensor API.
/// Each dimension must be positive and within [`MAX_PREPROCESS_AXIS`]; each
/// input, output, and intermediate image is limited to [`MAX_PREPROCESS_PIXELS`].
pub fn resize_rgb8(pixels: &[u8], source: ImageSize, target: ImageSize) -> Result<Vec<u8>> {
    resample::filtered(pixels, source, target, 3, resample::Filter::Bicubic)
}

/// Resize a grayscale removal mask with Pillow 13's nearest-neighbor filter.
///
/// The reference does not binarize the mask: intermediate grayscale values
/// remain intact. White means removal. Sizes and buffers have the same limits
/// as [`resize_rgb8`].
pub fn resize_mask8(pixels: &[u8], source: ImageSize, target: ImageSize) -> Result<Vec<u8>> {
    resample::nearest(pixels, source, target)
}

/// Resize predicted grayscale alpha bytes with Pillow 13's bilinear filter.
///
/// This is the *predicted edit alpha*, not the input removal mask. Sizes and
/// buffers have the same limits as [`resize_rgb8`].
pub fn resize_alpha8(pixels: &[u8], source: ImageSize, target: ImageSize) -> Result<Vec<u8>> {
    resample::filtered(pixels, source, target, 1, resample::Filter::Bilinear)
}

/// Apply the upstream CLI's dimension policy, bicubic RGB resize, and nearest
/// grayscale mask resize. The source mask may have its own dimensions.
///
/// The returned bytes are not tensors. Convert the interleaved image to NCHW
/// F32 RGB and divide each byte by 255; similarly divide each mask byte by 255
/// for `[B, 1, H, W]`. The inference API performs its own RGB normalization
/// to `[-1, 1]`. Check model admission separately: this resize policy does not
/// guarantee the model's minimum, maximum, or backend pixel limits.
pub fn resize_reference_inputs(
    rgb: &[u8],
    image_size: ImageSize,
    mask: &[u8],
    mask_size: ImageSize,
    resolution: u32,
) -> Result<ReferenceInputs> {
    resample::validate(rgb, image_size, 3, "RGB image")?;
    resample::validate(mask, mask_size, 1, "mask")?;
    let (width, height) =
        super::reference_resize_dimensions(image_size.width, image_size.height, resolution)?;
    let size = ImageSize { width, height };
    resample::layout(size, 3)?;
    Ok(ReferenceInputs {
        size,
        image_rgb8: resize_rgb8(rgb, image_size, size)?,
        mask8: resize_mask8(mask, mask_size, size)?,
    })
}

/// Convert one planar `[3, H, W]` F32 result to interleaved RGB8.
///
/// The reference clamps finite RGB values to `[0, 1]`, multiplies in F32 by
/// 255, and truncates toward zero before resizing. Nonfinite values and wrong
/// lengths are rejected. This takes one image, with no batch dimension.
pub fn quantize_rgb_chw_f32(values: &[f32], size: ImageSize) -> Result<Vec<u8>> {
    let layout = resample::layout(size, 3)?;
    if values.len() != layout.bytes || values.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidInput(
            "RGB result must contain exactly 3 * width * height finite F32 values".into(),
        ));
    }
    let (red, rest) = values
        .split_at_checked(layout.pixels)
        .ok_or_else(|| Error::InvalidInput("RGB result has an invalid channel layout".into()))?;
    let (green, blue) = rest
        .split_at_checked(layout.pixels)
        .ok_or_else(|| Error::InvalidInput("RGB result has an invalid channel layout".into()))?;
    let mut output = resample::zeroed::<u8>(layout.bytes)?;
    for (((pixel, &r), &g), &b) in output.chunks_exact_mut(3).zip(red).zip(green).zip(blue) {
        for (byte, value) in pixel.iter_mut().zip([r, g, b]) {
            *byte = (value.clamp(0.0, 1.0) * 255.0) as u8;
        }
    }
    Ok(output)
}

/// Convert one full-resolution `[H, W]` predicted F32 alpha to grayscale8.
///
/// Values must be finite and in `[0, 1]`. Like the reference, multiplication
/// is in F32 and conversion truncates rather than rounds. The caller supplies
/// the already-upsampled alpha returned by inference, not latent alpha.
pub fn quantize_alpha_f32(values: &[f32], size: ImageSize) -> Result<Vec<u8>> {
    let layout = resample::layout(size, 1)?;
    if values.len() != layout.bytes
        || values
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(Error::InvalidInput(
            "alpha must contain exactly width * height finite F32 values in [0, 1]".into(),
        ));
    }
    let mut output = resample::zeroed::<u8>(layout.bytes)?;
    for (byte, &value) in output.iter_mut().zip(values) {
        *byte = (value * 255.0) as u8;
    }
    Ok(output)
}

/// Restore a generated RGB8 result to the original size with bicubic resize.
///
/// `None` returns the entire resized result, matching the upstream default.
/// To explicitly opt into paste-back, pass the quantized predicted alpha and
/// its size in `Some`; it is resized bilinearly and used to composite the
/// generated result over `original` with Pillow's byte rounding. Paste-back
/// may discard model edits beyond the selected alpha region. The input mask
/// is not used for compositing. The original RGB buffer is validated in both
/// modes; all buffers must be tightly packed and satisfy the resize limits.
pub fn reference_output_rgb8(
    generated: &[u8],
    generated_size: ImageSize,
    original: &[u8],
    original_size: ImageSize,
    paste_back_alpha: Option<(&[u8], ImageSize)>,
) -> Result<Vec<u8>> {
    resample::validate(generated, generated_size, 3, "generated RGB image")?;
    resample::validate(original, original_size, 3, "original RGB image")?;
    if let Some((alpha, alpha_size)) = paste_back_alpha {
        resample::validate(alpha, alpha_size, 1, "predicted alpha")?;
    }
    let mut result = resize_rgb8(generated, generated_size, original_size)?;
    if let Some((alpha, alpha_size)) = paste_back_alpha {
        let alpha = resize_alpha8(alpha, alpha_size, original_size)?;
        for ((output, source), &alpha) in result
            .chunks_exact_mut(3)
            .zip(original.chunks_exact(3))
            .zip(&alpha)
        {
            let alpha = u32::from(alpha);
            for (output, &source) in output.iter_mut().zip(source) {
                let sum = u32::from(source) * (255 - alpha) + u32::from(*output) * alpha + 128;
                *output = ((sum + (sum >> 8)) >> 8) as u8;
            }
        }
    }
    Ok(result)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
