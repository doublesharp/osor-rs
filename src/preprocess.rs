//! The reference CLI's resize planning, separate from the tensor inference API.

use crate::{Error, Result};

/// Preserve the reference's short-side resize rule and round both axes up to 16.
///
/// Smaller images are not scaled up to `resolution`. This plans dimensions;
/// applications perform image/mask resampling and check inference admission.
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
