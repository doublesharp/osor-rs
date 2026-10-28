use crate::{
    Error, Result,
    preprocess::{
        ImageSize, quantize_alpha_f32, quantize_rgb_chw_f32, reference_output_rgb8, resize_alpha8,
        resize_mask8, resize_reference_inputs, resize_rgb8,
    },
};

#[path = "oracle_test.rs"]
mod oracle_test;

const fn size(width: u32, height: u32) -> ImageSize {
    ImageSize { width, height }
}

#[test]
fn pillow_rgb_resizes_and_identity() -> Result<()> {
    let source = [0, 10, 255, 255, 30, 0, 20, 240, 40, 100, 120, 140];
    assert_eq!(resize_rgb8(&source, size(2, 2), size(2, 2))?, source);
    assert_eq!(
        resize_rgb8(&source, size(2, 2), size(3, 3))?,
        [
            0, 0, 255, 132, 9, 130, 255, 26, 0, 8, 129, 144, 94, 100, 109, 180, 72, 74, 16, 255,
            18, 56, 191, 88, 95, 117, 157,
        ]
    );
    assert_eq!(
        resize_rgb8(&source, size(2, 2), size(1, 1))?,
        [94, 100, 109]
    );
    Ok(())
}

#[test]
fn pillow_13_uses_vertical_first_for_strong_asymmetric_reduction() -> Result<()> {
    let source: Vec<u8> = (0u8..42)
        .map(|i| i.wrapping_mul(47).wrapping_add(11))
        .collect();
    assert_eq!(
        resize_rgb8(&source, size(2, 7), size(3, 2))?,
        [
            39, 89, 151, 115, 140, 116, 191, 191, 81, 134, 184, 170, 98, 123, 135, 62, 62, 100
        ]
    );
    Ok(())
}

#[test]
fn mask_stays_grayscale_and_alpha_uses_bilinear() -> Result<()> {
    let mask = [0, 64, 127, 128, 200, 255];
    assert_eq!(
        resize_mask8(&mask, size(3, 2), size(5, 3))?,
        [
            0, 0, 64, 127, 127, 128, 128, 200, 255, 255, 128, 128, 200, 255, 255
        ]
    );
    assert_eq!(
        resize_alpha8(&mask, size(3, 2), size(5, 3))?,
        [
            0, 26, 64, 102, 127, 64, 92, 132, 168, 191, 128, 157, 200, 233, 255
        ]
    );
    // Pillow advances floating-point coordinates; direct center formulas
    // choose a different source pixel at the middle tie for 2 -> 7.
    assert_eq!(
        resize_mask8(&[0, 255], size(2, 1), size(7, 1))?,
        [0, 0, 0, 0, 255, 255, 255]
    );
    Ok(())
}

#[test]
fn output_is_full_by_default_and_paste_back_uses_predicted_alpha() -> Result<()> {
    let original = [0, 50, 255, 200, 100, 0, 10, 20, 30, 255, 200, 100];
    let generated = [255, 0, 50, 0, 200, 100, 30, 10, 20, 0, 0, 0];
    let alpha = [0, 64, 128, 255];
    assert_eq!(
        reference_output_rgb8(&generated, size(2, 2), &original, size(2, 2), None)?,
        generated
    );
    assert_eq!(
        reference_output_rgb8(
            &generated,
            size(2, 2),
            &original,
            size(2, 2),
            Some((&alpha, size(2, 2)))
        )?,
        [0, 50, 255, 150, 125, 25, 20, 15, 25, 0, 0, 0]
    );
    Ok(())
}

#[test]
fn normalized_chw_quantization_clamps_and_truncates() -> Result<()> {
    let values = [-0.5, 0.5, 1.1, 0.25, 1.0, 1.0 / 255.0];
    assert_eq!(
        quantize_rgb_chw_f32(&values, size(2, 1))?,
        [0, 255, 255, 127, 63, 1]
    );
    assert_eq!(
        quantize_alpha_f32(&[0.0, 0.5, 1.0], size(3, 1))?,
        [0, 127, 255]
    );
    Ok(())
}

#[test]
fn reference_inputs_use_independent_mask_geometry_and_preserve_gray() -> Result<()> {
    let resized = resize_reference_inputs(&[10, 20, 30], size(1, 1), &[17, 204], size(2, 1), 32)?;
    assert_eq!(resized.size, size(16, 16));
    assert!(
        resized
            .image_rgb8
            .chunks_exact(3)
            .all(|rgb| rgb == [10, 20, 30])
    );
    assert!(resized.mask8.chunks_exact(16).all(|row| {
        row.iter().take(8).all(|&value| value == 17)
            && row.iter().skip(8).all(|&value| value == 204)
    }));
    Ok(())
}

#[test]
fn rejects_bad_dimensions_buffers_and_nonfinite_values() {
    assert!(matches!(
        resize_rgb8(&[], size(0, 1), size(1, 1)),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        resize_rgb8(&[1, 2], size(1, 1), size(1, 1)),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        resize_mask8(&[], size(u32::MAX, 1), size(1, 1)),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        resize_alpha8(&[0], size(1, 1), size(16384, 16384)),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        quantize_rgb_chw_f32(&[0.0, f32::NAN, 0.0], size(1, 1)),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        quantize_alpha_f32(&[1.01], size(1, 1)),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        quantize_alpha_f32(&[f32::NAN], size(1, 1)),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        resize_reference_inputs(&[1, 2, 3], size(1, 1), &[], size(1, 1), 32),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        resize_reference_inputs(&[1, 2, 3], size(1, 1), &[0], size(1, 1), 0),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        reference_output_rgb8(&[1, 2, 3], size(1, 1), &[], size(1, 1), None),
        Err(Error::InvalidInput(_))
    ));
    assert!(matches!(
        reference_output_rgb8(
            &[1, 2, 3],
            size(1, 1),
            &[4, 5, 6],
            size(1, 1),
            Some((&[], size(1, 1)))
        ),
        Err(Error::InvalidInput(_))
    ));
}
