# Reference image preprocessing

These byte APIs are available in the unreleased checkout after v0.2.0.

The tensor inference API accepts normalized NCHW F32 RGB and grayscale mask tensors. The byte helpers in `osor::preprocess` reproduce the pinned upstream CLI's surrounding image operations using native Rust. They do not decode image files, download models, or load Python.

Call `resize_reference_inputs(rgb, image_size, mask, mask_size, resolution)` with tightly packed interleaved RGB8 and grayscale mask bytes. It applies `reference_resize_dimensions`, bicubic RGB resizing, and nearest-neighbor mask resizing. The mask may have different source dimensions. Grayscale mask values are preserved; white means removal. Convert the returned RGB bytes to planar `[1, 3, H, W]` F32 and divide by 255. Convert the returned mask to `[1, 1, H, W]` F32 and divide by 255. The inference pipeline performs the later RGB normalization to `[-1, 1]` itself.

The dimension policy rounds both axes up to multiples of 16. If the source short side is at least `resolution`, it first scales that side to `resolution` while preserving aspect ratio. Otherwise it only rounds the original dimensions. Check `model.inference_limits()` separately: the reference resize policy can produce dimensions outside model admission.

After inference, copy one image's F32 RGB and full-resolution predicted alpha to the host. `quantize_rgb_chw_f32` clamps finite RGB values to `[0, 1]`, multiplies by 255 in F32, truncates, and interleaves the channels. `quantize_alpha_f32` validates `[0, 1]` and truncates the same way. Quantize **before** resizing back, as the upstream CLI does.

```rust
use osor::{Result, preprocess::{ImageSize, reference_output_rgb8}};

fn restore(
    generated_rgb8: &[u8],
    model_size: ImageSize,
    original_rgb8: &[u8],
    original_size: ImageSize,
) -> Result<Vec<u8>> {
    reference_output_rgb8(
        generated_rgb8, model_size, original_rgb8, original_size, None,
    )
}
```

`None` returns the full generated result resized bicubically to the original dimensions, matching the upstream default. To opt into RGB paste-back, pass `Some((&predicted_alpha8, alpha_size))`; the helper resizes that alpha bilinearly and composites the result over the original image with Pillow's byte rounding. Use the predicted full-resolution alpha, not the input mask or latent alpha. Application selection, undo, and effect-region policies still require consumer qualification.

All byte helpers validate positive dimensions, exact buffer lengths, allocation sizes, and finite F32 values. Each axis is limited to 16,384 and each input, output, or intermediate image to 16,777,216 pixels. These preprocessing bounds are separate from the smaller inference bounds. Malformed inputs and failed allocations return errors.

The eight-bit resampling and compositing implementation is adapted from Pillow 13.0.0 under its preserved [MIT-CMU notice](../src/preprocess/LICENSE-PILLOW). On macOS arm64, 776 independent oracle cases compared 5,912,751 output bytes with zero mismatches. The corpus includes upsampling, downsampling, asymmetric passes, odd and identity sizes, grayscale masks, F32 quantization, and all original-byte/alpha-byte pairs for three generated channel values. This is measured local parity, not universal cross-platform proof. The opt-in `preprocess::bytes::tests::oracle_test::pillow_oracle_corpus` test reads `OSOR_PREPROCESS_ORACLE`; its corpus, Python environment, source hashes, and logs are preserved on Scratch under `reports/codex-finish-20261028/pillow-byte-api`.
