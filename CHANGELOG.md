# Changelog

## Unreleased

- Stabilize tiny-model batch checks with deterministic parameters and distinct,
  reordered images, retaining the existing F32 tolerance and conditioning checks.
- Bound native UNet and VAE attention score tensors to 256 MiB each by
  slicing query rows; weights and other activations have separate costs.
- Check cancellation between transformer operations, attention slices, and
  convolution bands. Individual kernels and opaque Candle blocks finish first.
- Check WGPU adapter availability before loading model weights and retain the
  selected device generation for graph construction.
- Validate accelerator tensors with device reductions and scalar readback,
  preserving finite/range checks and checking available storage bounds before
  conversion or dispatch.
- Add native RGB8/grayscale preprocessing, output quantization, and optional
  predicted-alpha paste-back matching the measured Pillow 13 corpus. Preserve
  the MIT-CMU notice and declare `MIT AND MIT-CMU` for the combined source.
- Optimize Metal F32 VAE normalization, bias addition and nearest upsampling.
  Separate SiLU is retained after a fused candidate failed the frozen limits.
- Use channel-major CPU F32 convolution throughout the UNet, preserving other
  device and dtype paths. Accelerate remains opt-in.
- Fix distinct-image VAE batches: materialize CPU attention projection inputs
  and evaluate Metal VAE attention per image. Batch-one paths are preserved.
- Return actionable errors if CPU convolution or bundle hashing workers cannot
  be created, with injected-failure regressions.
- Add deterministic exactness fixtures, an F64 attention oracle, Metal batch
  regressions, and opt-in WGPU VAE operator timings.
- Both original reference cases still pass all 18 stages on CPU, Metal and
  WGPU. Broader photo/batch/resolution cases still exceed the unchanged frozen
  limits; see [qualification](docs/qualification.md). No WGPU algorithm change
  or universal numerical-accuracy claim is included.

## 0.2.0 — 2026-10-28

All 18 reference stages pass on CPU, Metal and WGPU for both qualification
cases with the frozen tolerances; see [qualification](docs/qualification.md).
Timings are shared-host `profiling` builds on an Apple M2 Max; see
[performance](docs/performance.md).

### Breaking changes

- `ModelBackend` methods take a `Checkpoint`, and `denoise` no longer takes
  the prompt sequence or pooled embedding; backends bind them at construction.
- `NativeModel::new` and `WgpuModel::new` take the bundle's `sequence` and
  `pooled` prompt embeddings.
- Observers passed to `infer_with_observer` receive a `Stage` instead of a
  `&str`; `Stage::as_str` returns the pinned capture name.
- `Error` and `Backend` are `#[non_exhaustive]`.
- `InferenceLimits` outside 1..=4 images and 1..=4,194,304 pixels are rejected
  before bundle I/O. Batches above four were previously clamped silently.
- Candle types re-exported as `Tensor`, `Device` and `DType` now come from
  Candle 0.11.

### Added

- `Osor::builder` for backend, dtype, limits, trusted manifest digest and
  artifact limits. The earlier `load_with_*` functions remain.
- `Osor::inference_limits` reports the effective limits.
- `CancellationToken` and `Osor::infer_with_cancellation`, checked between
  encoder, UNet and decoder blocks on native backends.
- `noise::seeded_noise` and `Osor::seeded_noise`: versioned,
  platform-independent standard-normal noise from a `u64` seed.
- `preprocess::pad_to_model_size` and `Padding::crop_result`.
- Optional `wgpu-autotune` feature (see [WGPU](docs/wgpu.md)).
- Optional `accelerate` feature for macOS CPU matrix products. It passes both
  qualification cases but is slower than the portable build on Candle 0.11.
- `profiling` Cargo profile for measurements.
- `deny.toml` and CI jobs for `cargo deny`, Metal/Accelerate compilation and
  CUDA compilation in an NVIDIA container.
- Documentation for every public item, enforced by `missing_docs` and rustdoc
  warnings in `scripts/check.sh`.

### Performance

- Metal, 640×384 photo: 5.85 s → 2.76 s warm; peak footprint 30.6 GiB →
  11.3 GiB. VAE convolutions use a channel-major im2col kernel in bounded row
  bands, and Candle 0.11 runs contiguous binary ops on its fast kernel.
- CPU, 80×96: 21.7 s → 1.52 s warm. Most of the earlier figure was dependency
  debug assertions in the dev profile.
- WGPU, 640×384: 20.5 s → 18.2 s warm by removing fixed 64-query attention
  slicing and binding the prompt at load. The new opt-in `wgpu-autotune`
  feature reaches 16.7 s (1.05 s at 80×96) after a long first-job tuning pass.
- Cross-attention keys and values for the fixed prompt are computed once at
  load on native and WGPU graphs, releasing about 1.27 GiB of weights.
- Bundle hashing overlaps tensor construction (Metal load about 9.5 s → 6.4 s).

### Fixed

- The VAE decoder uses the encoder's bounded group normalization instead of
  Candle's single-pass sums.
- CPU VAE convolutions keep the channel-major reduction order, so Candle
  0.11's tiled CPU convolution does not move results away from the reference.

### Dependencies

- Candle 0.9.1 → 0.11.0, safetensors 0.4.5 → 0.8.0, sha2 0.10.9 → 0.11.1,
  Burn 0.22.0 → 0.22.1, pollster 0.4.0 → 1.0.1.
- Candle 0.10 and later always build `tokenizers` with the Oniguruma C library;
  cross builds from macOS use `cargo zigbuild`.

## 0.1.0

Initial native port: CPU, Metal and WGPU inference, pinned bundle loading and
conversion, and stage-by-stage reference comparison.
