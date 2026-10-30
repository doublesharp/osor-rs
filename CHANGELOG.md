# Changelog

## 0.1.0 — 2026-10-29

Initial release of the `osor` crate. The repository remains `osor-rs`.

### Packaging and documentation

- Crates.io metadata, curated package contents, CPU-only docs.rs builds and a
  manual publication workflow that runs the full CI suite before uploading.
- Model-download/conversion instructions and a compilable API quick start.

### Inference and artifacts

- Native OSOR SDXL Phase II inference with CPU, Metal and optional WGPU/CUDA
  backends; no Python runtime, implicit model download or bundled weights.
- Pinned local safetensors bundles, offline LoRA merging, strict graph
  validation, trusted manifest digests and caller-owned decoded-weight limits.
- Capability-based conversion with held input/output handles, explicit limits
  and cooperative cancellation.
- Full decoded RGB, full-resolution predicted alpha and latent alpha outputs,
  preserving posterior sampling, explicit noise and latent alpha blending.
- Typed observation of 18 inference stages, synchronized profiling,
  cooperative cancellation and versioned, platform-independent seeded noise.
- Builder-based loading, explicit backend selection and effective inference
  limits. Native batches support up to four images; WGPU admits batch one
  with at most 262,144 pixels.

### Preprocessing and execution

- Native RGB8/grayscale resizing, padding, cropping, quantization and optional
  predicted-alpha paste-back. The Pillow adaptation retains its MIT-CMU notice;
  the package declares `MIT AND MIT-CMU`.
- Bounded attention score tensors and banded VAE convolution, cached fixed
  prompt cross-attention, and overlapped bundle hashing.
- Channel-major CPU F32 convolution, Metal F32 VAE kernels, distinct-image
  VAE batch fixes and fallible worker creation.
- Optional Apple Accelerate and WGPU autotuning; see
  [performance](docs/performance.md) and [WGPU](docs/wgpu.md) for measured scope.
- Candle 0.11.0, safetensors 0.8.0 and optional Burn 0.22.1.

### Qualification

- Earlier snapshots pass all 18 stages on the two original reference cases
  across CPU, Accelerate, Metal and WGPU with unchanged frozen tolerances.
  Later attention, cancellation, validation-transfer and WGPU-preflight changes
  require their own verification; broader batch/resolution cases still exceed
  the frozen limits. See the exact snapshots in [qualification](docs/qualification.md).
- Native Linux/Windows full-model execution, CUDA device execution, reduced
  precision and broader image-quality acceptance remain open.
- CI covers portable CPU and WGPU source checks, Metal/Accelerate and CUDA
  compilation, and dependency policy. CI success and real-model qualification
  are separate gates.
