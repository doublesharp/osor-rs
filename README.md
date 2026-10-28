<div align="center">
  <img src="docs/assets/osor-logo.png" alt="OSOR logo" width="132">

  # OSOR

  **Object removal, native to Rust.**

  An experimental library for OSOR SDXL-Inpainting Phase II inference.

  ![Rust 1.95.0](https://img.shields.io/badge/Rust-1.95.0-C94B2C?style=flat-square) ![License MIT](https://img.shields.io/badge/license-MIT-343434?style=flat-square) ![Optional backends Metal, CUDA and WGPU](https://img.shields.io/badge/GPU-opt--in-343434?style=flat-square)

  [Status](#status) · [Install](#install) · [Use from Rust](#use-from-rust) · [Model bundles](#model-bundles) · [Backends](#backends) · [Development](#development)

  Maintained by 0xdoublesharp
</div>

---

Supply an image and a removal mask to receive the completed image and the model's predicted edit alpha. The edit can include shadows, reflections, and other effects beyond the painted mask. The runtime uses Candle 0.9.1, with an optional Burn 0.22.0 WGPU backend. It loads explicitly installed local safetensors bundles; there is no Python sidecar, libtorch dependency, runtime text encoder, or automatic model download.

The numerical recipe is pinned to [OSOR revision `d5b68e8`](https://github.com/Zhouqm-Git/osor/tree/d5b68e822996d61aae6ba612e5e9b5a14f576879).

## Status

**Experimental · real-model inference works on CPU, Metal, and WGPU on an Apple M2 Max.**

The named FP32 cases pass all 18 reference stages with measured, frozen tolerances. See the [qualification record](docs/qualification.md) for the evidence and its scope.

<details>
<summary>Acceptance scope and remaining platform gates</summary>

CPU, Metal, and WGPU pass all 18 reference stages on an independent 80×96 input against frozen small-case FP32 limits. The real-model missing-weight rejection test also passes.

An independent 640×384 photo passes all 18 stages on the same backends against a separately calibrated, frozen FP32 envelope with elementwise and RMS limits. Larger photos expose greater cross-backend FP32 sensitivity; this envelope is separate from the tighter small-case limits. These are bounded case results, not universal parity or image-quality claims. See the [qualification record](docs/qualification.md), [acceptance guide](tests/README.md), and [remaining work](docs/milestones.md).

Linux and Windows CPU cross-checks pass, but native runtime acceptance on those platforms and CUDA device qualification remain open. PhotoCraft integration is a separate project.

</details>

## Install

Use Rust **1.95.0**. The crate is not published to crates.io. Add the private Git repository at the verified source revision, using credentials with access to it:

```toml
[dependencies.osor-rs]
git = "https://github.com/doublesharp/osor-rs"
rev = "0263f15c9ddd0067f7224c686330e0d6e308fca6"
# features = ["metal"]  # Enable for macOS Metal.
```

<details>
<summary>Develop from a local checkout</summary>

```toml
[dependencies]
osor-rs = { path = "../osor-rs" }
# Optional macOS GPU build:
# osor-rs = { path = "../osor-rs", features = ["metal"] }
```

</details>

A prepared [model bundle](#model-bundles) is required; the crate does not download or include weights.

## Use from Rust

Load an `Osor` once with `Osor::load(bundle_path)` and retain it across jobs. Pass the loaded model and your tensors into synchronous inference:

```rust
use osor_rs::{InferenceInput, InferenceResult, Osor, Result, Tensor};

fn remove_object(
    model: &Osor,
    image: &Tensor,
    mask: &Tensor,
    posterior_noise: &Tensor,
    diffusion_noise: &Tensor,
) -> Result<InferenceResult> {
    model.infer(InferenceInput {
        image,
        mask,
        posterior_noise,
        diffusion_noise,
    })
}
```

Inputs are NCHW tensors on `model.device()`:

| Input | Shape | Values |
|---|---|---|
| `image` | `[B, 3, H, W]` | RGB F32 in `[0, 1]` |
| `mask` | `[B, 1, H, W]` | F32 in `[0, 1]`; white means removal |
| `posterior_noise` | `[B, 4, H/8, W/8]` | Explicit VAE posterior noise, model dtype |
| `diffusion_noise` | `[B, 4, H/8, W/8]` | Explicit diffusion noise, model dtype |

Height and width must be multiples of eight in **32..=1024**. CPU and native GPU admission supports batches up to four, subject to the configured total pixel limit. WGPU supports batch one and at most **262,144 pixels**, including rectangular inputs. These bounds do not guarantee that a job fits available memory.

Supply independent standard-normal noise tensors for ordinary inference. Preserve their actual values for reproducibility: identical seeds across frameworks do not guarantee identical random tensors.

`InferenceResult` contains:

| Field | Shape | Meaning |
|---|---|---|
| `image` | `[B, 3, H, W]` | Full decoded RGB result in `[0, 1]` |
| `alpha` | `[B, 1, H, W]` | Full-resolution predicted edit alpha |
| `latent_alpha` | `[B, 1, H/8, W/8]` | Alpha used for latent blending |

The image already includes alpha blending **in latent space before decoding**. The library returns the full image without RGB paste-back. Restricting it to the original mask can discard intended shadow or reflection removal; applications must choose and validate their compositing policy.

### Loading and integrity

To select a GPU, call `Osor::load_with_options` and set `LoadOptions::backend` to `Backend::Metal { ordinal: 0 }`, `Backend::Cuda { ordinal: 0 }`, or `Backend::Wgpu { ordinal: 0 }`. Enable the corresponding Cargo feature. An unavailable requested backend returns an error.

For an application that pins an imported bundle, use `Osor::load_with_options_and_manifest_sha256(bundle_path, options, trusted_manifest_sha256)`. The loader verifies the exact manifest bytes it parses before opening or allocating weight tensors, then verifies tensor bytes against that manifest. Keep the trusted digest from your import or approval step; hashing a replaceable path immediately before loading does not establish a trusted pin. Existing `manifest()`, `device()`, and `dtype()` getters are public. See the [integrity contract](docs/artifacts.md).

Applications with their own admission budget should use `Osor::load_with_options_and_manifest_sha256_and_limits(bundle_path, options, trusted_manifest_sha256, &artifact_limits)`. Set `ArtifactLimits::max_total_tensor_bytes` to the allowed aggregate decoded-weight bytes; F16 inputs loaded as F32 count at the F32 size. Each file's header is checked against the remaining budget before allocating that file's weights. This weight bound does not cover activation, conversion-buffer, allocator-cache, or other process memory, and is not an RSS bound.

### Jobs and cancellation

- `infer` runs one synchronous job.
- `infer_with_observer` exposes 18 named intermediate stages. Return `Error::Cancelled` from the observer to cancel between stages; an executing neural layer is synchronous.
- `infer_profiled` records synchronized stage timings.
- `preprocess::reference_resize_dimensions` plans the upstream CLI resize dimensions. Image resampling and crop selection remain application responsibilities.

Applications own model installation, memory admission, concurrency, selections, jobs, and undo. See the complete [local inference example](examples/local_inference.rs).

## Model bundles

The library expects this local directory:

```text
bundle/
├── manifest.json
├── unet.safetensors
├── vae.safetensors
└── prompt.safetensors
```

It contains the base model with Phase II rank-256 LoRA adapters merged into its UNet, the trained five-channel output head, the VAE, and cached fixed prompt embeddings. A generic SDXL checkpoint or LoRA file alone is insufficient.

The offline Rust converter accepts a conversion plan and local safetensors exports:

```sh
scripts/cargo.sh run --locked --bin osor -- convert /path/to/conversion.json
```

It validates critical weights and shapes, records source revisions and SHA-256 hashes, and creates a new bundle. Export released pickle-based checkpoints only in an isolated development environment outside this repository; the Rust converter never executes pickle. See the [artifact contract](docs/artifacts.md) for the conversion schema, provenance, and licensing requirements.

<details>
<summary>Run the local inference example and CLI</summary>

With a prepared local bundle and input capture, use the checked-in lockfile:

```sh
scripts/cargo.sh run --locked --example local_inference -- \
  /path/to/bundle /path/to/input.safetensors /path/to/new-output.safetensors
```

The example runs CPU inference. The input contains the four tensors described above; the output contains `image` and full-resolution `alpha`. Output paths must be new files.

For Metal on macOS:

```sh
scripts/cargo.sh run --locked --features metal --bin osor -- infer \
  /path/to/bundle /path/to/input.safetensors /path/to/new-output.safetensors metal
```

For WGPU, use `--features wgpu` and replace the final `metal` argument with `wgpu`.

</details>

## Backends

| Backend | Cargo feature | Current boundary |
|---|---|---|
| CPU | Default | F32; portable source; macOS real-model execution |
| Metal | `metal` | macOS; M2 Max small-case and independent photo acceptance |
| CUDA | `cuda` | Requires an NVIDIA device, compatible driver, toolkit, and host compiler; device acceptance pending |
| WGPU | `wgpu` | F32; resident GPU graph with CPU staging; M2 Max small-case and independent photo acceptance |

Native Metal/CUDA APIs also accept F16 and BF16, but reduced precision needs separate numerical qualification. F32 is the current acceptance baseline.

WGPU keeps UNet and VAE activations on the GPU, while noise, mask, scheduler, and alpha operations use CPU staging tensors. Consequently `model.device()` is CPU for WGPU; `compute_device_description()` identifies the actual adapter. See [WGPU details](docs/wgpu.md) and [platform setup](docs/platforms.md).

## Numerical checks and performance

The port preserves posterior sampling, 8× max-pooled masks, timestep 400, nine-channel UNet input, added text/time conditioning, five-channel output, scheduler predicted-original computation, and latent alpha blending. See the [numerical contract](docs/reference-contract.md).

On the measured 640×384 photo, warm inference averaged **0.74 s for upstream PyTorch MPS, 5.85 s for Rust Metal, and 20.49 s for Rust WGPU**. These are two-sample, shared-host observations from the optimized Rust development build, with loading excluded. The portable Rust CPU path also needs substantial optimization. See the [performance record](docs/performance.md) for build settings, first/warm timings, memory, and measured bottlenecks; release benchmarking remains open.

<details>
<summary>Capture, compare, and profile stages</summary>

Capture and compare all reference stages:

```sh
scripts/cargo.sh run --locked --bin osor -- infer \
  /path/to/bundle /path/to/input.safetensors /path/to/new-output.safetensors \
  cpu /path/to/new-stages.safetensors

scripts/cargo.sh run --locked --bin osor -- compare \
  /path/to/reference.safetensors /path/to/new-stages.safetensors \
  /path/to/tolerances.json
```

Comparison requires complete matching stage sets and explicit per-stage tolerances. It rejects nonfinite values and exits nonzero on failure. Optional RMS caps supplement the elementwise limits. The [acceptance guide](tests/README.md) describes opt-in real-model tests.

Measure loading, first inference, warm inference, and stage costs with a retained model:

```sh
scripts/cargo.sh run --locked --bin osor-profile -- \
  /path/to/bundle /path/to/input.safetensors cpu 3
```

</details>

## Development

```sh
scripts/check.sh

# Optional accelerator source and ordinary-test gates:
scripts/cargo.sh clippy --locked --all-targets --features metal,wgpu -- -D warnings
scripts/cargo.sh test --locked --all-targets --features metal,wgpu
```

The wrapper pins Rust 1.95.0, uses sccache when installed, and puts builds and temporary files on `/Volumes/Scratch/osor-rs`. It stops if Scratch is missing on this machine. Set `OSOR_SCRATCH_ROOT` explicitly on other hosts. Normal tests use small synthetic tensors and never download model weights. GPU and full-model tests require explicit local artifacts and opt-in execution.

Keep models, reference source, tensors, images, captures, and reports outside Git. Source checks and cross compilation establish different evidence from native device execution and real-model acceptance.

## License

Project code is MIT. Adapted Candle code retains its MIT notice; dependencies and upstream sources have their own terms. Model weights have separate licenses, and this repository does not embed or redistribute them. See [attribution](ATTRIBUTION.md), [artifact provenance](docs/artifacts.md), and the preserved license notices.
