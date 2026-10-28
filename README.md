<div align="center">
  <img src="docs/assets/osor-logo.png" alt="OSOR logo" width="132">

  # OSOR

  **Object removal, native to Rust.**

  Native Rust inference for OSOR SDXL-Inpainting Phase II.

  ![Rust](https://img.shields.io/badge/Rust-C94B2C?style=flat-square) ![Source licenses MIT and MIT-CMU](https://img.shields.io/badge/license-MIT%20%2B%20MIT--CMU-343434?style=flat-square) ![Backends CPU, Metal, CUDA and WGPU](https://img.shields.io/badge/CPU%20%2B%20GPU-343434?style=flat-square)

  [Features](#features) · [Install](#install) · [Use from Rust](#use-from-rust) · [Model bundles](#model-bundles) · [Backends](#backends) · [Development](#development)

  Maintained by 0xdoublesharp
</div>

---

Supply an image and a removal mask to receive the completed image and the model's predicted edit alpha. The edit can include shadows, reflections, and other effects beyond the painted mask. Candle provides CPU, Metal, and CUDA execution, with an optional Burn backend for WGPU. Models load from local safetensors bundles.

The numerical recipe is pinned to [OSOR revision `d5b68e8`](https://github.com/Zhouqm-Git/osor/tree/d5b68e822996d61aae6ba612e5e9b5a14f576879).

## Features

- Image completion with full-resolution RGB, predicted edit alpha, and latent alpha outputs.
- CPU, Metal, CUDA, and WGPU backends with explicit device selection.
- Native RGB and mask resizing, padding, output quantization, and alpha compositing.
- Explicit noise tensors and versioned seeded noise generation.
- Cooperative cancellation, 18 observable inference stages, and synchronized profiling.
- Verified local model bundles, offline LoRA merging, and checkpoint conversion.
- Attention query slicing and convolution bands to bound intermediate allocations.

## Install

Use the pinned toolchain in [rust-toolchain.toml](rust-toolchain.toml). Add the Git dependency and commit your `Cargo.lock` to retain its resolved revision:

```toml
[dependencies.osor]
git = "https://github.com/doublesharp/osor-rs"
branch = "main"
# features = ["metal"]  # Enable for macOS Metal.
```

See the [changelog](CHANGELOG.md) for release history and API changes.

<details>
<summary>Develop from a local checkout</summary>

```toml
[dependencies]
osor = { path = "../osor-rs" }
# Optional macOS GPU build:
# osor = { path = "../osor-rs", features = ["metal"] }
```

</details>

Install a [model bundle](#model-bundles) locally before loading the model.

## Use from Rust

Load an `Osor` once with `Osor::load(bundle_path)` or `Osor::builder(bundle_path)...load()` and retain it across jobs. A loaded model is `Send + Sync`. Pass it and your tensors into synchronous inference:

```rust
use osor::{InferenceInput, InferenceResult, Osor, Result, Tensor};

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

Height and width must be multiples of eight in **32..=1024**. CPU, Metal, and CUDA support batches up to four. WGPU supports batch one and at most **262,144 pixels**, including rectangular inputs. Configure `InferenceLimits` with up to four images and **4,194,304** batch-inclusive pixels; `model.inference_limits()` reports the effective limits for the selected backend.

Supply independent standard-normal noise tensors for inference. `model.seeded_noise(seed, batch, height, width)` generates both from a `u64` seed with a versioned, platform-independent algorithm (see `osor::noise`). Store the seed to reproduce noise within the library, or retain the explicit tensors for comparisons across frameworks.

`InferenceResult` contains:

| Field | Shape | Meaning |
|---|---|---|
| `image` | `[B, 3, H, W]` | Full decoded RGB result in `[0, 1]` |
| `alpha` | `[B, 1, H, W]` | Full-resolution predicted edit alpha |
| `latent_alpha` | `[B, 1, H/8, W/8]` | Alpha used for latent blending |

The image includes alpha blending **in latent space before decoding**. The full result includes the model's surrounding edits; the predicted alpha describes its edit region.

`preprocess::resize_reference_inputs` applies the reference CLI's dimension policy, bicubic RGB resize, and nearest-neighbor mask resize to raw RGB8 images and grayscale masks. After inference, quantize the result and use `reference_output_rgb8` to restore the original dimensions, optionally compositing with the predicted alpha. See [preprocessing](docs/preprocessing.md) for the byte APIs. Use `pad_to_model_size` and `Padding::crop_result` to pad and crop tensors instead of resampling.

### Loading and integrity

`Osor::builder` collects every load option:

```rust
use osor::{Backend, Osor, Result, artifact::ArtifactLimits};

fn load_pinned(bundle: &str, trusted_manifest_sha256: &str) -> Result<Osor> {
    Osor::builder(bundle)
        .backend(Backend::Metal { ordinal: 0 })
        .trusted_manifest_sha256(trusted_manifest_sha256)
        .artifact_limits(ArtifactLimits {
            max_total_tensor_bytes: 24 * 1024 * 1024 * 1024,
            ..ArtifactLimits::default()
        })
        .load()
}
```

Select `Backend::Metal`, `Backend::Cuda`, or `Backend::Wgpu` and enable the corresponding Cargo feature. The loader initializes the selected backend before reading model weights.

A trusted manifest digest pins an imported bundle. Provide the digest retained during conversion or installation. The loader verifies the manifest before opening or allocating weight tensors, then verifies tensor bytes against that manifest. The `manifest()`, `device()`, `dtype()`, and `inference_limits()` getters expose the loaded configuration. See the [integrity contract](docs/artifacts.md).

`ArtifactLimits::max_total_tensor_bytes` sets the aggregate decoded-weight budget; F16 weights loaded as F32 count at the F32 size. The loader checks each file's header against the remaining budget before allocating its weights. Use artifact limits for weight admission and inference limits for job geometry.

### Jobs and cancellation

- `infer` runs one synchronous job.
- `infer_with_observer` reports the 18 intermediate stages as typed `Stage` values in order; `Stage::as_str()` gives the capture name. Return `Error::Cancelled` from the observer to cancel between stages.
- `infer_with_cancellation` shares a `CancellationToken` across threads and returns `Error::Cancelled` at the next checkpoint. Native execution checks between transformer operations, attention slices, and convolution bands.
- `infer_profiled` records synchronized stage timings.
- `preprocess::reference_resize_dimensions` plans the reference CLI resize dimensions. `preprocess::pad_to_model_size` pads image and mask by repeating edge pixels, and `Padding::crop_result` restores the source dimensions.

Host applications manage model installation, memory admission, job scheduling, selections, and undo. See the complete [local inference example](examples/local_inference.rs).

## Model bundles

The library expects this local directory:

```text
bundle/
├── manifest.json
├── unet.safetensors
├── vae.safetensors
└── prompt.safetensors
```

It contains the base model with Phase II rank-256 LoRA adapters merged into its UNet, the trained five-channel output head, the VAE, and cached fixed prompt embeddings.

The offline Rust converter accepts a conversion plan and local safetensors exports:

```sh
scripts/cargo.sh run --locked --bin osor -- convert /path/to/conversion.json
```

It validates critical weights and shapes, records source revisions and SHA-256 hashes, and creates a new bundle. See the [artifact contract](docs/artifacts.md) for the conversion schema, provenance, and model licensing.

Installers holding directory capabilities can use `conversion::convert_with_io`
with already-open sources, trusted input hashes, caller limits, cancellation
checkpoints, and exclusive output sinks. See [conversion through held
capabilities](docs/artifacts.md#conversion-through-held-capabilities).

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

| Backend | Cargo feature | Execution |
|---|---|---|
| CPU | Default | Portable F32 inference |
| CPU + Accelerate | `accelerate` | CPU inference with Apple Accelerate matrix products on macOS |
| Metal | `metal` | F32, F16, and BF16 inference on macOS GPUs |
| CUDA | `cuda` | F32, F16, and BF16 inference on NVIDIA GPUs with a CUDA toolkit and driver |
| WGPU | `wgpu` | F32 GPU graph with CPU input/output staging |
| WGPU + autotune | `wgpu-autotune` | WGPU with runtime kernel tuning for the selected device |

WGPU keeps UNet and VAE activations on the GPU, while noise, mask, scheduler, and alpha operations use CPU staging tensors. Consequently `model.device()` is CPU for WGPU; `compute_device_description()` identifies the actual adapter. See [WGPU details](docs/wgpu.md) and [platform setup](docs/platforms.md).

## Profiling and reference comparison

The port preserves posterior sampling, 8× max-pooled masks, timestep 400, nine-channel UNet input, added text/time conditioning, five-channel output, scheduler predicted-original computation, and latent alpha blending. See the [numerical contract](docs/reference-contract.md).

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

The compare command checks complete stage sets against explicit absolute, relative, and optional RMS tolerances. It exits nonzero for mismatches or nonfinite values. See the [reference test guide](tests/README.md) for model tests.

Measure loading, first inference, warm inference, and stage costs with a retained model using the optimized `profiling` profile:

```sh
scripts/cargo.sh run --locked --profile profiling --bin osor-profile -- \
  /path/to/bundle /path/to/input.safetensors cpu 3
```

</details>

## Development

```sh
scripts/check.sh

# Accelerator checks:
scripts/cargo.sh clippy --locked --all-targets --features metal,wgpu -- -D warnings
scripts/cargo.sh test --locked --all-targets --features metal,wgpu

# Dependency licenses, sources, bans and advisories (deny.toml):
cargo deny check
```

The wrapper uses the pinned Rust toolchain and sccache, with build and temporary files under `/Volumes/Scratch/osor-rs`. Set `OSOR_SCRATCH_ROOT` to your Scratch directory on other hosts. Ordinary tests use small synthetic tensors; model tests use explicitly selected local bundles and reference artifacts.

Keep models, reference source, tensors, images, captures, and reports outside Git.

## License

Original project code is MIT. Adapted Candle code retains its MIT notice; the Pillow byte-resampling adaptation retains its [MIT-CMU notice](src/preprocess/LICENSE-PILLOW). The package records both source licenses. See [attribution](ATTRIBUTION.md), [model provenance and licensing](docs/artifacts.md), and the preserved license notices.
