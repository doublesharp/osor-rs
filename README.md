<div align="center">
  <img src="docs/assets/osor-logo.png" alt="OSOR logo" width="132">

  # OSOR

  **Object removal, native to Rust.**

  An experimental library for OSOR SDXL-Inpainting Phase II inference.

  ![Rust 1.95.0](https://img.shields.io/badge/Rust-1.95.0-C94B2C?style=flat-square) ![Source licenses MIT and MIT-CMU](https://img.shields.io/badge/license-MIT%20%2B%20MIT--CMU-343434?style=flat-square) ![Optional backends Metal, CUDA and WGPU](https://img.shields.io/badge/GPU-opt--in-343434?style=flat-square)

  [Status](#status) · [Install](#install) · [Use from Rust](#use-from-rust) · [Model bundles](#model-bundles) · [Backends](#backends) · [Development](#development)

  Maintained by 0xdoublesharp
</div>

---

Supply an image and a removal mask to receive the completed image and the model's predicted edit alpha. The edit can include shadows, reflections, and other effects beyond the painted mask. The runtime uses Candle 0.11.0, with an optional Burn 0.22.1 WGPU backend. It loads explicitly installed local safetensors bundles; there is no Python sidecar, libtorch dependency, runtime text encoder, or automatic model download.

The numerical recipe is pinned to [OSOR revision `d5b68e8`](https://github.com/Zhouqm-Git/osor/tree/d5b68e822996d61aae6ba612e5e9b5a14f576879).

## Status

**Experimental · real-model inference works on CPU, Metal, and WGPU on an Apple M2 Max.**

The named FP32 cases pass all 18 reference stages with measured, frozen tolerances. See the [qualification record](docs/qualification.md) for the evidence and its scope.

<details>
<summary>Acceptance scope and remaining platform gates</summary>

CPU, Metal, and WGPU pass all 18 reference stages on an independent 80×96 input against frozen small-case FP32 limits. The real-model missing-weight rejection test also passes.

An independent 640×384 photo passes all 18 stages on the same backends against a separately calibrated, frozen FP32 envelope with elementwise and RMS limits. Larger photos expose greater cross-backend FP32 sensitivity; this envelope is separate from the tighter small-case limits. These are bounded case results, not universal parity or image-quality claims. See the [qualification record](docs/qualification.md), [acceptance guide](tests/README.md), and [remaining work](docs/milestones.md).

Additional photo cases at batches two/four, 88×104 and 512×512 currently exceed those frozen limits. Follow-up work fixes demonstrated CPU and Metal VAE batch corruption, but does not establish broader numerical acceptance. Admission bounds describe executable inputs, not a guarantee of reference agreement.

Linux and Windows CPU cross-checks pass, but native runtime acceptance on those platforms and CUDA device qualification remain open. PhotoCraft integration is a separate project.

</details>

## Install

Use Rust **1.95.0**. The crate is not published to crates.io. Add the private Git repository at a release tag, using credentials with access to it, and keep the resolved commit in your `Cargo.lock`:

```toml
[dependencies.osor-rs]
git = "https://github.com/doublesharp/osor-rs"
tag = "v0.2.0"
# features = ["metal"]  # Enable for macOS Metal.
```

See the [changelog](CHANGELOG.md) for breaking changes between releases.

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

Load an `Osor` once with `Osor::load(bundle_path)` or `Osor::builder(bundle_path)...load()` and retain it across jobs. A loaded model is `Send + Sync`. Pass it and your tensors into synchronous inference:

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

Height and width must be multiples of eight in **32..=1024**. CPU and native GPU admission supports batches up to four, subject to the configured total pixel limit. WGPU supports batch one and at most **262,144 pixels**, including rectangular inputs. Requested `InferenceLimits` outside 1..=4 images and 1..=4,194,304 batch-inclusive pixels are rejected before any bundle I/O; `Osor::inference_limits()` reports the effective limits after WGPU lowers them. These bounds do not guarantee that a job fits available memory.

Supply independent standard-normal noise tensors for ordinary inference. `model.seeded_noise(seed, batch, height, width)` generates both from a `u64` seed with a versioned, platform-independent algorithm (see `osor_rs::noise`), so an application can store a seed instead of the tensors and retry with a different seed. Seeds are not portable to other frameworks: for cross-framework parity, preserve the actual tensor values.

`InferenceResult` contains:

| Field | Shape | Meaning |
|---|---|---|
| `image` | `[B, 3, H, W]` | Full decoded RGB result in `[0, 1]` |
| `alpha` | `[B, 1, H, W]` | Full-resolution predicted edit alpha |
| `latent_alpha` | `[B, 1, H/8, W/8]` | Alpha used for latent blending |

The image already includes alpha blending **in latent space before decoding**. The library returns the full image without RGB paste-back. Restricting it to the original mask can discard intended shadow or reflection removal; applications must choose and validate their compositing policy.

On the current unreleased checkout, `preprocess::resize_reference_inputs` applies the reference CLI's dimension policy, bicubic RGB resize, and nearest-neighbor mask resize to raw RGB8 images and grayscale masks. After inference, quantize the result and use `reference_output_rgb8` to restore the original dimensions, optionally compositing with the predicted alpha. These native Rust byte helpers match the measured Pillow 13.0.0 corpus; see [preprocessing](docs/preprocessing.md) for usage and limits. The tensor API also retains `pad_to_model_size` and `Padding::crop_result` for applications that choose padding instead of resampling.

### Loading and integrity

`Osor::builder` collects every load option:

```rust
use osor_rs::{Backend, Osor, Result, artifact::ArtifactLimits};

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

Select `Backend::Metal`, `Backend::Cuda`, or `Backend::Wgpu` and enable the corresponding Cargo feature. An unavailable requested backend returns an error. The earlier `load_with_options`, `load_with_options_and_manifest_sha256`, and `load_with_options_and_manifest_sha256_and_limits` functions remain and behave the same.

A trusted manifest digest pins an imported bundle. The loader verifies the exact manifest bytes it parses before opening or allocating weight tensors, then verifies tensor bytes against that manifest. Keep the trusted digest from your import or approval step; hashing a replaceable path immediately before loading does not establish a trusted pin. The `manifest()`, `device()`, `dtype()`, and `inference_limits()` getters expose the loaded configuration. See the [integrity contract](docs/artifacts.md).

`ArtifactLimits::max_total_tensor_bytes` caps aggregate decoded-weight bytes; F16 inputs loaded as F32 count at the F32 size. Each file's header is checked against the remaining budget before allocating that file's weights. This weight bound does not cover activation, conversion-buffer, allocator-cache, or other process memory, and is not an RSS bound.

### Jobs and cancellation

- `infer` runs one synchronous job.
- `infer_with_observer` reports the 18 intermediate stages as typed `Stage` values in order; `Stage::as_str()` gives the pinned capture name. Return `Error::Cancelled` from the observer to cancel between stages; an executing neural layer is synchronous.
- `infer_with_cancellation` stops with `Error::Cancelled` after another thread calls `CancellationToken::cancel`. CPU work stops between encoder, UNet, and decoder blocks; on GPU backends, already queued work completes first.
- `infer_profiled` records synchronized stage timings.
- `preprocess::reference_resize_dimensions` plans the upstream CLI resize dimensions. Alternatively, `preprocess::pad_to_model_size` pads image and mask to the model's size rules by repeating edge pixels, and `Padding::crop_result` crops the output back. Either policy changes results near edges; image resampling and crop selection remain application responsibilities.

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

| Backend | Cargo feature | Current boundary |
|---|---|---|
| CPU | Default | F32; portable source; macOS real-model execution |
| CPU + Accelerate | `accelerate` | macOS only; Apple Accelerate matrix products; passes both qualification cases; measure on the intended workload before enabling |
| Metal | `metal` | macOS; M2 Max small-case and independent photo acceptance |
| CUDA | `cuda` | Requires an NVIDIA device, compatible driver, toolkit, and host compiler; device acceptance pending |
| WGPU | `wgpu` | F32; resident GPU graph with CPU staging; M2 Max small-case and independent photo acceptance |
| WGPU + autotune | `wgpu-autotune` | Faster warm WGPU inference after a long first-job tuning pass; not bitwise reproducible across processes |

Native Metal/CUDA APIs also accept F16 and BF16, but reduced precision needs separate numerical qualification. F32 is the current acceptance baseline.

WGPU keeps UNet and VAE activations on the GPU, while noise, mask, scheduler, and alpha operations use CPU staging tensors. Consequently `model.device()` is CPU for WGPU; `compute_device_description()` identifies the actual adapter. See [WGPU details](docs/wgpu.md) and [platform setup](docs/platforms.md).

## Numerical checks and performance

The port preserves posterior sampling, 8× max-pooled masks, timestep 400, nine-channel UNet input, added text/time conditioning, five-channel output, scheduler predicted-original computation, and latent alpha blending. See the [numerical contract](docs/reference-contract.md).

The unreleased follow-up measured **1.57 s warm Metal inference and 10.6 GiB peak process footprint** on the 640×384 photo. A separate 80×96 run measured **1.00 s on portable CPU and 0.67 s with Accelerate**. These are shared-host `profiling` observations with two or three warm samples and loading excluded, not formal benchmarks. Earlier v0.2.0 measurements and upstream comparisons remain in the [performance record](docs/performance.md). WGPU convolution remains a measured bottleneck; no further WGPU speedup is claimed.

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

Measure loading, first inference, warm inference, and stage costs with a retained model. Use the `profiling` profile; dev-profile timings are not representative:

```sh
scripts/cargo.sh run --locked --profile profiling --bin osor-profile -- \
  /path/to/bundle /path/to/input.safetensors cpu 3
```

</details>

## Development

```sh
scripts/check.sh

# Optional accelerator source and ordinary-test gates:
scripts/cargo.sh clippy --locked --all-targets --features metal,wgpu -- -D warnings
scripts/cargo.sh test --locked --all-targets --features metal,wgpu

# Dependency licenses, sources, bans and advisories (deny.toml):
cargo deny check
```

The wrapper pins Rust 1.95.0, uses sccache when installed, and puts builds and temporary files on `/Volumes/Scratch/osor-rs`. It stops if Scratch is missing on this machine. Set `OSOR_SCRATCH_ROOT` explicitly on other hosts. Normal tests use small synthetic tensors and never download model weights. GPU and full-model tests require explicit local artifacts and opt-in execution.

Keep models, reference source, tensors, images, captures, and reports outside Git. Source checks and cross compilation establish different evidence from native device execution and real-model acceptance.

## License

Original project code is MIT. Adapted Candle code retains its MIT notice; the Pillow byte-resampling adaptation retains its [MIT-CMU notice](src/preprocess/LICENSE-PILLOW). The package records both source licenses. Dependencies and upstream sources have their own terms. Model weights have separate licenses, and this repository does not embed or redistribute them. See [attribution](ATTRIBUTION.md), [artifact provenance](docs/artifacts.md), and the preserved license notices.
