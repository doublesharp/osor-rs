# Platforms and build checks

CPU is the default build on macOS, Linux, and Windows. Rust 1.95.0 is pinned in
`rust-toolchain.toml`. CI is configured to run formatting, strict Clippy on all targets, all
ordinary targets (including compiling the example), and doctests on each host.
CI also compiles the Metal and Accelerate features on macOS, compiles the CUDA
feature in an NVIDIA CUDA development container without a GPU, and runs
`cargo deny check` against `deny.toml` (allowed code licenses, crates.io-only
sources, banned libtorch/Python/model-download crates, and triaged advisories).
CI downloads Rust dependencies but no model weights. A successful CI run is
source/build and test evidence for that runner; it is not evidence of real-model
numerical parity, GPU inference, performance, or acceptance on a user's machine.
Actual host and model qualification must be recorded separately.

On this machine, use the default `/Volumes/Scratch/osor-rs`; the wrapper refuses
to proceed if `/Volumes/Scratch` is missing. On other hosts, set
`OSOR_SCRATCH_ROOT` to a writable location outside the checkout before using the
wrapper. It creates `cargo/native-sdxl` and `tmp/native-sdxl` there, sets
`CARGO_TARGET_DIR` and `TMPDIR`, and uses sccache when it is available on PATH.
Keep models, inputs, outputs, and captures outside the repository as well.
The GitHub Actions jobs set this root under `runner.temp`; Windows runners also
provide their system temporary directory outside the checkout. The scripts are
POSIX shell scripts; use a shell such as Git Bash on Windows.

```sh
# Replace this example with an appropriate writable path on the host.
export OSOR_SCRATCH_ROOT="$HOME/osor-scratch"
./scripts/check.sh
```

The check script runs `fmt --all -- --check`, `clippy --locked --all-targets --
-D warnings`, `test --locked --all-targets`, `test --locked --doc`, and
`doc --locked --no-deps` with rustdoc warnings denied through
`scripts/cargo.sh`. Every public item must be documented (`missing_docs`). A locked build requires the checked-in lockfile to match the
manifest. Install `sccache` separately for local compiler caching; its cache
location and service are host configuration, not model data.

## Local inference example

With a locally prepared, validated bundle and a safetensors input capture:

```sh
./scripts/cargo.sh run --locked --example local_inference -- \
  /path/to/bundle /path/to/input.safetensors /path/to/new-output.safetensors
```

The default CPU path uses F32. The input capture contains `image` with shape
`[B,3,H,W]`, `mask` with shape `[B,1,H,W]`, and `posterior_noise` and
`diffusion_noise`, each `[B,4,H/8,W/8]`. Image and mask values are in [0,1];
white means removal. Height and width must be multiples of eight in 32..=1024.
Image and mask tensors use F32; noise tensors use the model dtype. All tensors
must be on the model device. The output capture
contains the full decoded `image` and full-resolution predicted `alpha`.
Effects can extend beyond the mask. The output path must not already exist.
See [the reference contract](reference-contract.md) for the numerical recipe
and [artifact documentation](artifacts.md) for local bundle preparation and
provenance. No model is fetched implicitly.

## Build profiles

Use the `profiling` profile (release optimizations with line tables) for any
latency or memory measurement:

```sh
./scripts/cargo.sh run --locked --profile profiling --bin osor-profile -- \
  /path/to/bundle /path/to/input.safetensors cpu 3
```

The dev profile keeps the crate at `opt-level = 1` for fast rebuilds and builds
dependencies at `opt-level = 2` without debug assertions or overflow checks.
Before that change, dependency debug assertions made Candle CPU inference about
seven times slower than release on the 80×96 qualification input.

## Optional accelerators

The `accelerate` feature (macOS only) routes Candle CPU matrix products through
Apple Accelerate. It changes CPU accumulation order, so its acceptance is
recorded separately in the [qualification record](qualification.md).

```sh
./scripts/cargo.sh check --locked --all-targets --features accelerate
```


On macOS with the Apple development tools installed, this checks that the Metal
feature compiles; CI also performs this check on a macOS runner:

```sh
./scripts/cargo.sh check --locked --all-targets --features metal
```

A Metal build does not establish that a device is available or that inference,
precision, memory use, and latency are qualified. Select the Metal backend
explicitly with `LoadOptions` in an application, and test it on the intended
hardware with the exact local artifacts.

CUDA is a separate opt-in build for a supported Linux or Windows host with a
compatible NVIDIA GPU, driver, CUDA toolkit (`nvcc` and development libraries),
and a supported host compiler. Toolkit, driver, compiler, and GPU compatibility
must be checked for the selected Candle/CUDA dependency versions. For example,
with the toolkit configured on PATH and its normal library search paths:

```sh
./scripts/cargo.sh check --locked --all-targets --features cuda
./scripts/cargo.sh test --locked --all-targets --features cuda
./scripts/cargo.sh test --locked --doc --features cuda
```

These commands require the host toolchain and do not by themselves qualify
real-model CUDA inference. The CPU CI does not pretend to qualify CUDA. Select
`Backend::Cuda` explicitly and collect real-device parity and performance
measurements separately.

## Cross compilation

Install the target's Rust standard library with `rustup target add <triple>` and
provide a compatible target linker, sysroot, and native libraries for the chosen
host/target pair. Set the corresponding Cargo target linker configuration, then
for example run `./scripts/cargo.sh check --locked --all-targets --target
<triple>`. GPU cross builds additionally need a compatible target GPU SDK and
native toolchain. A cross check does not run target binaries; execute tests and
real-model acceptance on the actual target platform before making runtime claims.
