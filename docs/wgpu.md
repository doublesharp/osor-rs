# WGPU backend

Enable the optional `wgpu` feature and select `Backend::Wgpu { ordinal: 0 }` in
`LoadOptions`. Burn 0.22.0 supplies WGPU tensor execution; its Rust requirement
matches this project's pinned 1.95.0 toolchain. The feature is independent of
the Candle Metal and CUDA features. Normal CPU builds do not activate Burn.
The selected adapter must identify as a GPU; CPU/software and noop adapters are
not accepted as native GPU evidence. Current selection supports ordinal zero.

The implementation uses the same validated local Phase II bundle as the other
backends. It uploads the merged F32 neural weights once, consuming their CPU
handles while constructing the graph, and retains those weights across jobs.
Internal UNet and VAE activations execute on the GPU. The surrounding explicit
noise, mask, scheduler and alpha-blending pipeline uses CPU tensors. Each VAE
encode, UNet forward and VAE decode crosses that boundary once in each direction.
`Osor::device()` therefore returns the CPU input staging device for this backend;
`compute_device_description()` identifies the actual GPU adapter.

Initial admission is F32, batch one, at most 262144 image pixels, axes from 32
through 1024 in multiples of eight. A 512 by 512 image is supported by these
limits, as are rectangular images with the same pixel budget. Attention slices
query rows only when one score matrix would exceed 64 Mi F32 elements
(256 MiB); within the current admission bound every UNet and VAE attention
runs unsliced. Cross-attention keys and values for the bundle's fixed prompt
are computed once at load, so the `attn2.to_k`/`to_v` weights are released and
the prompt is no longer uploaded per job. Applications still own total memory
admission and concurrency. Reuse a loaded `Osor` for repeated jobs.

```rust,ignore
let model = osor_rs::Osor::load_with_options(bundle_path, osor_rs::LoadOptions {
    backend: osor_rs::Backend::Wgpu { ordinal: 0 },
    ..Default::default()
})?;
```

The bridge validates tensor bounds, shapes, dtypes and finite values before
calling Burn. Fallible initialization/readback errors and dependency unwinds are
reported as library errors. An application compiled with `panic=abort` cannot
recover an unwinding dependency failure. The library does not change the
application's panic policy or install a panic hook.

WGPU uses no Python, pickle loader or libtorch runtime. The locked runtime
dependency graph for `--features wgpu` was inspected with `cargo tree -e normal
-i torch-sys`; libtorch was not activated. Optional packages appearing in the
lockfile are not proof that those packages are built or used.

# Verification

Normal tests never require a GPU or download weights. The WGPU transfer and
operator tests are explicitly ignored unless requested on a real device.
Compile, strict Clippy, synthetic tests, transfer/operator execution, complete
model parity and platform acceptance are separate gates. Record actual results
in the project milestones; source presence is not device qualification.

With exact local bundle, input, reference and tolerance paths set as documented
in `tests/README.md`, run the WGPU acceptance test explicitly:

```sh
scripts/cargo.sh test --locked --features wgpu --test real_model \
  real_model_wgpu_reference_stages -- --ignored --exact --nocapture --test-threads=1
```

Use `osor-profile ... wgpu` for cold/warm timing and `osor infer ... wgpu` for
stage captures. Keep all captures and reports on Scratch. The native backend
comparison must name the actual adapter, precision, artifact hashes, explicit
noise and stage tolerances. Windows/Linux builds and device behavior need their
own evidence even after qualification on macOS.
