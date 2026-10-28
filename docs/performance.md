# Performance observations: 2026-10-28

The Rust port currently trails the upstream PyTorch implementation on a real photo. These are preliminary engineering measurements on a shared Apple M2 Max with 96 GiB memory, not release benchmarks or a general comparison of programming languages.

## Same-photo GPU comparison

All runs use the same 640×384 CORNE-Val sample 000001, F32, batch one, identical captured posterior/diffusion noise, and a retained loaded model. Timing synchronizes the selected device and excludes model loading and input/output file I/O. Each process performs one first inference and two warm inferences. Rust additionally measures a separate synchronized stage profile afterward.

| Implementation | First inference | Warm mean | Warm samples |
|---|---:|---:|---|
| Upstream PyTorch 2.7.1 / MPS | 4.131 s | 0.741 s | 0.795, 0.687 s |
| Rust / Candle Metal | 6.012 s | 5.846 s | 5.856, 5.835 s |
| Rust / Burn WGPU | 20.674 s | 20.488 s | 20.644, 20.333 s |

The Rust binary is the current optimized development build: crate `opt-level=1`, dependencies `opt-level=2`, with Metal/WGPU enabled. It is **not a release build**. Rust uses the F32 merged LoRA bundle; PyTorch uses the released unfused LoRA graph. Rust input validation is included; equivalent extra validation was not added to PyTorch. WGPU ran with `CUBECL_AUTOTUNE_LEVEL=0`. These qualifications matter when interpreting the gap.

The GPU photo benchmarks ran serially, but the host was not reserved or idle. Two warm samples do not establish a latency distribution. A release-build comparison under controlled load remains open.

## Where the time goes

The extra synchronized Rust stage profile on that photo measured:

| Stage group | Metal | WGPU |
|---|---:|---:|
| VAE encode / posterior moments | 1.815 s | 4.528 s |
| UNet denoising | 0.706 s | 5.884 s |
| VAE decode | 3.309 s | 10.031 s |

VAE encoding and decoding account for most measured Metal time and much of WGPU time. Optimized VAE convolution/kernel execution is the first target to investigate. Attention changes alone cannot remove the measured VAE convolution costs. WGPU kernel selection and buffer reuse also warrant measured investigation. Every numerical-kernel change needs the fixed stage comparisons again.

Keep a model loaded across jobs. Reloading the full bundle, verifying it, and uploading weights are separate startup costs; avoid paying them for every edit. The new manifest-pinning API preserves byte verification without requiring the application to duplicate the bundle.

## CPU and small inputs

On the identical 80×96 F32 input with four requested CPU threads, the latest two-sample warm mean was 1.003 s for PyTorch CPU and 21.732 s for the portable Rust CPU backend. Rust samples ranged from 17.770 to 25.694 s. An earlier same-input Rust measurement ranged from 8.517 to 10.302 s, illustrating substantial shared-host variability. These are observations, not a stable slowdown estimate.

The same small-input GPU means were 0.355 s for PyTorch MPS, 0.300 s for Rust Metal, and 4.188 s for Rust WGPU. Small-input similarity between Metal and PyTorch did not predict the larger-photo result.

The current portable Candle CPU matrix backend has no Apple Accelerate feature enabled. An optional platform-optimized CPU backend is a concrete optimization candidate; default CPU portability and the existing numerical gates must remain intact. Do not infer a speedup before implementing and measuring it.

## Memory admission

`/usr/bin/time -l` reported peak process footprint over model loading, repeated inferences, and Rust's additional stage-profile run:

| Photo profiling process | Peak footprint |
|---|---:|
| PyTorch MPS | 16.1 GiB |
| Rust Metal | 30.6 GiB |
| Rust WGPU | 11.4 GiB |

The earlier single Metal inference reached about 20.1 GiB. Retained execution and allocator/kernel caches can increase the peak. These figures are not memory ceilings, GPU-only allocations, or promises for another resolution, batch, machine, or application. Applications must include their other allocations in admission and hold that admission through loading, inference, and output drain.

## Quality priorities

The current F32 implementation passes the named stage-acceptance cases in [qualification](qualification.md). Further quality work should expand those cases to difficult edges, shadows, reflections, and more resolutions; preserve reference RGB/mask preprocessing; and qualify application crop context and effect-region compositing. The returned image already includes latent alpha blending.

Changing sampling noise can produce different candidates, with additional inference cost. Reduced precision, quantization, or changes to weights require separate quality/numerical evaluation. Performance optimizations should preserve the pinned recipe and measured acceptance limits.

## Reproduction

Inputs, exact noise, raw timing JSON, stage profiles, process measurements and isolated reference scripts are under `/Volumes/Scratch/osor-rs/reports/resumed-qualification` and `/Volumes/Scratch/osor-rs/tmp/resumed-qualification`.

The photo input SHA-256 is `aaf0d566970b0b5fe189211f4cf05e3e8b4511f45bd388eeb3123432ee63836c`. The small input SHA-256 is `24303b7370116cd71eb57927d81d33ce22568b5ebe7776a3255f9853c5f71631`. The bundle and immutable source revisions are identified in [qualification](qualification.md).
