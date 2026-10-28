# Performance observations

The Rust port still trails upstream PyTorch on Apple GPUs. These are engineering measurements on a shared Apple M2 Max with 96 GiB memory, not release benchmarks or a general comparison of programming languages.

## Unreleased follow-up: 2026-10-28

Metal F32 VAE normalization now uses bounded, anchored two-pass reductions
with affine application in a custom kernel. Bias addition and nearest
upsampling also use custom kernels. SiLU stays separate: the fused candidate
failed the frozen small-case comparison and was rejected. CPU F32 UNet
convolutions now use channel-major columns throughout the residual and
sampling blocks; other device and dtype paths retain Candle.

The final combined Metal profiling binary measured the same 640×384 photo,
bundle, and explicit noise as v0.2.0. Loading took 8.115 s; the first inference
took 1.493 s; three warm runs took 1.494, 1.537 and 1.679 s (mean **1.570 s**).
Direct `/usr/bin/time -l` measured **10.596 GiB peak process footprint** over
loading, retained inferences and the final synchronized stage profile. The
earlier worker measured 1.429 s warm. These shared-host observations do not
establish a latency distribution, a memory ceiling, or an isolated before/after
benchmark. The two original frozen cases pass all 18 stages on the combined
CPU, Metal and WGPU build; see [qualification](qualification.md).

In a separate small-case CPU comparison with two warm samples, portable CPU
measured 1.265 s before the UNet change and 1.002 s after it; Accelerate
measured 0.672 s. The photo's worst decoded-image error did not improve
(0.000121355 versus 0.000121266); no broader accuracy improvement is claimed.
`RAYON_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` were four; column filling can
still use up to 16 scoped workers. These are separate shared-host runs, not a
four-thread CPU ceiling or a general recommendation to enable Accelerate.

Current-source WGPU operator timing isolates the next bottleneck:

| Shape | Group norm | SiLU | 3×3 convolution | 2× upsampling |
|---|---:|---:|---:|---:|
| 128×384×640 | 4.84 ms | 5.72 ms | 270.43 ms | 1.04 ms |
| 256×192×320 | 2.64 ms | 2.96 ms | 289.05 ms | 0.66 ms |
| 512×96×160 | 1.50 ms | 1.33 ms | 296.33 ms | 0.45 ms |

Each value averages three synchronized warm dispatches on fixed synthetic
F32 tensors, excluding the first dispatch, readback and model loading. These
measurements identify convolution as the priority; they are not model latency
or evidence for a replacement algorithm. No production WGPU algorithm was
changed. The opt-in benchmark is
`wgpu::model::operator_tests::gpu_vae_operator_timings`.

Raw records remain under `/Volumes/Scratch/osor-rs/reports/codex-integration-20261028`
and `/Volumes/Scratch/osor-rs/reports/codex-finish-20261028`.

## v0.2.0 measurements: Candle 0.11

Measured with `--profile profiling` on Candle 0.11.0, Burn 0.22.1 and the column-convolution VAE path. Same inputs, bundle and captured noise as the earlier sections; F32, batch one, two warm runs after one first run; loading excluded from inference times. The host was shared (load average about 15).

| Implementation | Case | Load | Warm mean | Encode / UNet / decode | Peak footprint |
|---|---|---:|---:|---|---:|
| Upstream PyTorch 2.7.1 / MPS | 640×384 | — | 0.741 s | — | 16.1 GiB |
| Rust / Candle Metal | 640×384 | 6.4 s | **2.76 s** | 0.75 / 0.62 / 1.40 s | 11.3 GiB |
| Rust / Burn WGPU | 640×384 | 9.1 s | 18.2 s | 4.18 / 4.68 / 9.63 s | 10.4 GiB |
| Rust / Candle CPU | 640×384 | 6.4 s | 22.0 s | 4.88 / 5.13 / 8.60 s | 10.0 GiB |
| Upstream PyTorch 2.7.1 / CPU, 4 threads | 80×96 | — | 1.003 s | — | — |
| Rust / Candle CPU, 4 threads | 80×96 | 6.4 s | 1.52 s | 0.16 / 0.98 / 0.36 s | 10.0 GiB |

Stage times come from the separate synchronized profile run and do not sum exactly to the warm mean.

- **Metal, 5.98 s → 3.78 s → 2.76 s.** Candle's Metal convolution builds a pixel-major im2col buffer, runs a GEMM, then copies back to NCHW; at VAE shapes the im2col kernel and copy took most of the time while the GEMM ran at 6–10 TFLOPS. A small Metal kernel now writes channel-major columns so the GEMM output is already NCHW: 2.9–4.4× faster per layer, with bitwise-identical stage outputs (3.78 s). Candle 0.11 then fixed Metal binary ops, which 0.9.1 always ran through its strided kernel (contiguous add 23.6 ms → 2.2 ms on a 126 MB tensor), bringing warm inference to 2.76 s.
- **CPU.** Candle 0.11's tiled CPU convolution reduces over (ky, kx, channel) where 0.9.1 reduced over (channel, ky, kx). Both are F32-accurate against an F64 reference, but the new order drifted further from PyTorch and failed the frozen 80×96 limits. VAE convolutions on CPU therefore use the same channel-major column path, which restores parity (all 18 stages pass) without slowing the 80×96 case (1.69 s → 1.52 s). The UNet's convolutions inside Candle's ResNet blocks still use Candle's tiled CPU convolution.
- **WGPU** is unchanged by the Candle upgrade (18.2 s); its VAE stages are the next WGPU bottleneck. The opt-in `wgpu-autotune` feature measured 16.7 s warm on the photo and 1.05 s at 80×96 (baseline 3.31 s), after first jobs of about 75 s and 50 s spent tuning; see [WGPU](wgpu.md). Burn's `fusion` feature was no faster and failed the 80×96 limits.
- Remaining Metal VAE time is spread across group normalization (built from broadcast ops, about 39 ms per full-resolution call), nearest upsampling (14 ms) and per-channel bias adds (14 ms), alongside the column convolutions. Fused Metal kernels for these are the next measured step.

**Opt-in Apple Accelerate no longer helps.** On Candle 0.11 the `accelerate` build measured 2.13 s warm on the 80×96 input (two samples, 1.94 and 2.33 s; encode 0.14 s, UNet 1.96 s, decode 0.22 s), slower than the portable build's 1.52 s because its UNet stage doubled. On Candle 0.9.1 it had been the faster CPU option (1.16 s). It still passes all 18 stages on both cases; measure before enabling it.

Reproduction data is under `/Volumes/Scratch/osor-rs/reports/dependency-upgrade-20261028` and `/Volumes/Scratch/osor-rs/reports/columns-20261028`.

## Follow-up measurements on Candle 0.9.1

Measured with `--profile profiling` (release optimizations plus line tables) after the decoder, banded-convolution, cached cross-attention and overlapped-hashing changes. Same inputs, bundle and captured noise as below; F32, batch one, two warm runs after one first run; loading excluded from inference times. The host was shared (load average 7–30 during runs).

| Implementation | Case | Load | Warm mean | Peak footprint |
|---|---|---:|---:|---:|
| Upstream PyTorch 2.7.1 / MPS | 640×384 | — | 0.741 s | 16.1 GiB |
| Rust / Candle Metal | 640×384 | 7.1 s | 5.98 s | 13.4 GiB |
| Rust / Burn WGPU | 640×384 | 9.3 s | 18.4 s | 10.5 GiB |
| Upstream PyTorch 2.7.1 / CPU, 4 threads | 80×96 | — | 1.003 s | — |
| Rust / Candle CPU, 4 threads | 80×96 | 7.0 s | 1.69 s | 10.0 GiB |

WGPU warm time was the same with `CUBECL_AUTOTUNE_LEVEL=0` (18.2 s). All 18 stages still pass on every backend and both qualification cases; see [qualification](qualification.md).

What changed, measured against the earlier records below:

- **CPU, 21.7 s → 1.7 s warm.** The earlier figure came from the dev profile, whose dependencies kept debug assertions and overflow checks. A plain release build already measured 3.0 s; the cached prompt keys and values cut the UNet stage further (1.08 s → 0.69 s at 80×96). The dev profile now disables dependency assertions.
- **Metal peak footprint, 30.6 GiB → 13.4 GiB.** Candle lowers each convolution to an im2col buffer that Metal rounds up to a power-of-two allocation; one full-resolution 3×3 decoder convolution needed 2.26 GB (a 4 GiB buffer). VAE convolutions now run in bands of output rows with at most 128 MiB of scratch each. Banding alone measured 15.5 GiB; dropping the cached prompt's `attn2.to_k`/`to_v` weights (about 1.27 GiB F32) accounts for most of the rest.
- **Metal latency, unchanged at that point (5.85 s → 5.98 s).** Metal is GPU-bound: the whole process used about 5.4 s of CPU time. VAE encode (1.78 s) and decode (3.48 s) were convolution-throughput bound; the current section shows the later column-convolution and Candle 0.11 results.
- **WGPU, 20.5 s → 18.4 s.** Attention now slices query rows only above a 256 MiB score budget, which never triggers within WGPU admission, and the prompt is no longer uploaded per job. The UNet stage fell from 5.88 s to 4.61 s.
- **Load, about 9.5 s → 7.1 s (Metal).** SHA-256 of the 10.6 GB bundle costs about 4.5 s of CPU on one core. Tensors of 1 MiB or more are now hashed on a helper thread while the tensor is built from the same bytes.

**Opt-in Apple Accelerate (`--features accelerate`).** On the same 80×96 input with four threads, warm inference averaged 1.16 s (encode 0.18 s, UNet 0.58 s, decode 0.41 s) against 1.69 s for the portable CPU build and 1.00 s for PyTorch CPU. On the 640×384 photo, two warm samples were 49.3 s and 34.0 s under host load average 21; no same-conditions portable CPU photo timing was taken, so no photo speedup is claimed. Accelerate changes CPU accumulation order; it passes all 18 stages on both qualification cases (see [qualification](qualification.md)). The default CPU build is unchanged.

Reproduction data for this section is under `/Volumes/Scratch/osor-rs/reports/review-apply-20261028` (`final-*`, `accelerate-*`, and `kv-cache-*` logs and JSON).

## Earlier development-build measurements

The measurements below predate the follow-up and used the dev profile. They are kept for comparison.

### Same-photo GPU comparison

All runs use the same 640×384 CORNE-Val sample 000001, F32, batch one, identical captured posterior/diffusion noise, and a retained loaded model. Timing synchronizes the selected device and excludes model loading and input/output file I/O. Each process performs one first inference and two warm inferences. Rust additionally measures a separate synchronized stage profile afterward.

| Implementation | First inference | Warm mean | Warm samples |
|---|---:|---:|---|
| Upstream PyTorch 2.7.1 / MPS | 4.131 s | 0.741 s | 0.795, 0.687 s |
| Rust / Candle Metal | 6.012 s | 5.846 s | 5.856, 5.835 s |
| Rust / Burn WGPU | 20.674 s | 20.488 s | 20.644, 20.333 s |

The Rust binary is the current optimized development build: crate `opt-level=1`, dependencies `opt-level=2`, with Metal/WGPU enabled. It is **not a release build**. Rust uses the F32 merged LoRA bundle; PyTorch uses the released unfused LoRA graph. Rust input validation is included; equivalent extra validation was not added to PyTorch. WGPU ran with `CUBECL_AUTOTUNE_LEVEL=0`. These qualifications matter when interpreting the gap.

The GPU photo benchmarks ran serially, but the host was not reserved or idle. Two warm samples do not establish a latency distribution. A release-build comparison under controlled load remains open.

### Where the time goes

The extra synchronized Rust stage profile on that photo measured:

| Stage group | Metal | WGPU |
|---|---:|---:|
| VAE encode / posterior moments | 1.815 s | 4.528 s |
| UNet denoising | 0.706 s | 5.884 s |
| VAE decode | 3.309 s | 10.031 s |

VAE encoding and decoding account for most measured Metal time and much of WGPU time. Optimized VAE convolution/kernel execution is the first target to investigate. Attention changes alone cannot remove the measured VAE convolution costs. WGPU kernel selection and buffer reuse also warrant measured investigation. Every numerical-kernel change needs the fixed stage comparisons again.

Keep a model loaded across jobs. Reloading the full bundle, verifying it, and uploading weights are separate startup costs; avoid paying them for every edit. The new manifest-pinning API preserves byte verification without requiring the application to duplicate the bundle.

### CPU and small inputs

On the identical 80×96 F32 input with four requested CPU threads, the latest two-sample warm mean was 1.003 s for PyTorch CPU and 21.732 s for the portable Rust CPU backend. Rust samples ranged from 17.770 to 25.694 s. An earlier same-input Rust measurement ranged from 8.517 to 10.302 s, illustrating substantial shared-host variability. These are observations, not a stable slowdown estimate.

The same small-input GPU means were 0.355 s for PyTorch MPS, 0.300 s for Rust Metal, and 4.188 s for Rust WGPU. Small-input similarity between Metal and PyTorch did not predict the larger-photo result.

The current portable Candle CPU matrix backend has no Apple Accelerate feature enabled. An optional platform-optimized CPU backend is a concrete optimization candidate; default CPU portability and the existing numerical gates must remain intact. Do not infer a speedup before implementing and measuring it.

### Memory admission

`/usr/bin/time -l` reported peak process footprint over model loading, repeated inferences, and Rust's additional stage-profile run:

| Photo profiling process | Peak footprint |
|---|---:|
| PyTorch MPS | 16.1 GiB |
| Rust Metal | 30.6 GiB |
| Rust WGPU | 11.4 GiB |

The earlier single Metal inference reached about 20.1 GiB. Retained execution and allocator/kernel caches can increase the peak. These figures are not memory ceilings, GPU-only allocations, or promises for another resolution, batch, machine, or application. Applications must include their other allocations in admission and hold that admission through loading, inference, and output drain.

### Quality priorities

The current F32 implementation passes the named stage-acceptance cases in [qualification](qualification.md). Further quality work should expand those cases to difficult edges, shadows, reflections, and more resolutions; preserve reference RGB/mask preprocessing; and qualify application crop context and effect-region compositing. The returned image already includes latent alpha blending.

Changing sampling noise can produce different candidates, with additional inference cost. Reduced precision, quantization, or changes to weights require separate quality/numerical evaluation. Performance optimizations should preserve the pinned recipe and measured acceptance limits.

## Reproduction

Inputs, exact noise, raw timing JSON, stage profiles, process measurements and isolated reference scripts are under `/Volumes/Scratch/osor-rs/reports/resumed-qualification` and `/Volumes/Scratch/osor-rs/tmp/resumed-qualification`.

The photo input SHA-256 is `aaf0d566970b0b5fe189211f4cf05e3e8b4511f45bd388eeb3123432ee63836c`. The small input SHA-256 is `24303b7370116cd71eb57927d81d33ce22568b5ebe7776a3255f9853c5f71631`. The bundle and immutable source revisions are identified in [qualification](qualification.md).
