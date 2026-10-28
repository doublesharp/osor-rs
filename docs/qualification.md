# Local qualification: 2026-10-28

The complete Phase II model has executed on macOS/aarch64, Apple M2 Max, with CPU, Candle Metal, and Burn WGPU. This qualifies the named FP32 cases below; it does not establish accuracy for every image, reduced precision, or another platform.

## Unreleased follow-up

This section records the follow-up committed as `b370d0d` and its loader
resource-error correction. Subsequent attention-memory, cancellation,
validation-transfer and WGPU-preflight changes require their own verification.

These inference algorithms pass all 18 stages on both original cases for
CPU, CPU with Accelerate, Metal and WGPU. The trusted manifest and both frozen
tolerance files retain the hashes recorded below. The photo decoded maximum /
RMS errors are 0.000121355 / 0.000002554 on CPU, 0.000161111 / 0.000002843
with Accelerate, 0.000294715 / 0.000002955 on Metal, and
0.000290096 / 0.000004027 on WGPU. These results do not establish a general
accuracy improvement: the CPU photo maximum did not improve.

Broader checks uncovered and repaired distinct-image VAE batch corruption.
CPU attention now materializes the transposed input before projection; Metal
VAE attention evaluates each image separately. Deterministic B2/B4 operator
regressions failed before the repairs and now match separate calls exactly.
The current CPU real-model posterior for the controlled second B2 entry also
matches its separate call bit for bit. Remaining errors for the first B2 entry
also occur when that entry runs alone.

The following additional photo cases **do not pass** the existing frozen
limits. Counts are passing stages out of 18; the small envelope was selected
for the 80×96 and 88×104 cases, and the full-photo envelope for 512² and 1024²,
before reference capture. No limits were widened or swapped after failure.

| Additional case | CPU | Metal | WGPU |
|---|---:|---:|---:|
| B2, 80×96 | 6 | 14 | Outside admission |
| B4, 80×96 | 7 | 15 | Outside admission |
| B1, 88×104, latent 11×13 | 7 | 16 | 5 |
| B1, 512×512 | 17 | 16 | 16 |
| B1, 1024×1024 | 16 | 16 | Outside admission |

At 1024², CPU and Metal exceed the denoising-output and alpha-logit limits;
the other 16 stages pass. The independent CPU reference completed in the
isolated environment at about 29.4 GiB sampled peak RSS, with a 45 GiB guard
and a minimum 40% free-memory guard. The earlier 29 GiB guarded stop remains
preserved. This is shape coverage, not independent image-quality acceptance:
these inputs resize and sometimes flip the preserved CORNE-Val photos.
Broader numerical acceptance remains open, including the residual odd-size
and alpha errors; admission does not promise agreement with either envelope.

For this snapshot, CPU and Accelerate ordinary suites pass 96 tests each; combined
Metal/WGPU ordinary tests pass 98. Formatting, strict Clippy, doctests and
warning-free rustdoc pass. Bitwise convolution/cache fixtures use exact
dyadic coefficients on both portable CPU and Accelerate, with separate F64
numerical checks. The 776-case Pillow oracle compares 5,912,751 output bytes
with zero mismatches. See [preprocessing](preprocessing.md) for its scope.
The reviewed convolution thread-creation panic path and the analogous bundle
hashing path now use fallible spawning with injected-failure regressions.

Unsuppressed feature checks, actual Metal operator/batch checks, WGPU operator
and lifecycle checks, and Linux/Windows CPU cross-builds are recorded under
`/Volumes/Scratch/osor-rs/reports/codex-integration-20261028`. Broader inputs,
references, failures and source fingerprints remain under
`/Volumes/Scratch/osor-rs/reports/codex-finish-20261028`. Model loading verifies
the original pinned bytes. Later changes to exactness fixtures, operational
error classification and package license metadata do not change the measured
inference algorithms. Native
Linux/Windows full-model execution, CUDA device execution and reduced-precision
acceptance remain separate gates.

## v0.2.0 numerical results

- The independent 80×96 random input passes all 18 stages against the existing frozen small-case limits on CPU, Metal, and WGPU. The full-bundle missing-critical-weight test passes.
- A 704×368 CORNE-Val photo exposed a long-reduction precision problem. The measured WGPU normalization fixture improved from maximum error 0.004425 to 0.000005722 without changing its `1e-5` limit. A normal CPU regression also checks long groups against an independent F64 closed-form result.
- The larger photo still fails the original small-case limits. Those files and failures are preserved. Upstream PyTorch 2.7.1 CPU versus MPS itself differs by up to 0.011065 in posterior moments on that image.
- A separate full-size envelope was frozen from that upstream cross-backend comparison before downloading or executing CORNE-Val sample 000001. Each non-exact stage uses the larger of the old absolute budget and three times the measured reference maximum error; its RMS cap is the larger of the old absolute budget divided by 32 and three times reference RMS error. Image and alpha absolute budgets are also capped at `1/255`. Relative limits are unchanged; sequence, pooled embeddings, time IDs, and pooled mask require exact equality.
- The independent 640×384 sample 000001 passes **all 18 stages on all three backends**, including both elementwise and RMS limits, without changing that frozen envelope.

On the independent photo, decoded-image maximum absolute error / RMS error is:

| Backend | Maximum | RMS |
|---|---:|---:|
| CPU | 0.00007898 | 0.000002250 |
| Metal | 0.00026736 | 0.000002627 |
| WGPU | 0.00029010 | 0.000004027 |

These are numerical differences from the pinned upstream output, not perceptual quality scores or comparisons with an object-free ground truth.

The independent photo also exposed a reference-capture mismatch: `posterior_moments` must contain raw encoder parameters, while `posterior_logvar` is clamped. Correcting the reference changed only `posterior_moments`; the other 17 tensors and the frozen tolerance bytes remained unchanged. See [the exact stage contract](reference-contract.md).

## Source and device checks

The CPU formatting/strict-Clippy/all-targets/doctest gate passes. Metal/WGPU strict Clippy and ordinary tests pass. Nine explicit WGPU device/operator tests pass, including large normalization, transfer, device recovery, and coexistence with a host application's WGPU initialization. Default CPU source checks pass for the installed Linux and Windows cross targets. Cross checks do not execute target binaries.

Real-model comparison, synthetic tests, source checks, and device execution establish separate results. The additional manifest-pinning API has its own behavioral and loading checks; it preserves the inference graph.

## Reproduction artifacts

PhotoCraft's consumer-owned ML-service smoke also passed against the preserved bundle: a 101×47 source used a 104×48 model context and restored RGB/alpha dimensions, cold/warm output differences were zero, and holding the result reservation forced Busy. Its log is `/Volumes/Scratch/photocraft/reports/osor-ml-20261028/real-service-smoke.log`. This is a synthetic-crop bridge result, without an independent reference or visual-quality claim. Native app preview/apply/undo remains separate.

All models, inputs, explicit noises, reference captures, comparison reports, measurements, and development environments remain outside Git on Scratch:

- Bundle: `/Volumes/Scratch/osor-rs/data/model-qualification/bundle-f32-v1`.
- Bundle manifest SHA-256: `75324653945a3c74b69fdfd2bde8b852e858bae461a3d9992ce48168c6987a0f`.
- Current run directory: `/Volumes/Scratch/osor-rs/reports/resumed-qualification`.
- Frozen full-size envelope: `full-photo-f32-tolerances-v1.json`, SHA-256 `084de09e4f5c95cbbc5a9d84d9410173b0454cc8f7a5ea3375c11ac614fd796f`.
- Independent input: `input-holdout-000001-640x384.safetensors`, SHA-256 `aaf0d566970b0b5fe189211f4cf05e3e8b4511f45bd388eeb3123432ee63836c`.
- Corrected reference: `reference-holdout-000001-cpu-raw.safetensors`.
- Passing comparisons: `{cpu,metal,wgpu}-holdout-000001-raw-v2-comparison.json`.

The bundle manifest identifies the exact source revisions, licenses, conversion-input hashes, and model-file hashes. The independent dataset is CORNE-Val revision `6d1c6c9bd0822e16bb5293b3d9ef2f6269222baf`; source image/mask hashes and preprocessing are recorded alongside the input. No weights or dataset images are distributed in this repository.

## Dependency upgrade: Candle 0.11

The library now depends on Candle 0.11.0, safetensors 0.8.0, sha2 0.11.1, Burn 0.22.1 and pollster 1.0.1, with every transitive dependency at its latest Rust 1.95-compatible release (`generic-array` stays at 0.14.7 because 0.14.9 requires a newer compiler). All 18 stages were rerun against the same frozen files with the `profiling` build:

| Backend | 80×96, `wgpu-f32-tolerances-v1` | 640×384, `full-photo-f32-tolerances-v1` | Photo decoded max / RMS |
|---|---|---|---|
| CPU | 18/18 pass | 18/18 pass | 0.00012127 / 0.000002556 |
| Metal | 18/18 pass | 18/18 pass | 0.00026733 / 0.000002626 |
| WGPU | 18/18 pass | 18/18 pass | 0.00029010 / 0.000004027 |
| CPU + `accelerate` | 18/18 pass | 18/18 pass | 0.00016105 / 0.000002851 |
| WGPU + `wgpu-autotune` | 18/18 pass | 18/18 pass | 0.00038100 / 0.000004980 |

The full-bundle missing-critical-weight test passes. On the first CPU run, Candle 0.11's tiled CPU convolution failed the 80×96 limits (decoded image maximum 0.0021 against 0.001; alpha logits 1.09e-4 against 1e-4) although its F64-referenced error was ordinary F32 rounding: its reduction order differs from PyTorch's. VAE convolutions on CPU now use the channel-major column path, which restored all 18 stages; the 80×96 CPU decoded maximum is 0.00054526, below the 0.00082076 measured on Candle 0.9.1. Tolerances were not changed. The CPU photo maximum rose from 0.00007838 to 0.00012127 because the UNet's convolutions still use Candle's tiled CPU convolution. Logs are under `/Volumes/Scratch/osor-rs/reports/dependency-upgrade-20261028`.

## Review follow-up: 2026-10-28

The follow-up replaced Candle's `AutoEncoderKL` decoder with an owned decoder using the bounded group normalization already used by the encoder, ran VAE convolutions in bands of output rows, computed the fixed prompt's cross-attention keys and values once at load on both native and WGPU graphs, removed WGPU's fixed 64-query attention slicing below a 256 MiB score budget, and overlapped bundle hashing with tensor construction. Every change was rerun against the frozen files above with the `profiling` build:

| Backend | 80×96, `wgpu-f32-tolerances-v1` | 640×384, `full-photo-f32-tolerances-v1` | Photo decoded max / RMS |
|---|---|---|---|
| CPU | 18/18 pass | 18/18 pass | 0.00007838 / 0.000002050 |
| Metal | 18/18 pass | 18/18 pass | 0.00026736 / 0.000002630 |
| WGPU | 18/18 pass | 18/18 pass | 0.00029010 / 0.000004027 |
| CPU + `accelerate` | 18/18 pass | 18/18 pass | 0.00017026 / 0.000002482 |

The full-bundle missing-critical-weight test also passes. CPU photo error improved slightly (previously 0.00007898 / 0.000002250); Metal and WGPU photo maxima are unchanged. Focused regressions check that banded convolution is bitwise equal to direct convolution, that cached and self-attention are bitwise equal to Candle's `CrossAttention` for batch one and two, that the owned decoder matches `AutoEncoderKL` within 1e-6 where groups reduce in one pass, and that large-tensor hashing preserves digest order and rejects changed bytes. Nine explicit WGPU device/operator tests pass on the M2 Max, including a new cached cross-attention comparison. Logs are under `/Volumes/Scratch/osor-rs/reports/review-apply-20261028`.

## Remaining gates

The additive handle-based `conversion::convert_with_io` API reproduced all
three preserved bundle file hashes and the exact serialized manifest from the
four original inputs on macOS. Its digest-only sink wrote no weight files.
This checks full input verification, adapter merging, graph validation, output
encoding and manifest publication ordering without rerunning inference. The
mirror's different F32 prompt requires its own numerical qualification.
Cancellation, changed input bytes, decoded-byte limits and sink failures have
focused synthetic regressions. Independent Terra review accepted the
conversion boundary after a manifest-size preflight correction. Raw receipts
remain under `/Volumes/Scratch/osor-rs/reports/capability-conversion-20261028`.

The additive manifest-pinning API passed its three behavioral regressions, final CPU and accelerator gates, and a real-bundle Metal load followed by all 18 small-case comparisons. A subsequent limits-aware API adds caller-owned decoded-weight admission without changing the original API. Its small F16 fixtures verify target-dtype expansion, cross-file aggregation, exact budget boundaries and rejection before graph validation. The preserved bundle contains 10,605,175,216 decoded F32 bytes (about 9.88 GiB); loading it under a 24 GiB cap and running all 18 Metal reference comparisons passed.

Final CPU gates passed 43 ordinary tests; Metal/WGPU gates passed 45. Strict Clippy, formatting, doctests and CPU source cross-checks passed. Independent Terra source reviews accepted both the manifest pin and the caller-limit follow-up with no P1/P2 findings. These reviews cover the loading boundaries, not the entire neural implementation; the earlier normalization-specific independent review was unavailable due to capacity.

Native Linux and Windows execution, CUDA device acceptance, reduced-precision acceptance, broader image/resolution coverage, and application integration remain separate work. PhotoCraft owns its model import, admission, effect-region compositing, selection preservation, cancellation experience, and undo qualification. See [performance observations](performance.md) for the measured latency and memory limitations.
