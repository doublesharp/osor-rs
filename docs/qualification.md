# Local qualification: 2026-10-28

The complete Phase II model has executed on macOS/aarch64, Apple M2 Max, with CPU, Candle Metal, and Burn WGPU. This qualifies the named FP32 cases below; it does not establish accuracy for every image, reduced precision, or another platform.

## Numerical results

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

## Remaining gates

The additive manifest-pinning API passed its three behavioral regressions, final CPU and accelerator gates, and a real-bundle Metal load followed by all 18 small-case comparisons. A subsequent limits-aware API adds caller-owned decoded-weight admission without changing the original API. Its small F16 fixtures verify target-dtype expansion, cross-file aggregation, exact budget boundaries and rejection before graph validation. The preserved bundle contains 10,605,175,216 decoded F32 bytes (about 9.88 GiB); loading it under a 24 GiB cap and running all 18 Metal reference comparisons passed.

Final CPU gates passed 43 ordinary tests; Metal/WGPU gates passed 45. Strict Clippy, formatting, doctests and CPU source cross-checks passed. Independent Terra source reviews accepted both the manifest pin and the caller-limit follow-up with no P1/P2 findings. These reviews cover the loading boundaries, not the entire neural implementation; the earlier normalization-specific independent review was unavailable due to capacity.

Native Linux and Windows execution, CUDA device acceptance, reduced-precision acceptance, broader image/resolution coverage, and application integration remain separate work. PhotoCraft owns its model import, admission, effect-region compositing, selection preservation, cancellation experience, and undo qualification. See [performance observations](performance.md) for the measured latency and memory limitations.
