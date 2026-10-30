# Reference acceptance tests

The ignored `real_model_conversion_io_matches_qualified_bundle` test exercises
the complete handle-based conversion and compares all output bytes by SHA-256
with a pinned qualified bundle. It uses a digest-only sink and writes no weight
files. Set `OSOR_TEST_BUNDLE`, `OSOR_TEST_MANIFEST_SHA256`, and the four local
input paths `OSOR_TEST_BASE_UNET`, `OSOR_TEST_BASE_VAE`,
`OSOR_TEST_CHECKPOINT`, and `OSOR_TEST_PROMPT`. Run it in release mode with
`--ignored --nocapture --test-threads=1`. It uses a 14 GiB decoded input cap;
allow additional transient memory. This is conversion byte-identity evidence,
separate from numerical inference acceptance and application filesystem tests.

Normal tests use small synthetic inputs and do not download weights. The ignored tests in [`real_model.rs`](real_model.rs) opt in to local full-model acceptance. They run inference with captured posterior and diffusion noise, collect all 18 pinned stages, and compare each element with the corresponding reference stage using an explicit absolute and relative tolerance. A seed alone is not sufficient for cross-framework parity.

The pre-release follow-up retains passing acceptance for the two original cases.
Additional B2/B4, odd-size, 512² and native 1024² photo cases have executed but
exceed the unchanged frozen limits; see [qualification](../docs/qualification.md).
The isolated CPU 1024² reference is now available after a guarded solo run.
Keep numerical failure reports even when the test correctly exits nonzero.

Capture `posterior_moments` from the raw VAE output: diffusers exposes it as `latent_dist.parameters`. Its `latent_dist.logvar` is already clamped to [-30,20], so concatenating `mean` and `logvar` does not reconstruct the raw moments. The separate `posterior_logvar` stage and posterior sampling use the clamped values. This distinction matters on actual photos with extreme log variance.

Prepare these local artifacts, keeping tensors, models and reports under Scratch rather than in Git:

- A converted bundle with immutable source revisions and license provenance. The Rust CLI accepts a local conversion plan: `scripts/cargo.sh run --release --bin osor -- convert /path/to/conversion.json`. See the [artifact contract](../docs/artifacts.md) for inputs and strict validation. Export any upstream pickle checkpoint only in an isolated development environment outside this repository.
- An input safetensors file containing `image`, `mask`, `posterior_noise` and `diffusion_noise`. Capture the same explicit noise tensors for the independently executed reference.
- A reference safetensors file containing every name in `pipeline::STAGES`, and a JSON object keyed by those same names, each with finite, nonnegative `absolute` and `relative` tolerances. Derive tolerances from measured comparisons, freeze them for acceptance, and do not widen them to conceal structural mismatches. Comparison requires identical complete stage sets and shapes, rejects nonfinite values, and applies `abs(actual - reference) <= absolute + relative * abs(reference)` to each element.

Each stage may also specify a finite, nonnegative `maximum_root_mean_square_error` in its tolerance JSON. When present, the measured RMS error must be at or below that cap in addition to every element passing the absolute and relative check. Reports include `root_mean_square_error_passed` so a stage can clearly fail the RMS cap even with zero element violations. Omitting the field retains the existing elementwise-only verdict; choose any cap from measured, backend-appropriate comparisons rather than treating it as a default budget.

For example, after setting paths to the prepared files on Scratch, run the CPU acceptance test serially:

```sh
export OSOR_TEST_BUNDLE=/path/to/bundle
export OSOR_TEST_INPUT=/path/to/input.safetensors
export OSOR_TEST_REFERENCE=/path/to/reference.safetensors
export OSOR_TEST_TOLERANCES=/path/to/tolerances.json
export OSOR_TEST_ACTUAL_CAPTURE=/path/to/new-actual.safetensors
scripts/cargo.sh test --release --test real_model real_model_cpu_reference_stages -- --ignored --nocapture --test-threads=1
```

`OSOR_TEST_ACTUAL_CAPTURE` is optional and must name a new file; the test writes the capture before asserting comparison success. The printed report records platform, architecture, backend, dtype, compute device, artifact paths, model hashes, upstream revision and per-stage metrics. Preserve the report and the exact input, reference and tolerance provenance outside Git. For Metal on macOS, CUDA or WGPU, enable the corresponding Cargo feature and select `real_model_metal_reference_stages`, `real_model_cuda_reference_stages` or `real_model_wgpu_reference_stages`, respectively; each requires a working device. The separate ignored `real_model_missing_critical_weights_fail` test uses `OSOR_TEST_BUNDLE` and checks that missing required UNet and VAE weights are rejected; run full-bundle tests with `--test-threads=1`.

Set optional `OSOR_TEST_MANIFEST_SHA256` to a trusted 64-character manifest digest to run reference acceptance through the pinned-manifest loading API. The report records that expected digest. With the variable unset, tests retain the ordinary load path.

With that pin set, optional `OSOR_TEST_MAX_DECODED_BYTES` selects the pinned API that also accepts `ArtifactLimits`. For example, `25769803776` caps aggregate decoded weights at 24 GiB. This covers target-dtype expansion and the combined UNet/VAE/prompt budget, not peak execution memory. The report records the cap; supplying a cap without a trusted manifest pin is rejected.

The local CLI can also capture and compare stages independently:

```sh
scripts/cargo.sh run --release --bin osor -- infer /path/to/bundle /path/to/input.safetensors /path/to/new-output.safetensors cpu /path/to/new-stages.safetensors
scripts/cargo.sh run --release --bin osor -- compare /path/to/reference.safetensors /path/to/new-stages.safetensors /path/to/tolerances.json
```

Inference and capture outputs must be new files. The compare command requires the full pinned stage set and returns nonzero on failure. CPU, Metal and WGPU have passed the independent 80×96 case with frozen small-case limits and a separate 640×384 photo with a full-size envelope frozen before that photo was executed. The full-size envelope uses the measured upstream CPU/MPS spread and explicit RMS caps; it does not replace or erase failures against the earlier tighter limits. See [the qualification record](../docs/qualification.md) for scope, hashes and remaining platform gates. Source checks, conversion, inference and timing alone do not establish numerical acceptance.
