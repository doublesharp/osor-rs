# osor-rs development

This is the standalone native Rust OSOR inference project. Read README.md, docs/reference-contract.md and docs/milestones.md before implementation.

- Rust product/runtime code only. No Python sidecar, libtorch runtime, desktop UI framework or implicit model download.
- Use the pinned Rust toolchain and sccache. On this machine, `scripts/cargo.sh` puts build and temporary output under /Volumes/Scratch/osor-rs; stop if Scratch is missing. Set OSOR_SCRATCH_ROOT explicitly on other hosts.
- Keep models, copied upstream source, raw JSON, tensors, logs, images, benchmarks and captures on Scratch. Never commit evidence, weights or font binaries.
- Never commit one-off agent prompts, implementation briefs or session handoff files. Keep transient instructions outside the repository; maintain durable project guidance in AGENTS.md and docs instead.
- No unsafe, unwrap/expect/panic paths, input-derived unchecked indexing or unbounded input allocations. Return actionable errors; unfinished APIs report unavailable rather than claiming inference.
- Preserve the pinned numerical recipe, including posterior sampling, explicit noise tensors, mask pooling, added text/time conditioning, five-channel output and latent alpha blending. A seed alone is insufficient for cross-framework parity.
- Execute the upstream reference only in an isolated development environment outside this repository. Do not execute pickle input in the Rust runtime. Conversion must fail on missing or mismatched critical tensors.
- Keep checkpoint/license provenance and immutable revisions. Code licensing does not override weight licensing. No downloaded weights are embedded or redistributed by default.
- GPU features remain optional; CPU source must build on macOS, Linux and Windows. Native runtime acceptance requires actual platform evidence.
- Add focused behavioral regressions, then run appropriate tests, formatting and strict Clippy. Real-model numerical comparison and performance claims require the exact artifacts and measured outputs.
- Implementation subagents use the user's requested GPT-6.1 Terra at max effort and Fast/priority service. Report capacity failures; do not silently substitute models. No nested delegation unless explicitly assigned.
