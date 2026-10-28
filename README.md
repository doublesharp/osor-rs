# osor-rs

Standalone native Rust port of [OSOR](https://github.com/Zhouqm-Git/osor), starting with SDXL-Inpainting Phase II. PhotoCraft will consume a pinned library revision after numerical and native runtime qualification.

**Status:** repository scaffold and pinned reference contract. Neural inference, checkpoint conversion and the numerical parity runner are not implemented yet. No weights are bundled, downloaded at startup or redistributed here.

The first milestone is matching the released Phase II pipeline with fixed intermediate tensors. CPU correctness comes first, then Metal on this machine and feature-gated CUDA qualification on supported Linux/Windows hosts. A generic SDXL LoRA loader does not cover the additional alpha head and conditioning.

Reference: `d5b68e822996d61aae6ba612e5e9b5a14f576879`. See [the contract](docs/reference-contract.md) and [milestones](docs/milestones.md).

Use `scripts/cargo.sh check` for local development. It routes target and temporary output to Scratch and uses sccache when present. Override `OSOR_SCRATCH_ROOT` on another host. The pinned toolchain is Rust 1.95.0.

The library owns tensors, model loading, inference and an alpha-bearing result. PhotoCraft owns user model choices, installation, crop planning, resource admission, jobs and undo. OSOR's effect region can extend beyond the painted mask, so the PhotoCraft adapter needs a qualified application policy for that result.

Project code is MIT. Upstream code and model weights have separate terms; see [attribution](ATTRIBUTION.md). All model artifacts, reference tensors and measurements stay outside Git on Scratch.
