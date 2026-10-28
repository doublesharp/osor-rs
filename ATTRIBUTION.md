# Attribution

| Component | Source | License / role |
|---|---|---|
| OSOR reference architecture and recipe | [Zhouqm-Git/osor at d5b68e822996d61aae6ba612e5e9b5a14f576879](https://github.com/Zhouqm-Git/osor/tree/d5b68e822996d61aae6ba612e5e9b5a14f576879) | MIT source; preserved notice in LICENSE-OSOR-reference. Referenced for this port; no model weights included. |
| OSOR paper | [arXiv:2606.28094](https://arxiv.org/abs/2606.28094) | Research reference; no paper assets bundled. |
| Candle 0.9.1 layers and adapted SDXL block orchestration | [huggingface/candle at cd96fa80da255e34f7b16b4ff98b6a31d557201b](https://github.com/huggingface/candle/tree/cd96fa80da255e34f7b16b4ff98b6a31d557201b) | MIT or Apache-2.0 dependency. Adapted orchestration uses the MIT option; preserved notice in LICENSE-CANDLE-MIT. |
| Optional Burn/WGPU tensor execution | [Burn 0.22.0](https://docs.rs/crate/burn/0.22.0) | MIT or Apache-2.0 dependency; enabled only by the `wgpu` Cargo feature. Resolved component versions are recorded in Cargo.lock. |

The upstream license table identifies SDXL checkpoints as CreativeML Open RAIL++-M and FLUX checkpoints as FLUX.1-dev Non-Commercial. These are separate from the MIT source-code license. Runtime artifact acceptance and any distribution decision remain separate work.
