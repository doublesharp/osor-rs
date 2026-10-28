# Native port milestones

1. Pin reference and model configuration; define safe tensor/manifest schemas, limits and reproducible golden capture with explicit random tensors. Scaffold is started; the runner and reference tensors are pending.
2. Build safe offline checkpoint conversion, including all rank-256 adapters and the five-channel head. Use fixed prompt embeddings so runtime text encoders are unnecessary. No pickle execution in the Rust runtime.
3. Implement the SDXL UNet added conditioning, alpha output and scheduler/VAE path in Rust. Candidate building blocks are Candle layers; inspect and pin the actual backend APIs before adding dependencies. Match every reference stage on CPU before claiming native inference.
4. Qualify Metal precision, memory admission and warm/cold performance on this machine. Add optional CUDA and validate Linux/Windows CPU/runtime behavior separately. No platform performance claim comes from source checks alone.
5. Integrate a pinned crate into PhotoCraft behind an explicit model/process choice. Qualify effect-region application, source/crop/selection preservation, cancellation and one-step undo before adding the model to automatic selection.

Repository creation is independent of these inference gates. No weights or large SDKs are required for the scaffold.
