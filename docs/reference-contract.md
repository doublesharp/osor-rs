# SDXL Phase II reference contract

Reference source: [Zhouqm-Git/osor at d5b68e822996d61aae6ba612e5e9b5a14f576879](https://github.com/Zhouqm-Git/osor/tree/d5b68e822996d61aae6ba612e5e9b5a14f576879). The detailed recipe is in `osor-sdxlinpainting/scripts/inference_enhance.py`, `configs/phase2.yaml` and the generator/prompt modules they import. This is a port specification, not evidence of native inference.

- Normalize encoded RGB input from [0,1] to [-1,1]. Encode with the base VAE and **sample** its latent posterior, then apply the VAE scaling factor.
- Max-pool the white-is-removal mask with kernel and stride 8; preserve the reference resize/aspect policy separately from the tensor API.
- Supply explicit diffusion noise and use the base DDPM scheduler at timestep 400. Form the UNet input by concatenating noised original latents (4), pooled mask (1), and original latents (4): nine channels.
- Restore rank-256 attention, convolution and feed-forward LoRA targets from the Phase II recipe. Restore the trained five-channel output head and validate every required tensor.
- Feed the cached fixed sequence embeddings, pooled embeddings and SDXL time IDs `[height,width,0,0,height,width]` through the same added conditioning as the reference.
- Split the output into four denoising channels and one alpha logit. Apply sigmoid to alpha. Read the scheduler's predicted original sample using its pinned configuration, including clipping/thresholding policy, then blend denoised and original latents with alpha **before** VAE decode.
- Divide by the VAE scaling factor, decode, map to [0,1] and clamp as in the reference. Return the full decoded image and the predicted alpha. The reference's optional RGB paste-back is disabled by default.

For reproducible parity, capture both posterior-sampling noise and diffusion noise as tensors, alongside intermediate shapes, dtypes and scheduler configuration. Identical RNG seeds do not guarantee identical random tensors across PyTorch and a Rust backend.

Required reference stages: VAE posterior statistics/sample; scaled original latent; pooled mask; noisy latent and nine-channel input; prompt/pooled/time conditioning; UNet denoising output and alpha logits; scheduler predicted original sample; alpha-blended latent; decoded image. Set stage-specific tolerances from actual fp32/bfloat16 comparison, never widen tolerances to hide structural mismatches.

The standalone result must preserve effect-aware semantics beyond the input mask. PhotoCraft's existing LaMa/MI-GAN bridge writes within its hole; the future OSOR adapter must separately qualify how a full-image result and predicted edit region are applied with selections, crops, damage tracking and undo.
