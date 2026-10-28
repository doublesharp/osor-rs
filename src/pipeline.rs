//! Explicit-noise OSOR inference, independent of a particular model backend.
use crate::{Error, Result, scheduler::Scheduler};
use candle_core::{DType, Tensor};

/// Model operations used by the deterministic pipeline.
pub trait ModelBackend {
    fn encode_moments(&self, image: &Tensor) -> Result<Tensor>;
    fn denoise(
        &self,
        input: &Tensor,
        timestep: usize,
        sequence: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
    ) -> Result<Tensor>;
    fn decode(&self, latents: &Tensor) -> Result<Tensor>;
}

/// Limits apply before model execution and include the batch in pixel counts.
#[derive(Debug, Clone, Copy)]
pub struct InferenceLimits {
    pub max_batch: usize,
    pub max_pixels: usize,
}
impl Default for InferenceLimits {
    fn default() -> Self {
        Self {
            max_batch: 4,
            max_pixels: 4 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub timestep: usize,
    pub vae_scaling_factor: f64,
    pub limits: InferenceLimits,
}
impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            timestep: 400,
            vae_scaling_factor: 0.13025,
            limits: InferenceLimits::default(),
        }
    }
}

pub struct InferenceInput<'a> {
    pub image: &'a Tensor,
    pub mask: &'a Tensor,
    pub posterior_noise: &'a Tensor,
    pub diffusion_noise: &'a Tensor,
}
pub struct InferenceResult {
    pub image: Tensor,
    pub alpha: Tensor,
    pub latent_alpha: Tensor,
}

/// Observer receives these stages in order, synchronously, without retaining tensors.
pub const STAGES: &[&str] = &[
    "posterior_moments",
    "posterior_mean",
    "posterior_logvar",
    "posterior_sample",
    "original_latent",
    "pooled_mask",
    "noisy_latent",
    "unet_input",
    "sequence",
    "pooled",
    "time_ids",
    "denoising_output",
    "alpha_logits",
    "latent_alpha",
    "predicted_original",
    "blended_latent",
    "decoded_image",
    "alpha",
];

pub struct Pipeline<B> {
    backend: B,
    scheduler: Scheduler,
    sequence: Tensor,
    pooled: Tensor,
    config: PipelineConfig,
}

fn validate_tensor(
    t: &Tensor,
    name: &str,
    shape: &[usize],
    dtype: DType,
    device: &candle_core::Device,
    range: Option<(f32, f32)>,
) -> Result<()> {
    if t.dims() != shape || t.dtype() != dtype || !t.device().same_device(device) {
        return Err(Error::InvalidInput(format!(
            "{name} must have shape {shape:?} and match dtype and device"
        )));
    }
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
        return Err(Error::InvalidInput(format!(
            "{name} must have a floating dtype"
        )));
    }
    // Tensor sizes are bounded by the caller before this conversion.
    let values = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    if values
        .iter()
        .any(|v| !v.is_finite() || range.is_some_and(|(lo, hi)| *v < lo || *v > hi))
    {
        return Err(Error::InvalidInput(format!(
            "{name} contains non-finite or out-of-range values"
        )));
    }
    Ok(())
}

impl<B: ModelBackend> Pipeline<B> {
    pub fn new(
        backend: B,
        scheduler: Scheduler,
        sequence: Tensor,
        pooled: Tensor,
        config: PipelineConfig,
    ) -> Result<Self> {
        if config.limits.max_batch > 4 || config.limits.max_pixels > 4 * 1024 * 1024 {
            return Err(Error::InvalidInput(
                "pipeline limits exceed hard safety bounds".into(),
            ));
        }
        if !config.vae_scaling_factor.is_finite()
            || config.vae_scaling_factor <= 0.0
            || config.timestep >= scheduler.config().num_train_timesteps
            || config.limits.max_batch == 0
            || config.limits.max_pixels == 0
        {
            return Err(Error::InvalidInput(
                "invalid pipeline timestep, scaling factor, or limits".into(),
            ));
        }
        let dtype = sequence.dtype();
        let device = sequence.device();
        validate_tensor(&sequence, "sequence", &[1, 77, 2048], dtype, device, None)?;
        validate_tensor(&pooled, "pooled", &[1, 1280], dtype, device, None)?;
        Ok(Self {
            backend,
            scheduler,
            sequence,
            pooled,
            config,
        })
    }

    pub fn infer(&self, input: InferenceInput<'_>) -> Result<InferenceResult> {
        self.infer_with_observer(input, &mut |_, _| Ok(()))
    }

    pub fn infer_with_observer(
        &self,
        input: InferenceInput<'_>,
        observer: &mut dyn FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<InferenceResult> {
        let (batch, channels, height, width) = input
            .image
            .dims4()
            .map_err(|_| Error::InvalidInput("image must have shape [B,3,H,W]".into()))?;
        let pixels = batch
            .checked_mul(height)
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| Error::InvalidInput("image size overflow".into()))?;
        if batch == 0
            || batch > self.config.limits.max_batch
            || pixels > self.config.limits.max_pixels
            || channels != 3
            || height < 32
            || width < 32
            || height > 1024
            || width > 1024
            || !height.is_multiple_of(8)
            || !width.is_multiple_of(8)
        {
            return Err(Error::InvalidInput("image dimensions or batch exceed limits; H and W must be multiples of 8 in 32..=1024".into()));
        }
        // Also bound intermediate channel multiplications before asking a backend to allocate.
        pixels
            .checked_mul(9)
            .ok_or_else(|| Error::InvalidInput("intermediate size overflow".into()))?;
        let dtype = self.sequence.dtype();
        let device = self.sequence.device();
        let latent_shape = [batch, 4, height / 8, width / 8];
        validate_tensor(
            input.image,
            "image",
            &[batch, 3, height, width],
            DType::F32,
            device,
            Some((0.0, 1.0)),
        )?;
        validate_tensor(
            input.mask,
            "mask",
            &[batch, 1, height, width],
            DType::F32,
            device,
            Some((0.0, 1.0)),
        )?;
        validate_tensor(
            input.posterior_noise,
            "posterior_noise",
            &latent_shape,
            dtype,
            device,
            None,
        )?;
        validate_tensor(
            input.diffusion_noise,
            "diffusion_noise",
            &latent_shape,
            dtype,
            device,
            None,
        )?;
        let normalized = input.image.affine(2.0, -1.0)?.to_dtype(dtype)?;
        let moments = self.backend.encode_moments(&normalized)?;
        validate_tensor(
            &moments,
            "posterior moments",
            &[batch, 8, height / 8, width / 8],
            dtype,
            device,
            None,
        )?;
        observer("posterior_moments", &moments)?;
        let mean = moments.narrow(1, 0, 4)?;
        let logvar = moments.narrow(1, 4, 4)?.clamp(-30.0, 20.0)?;
        observer("posterior_mean", &mean)?;
        observer("posterior_logvar", &logvar)?;
        let sample = (&mean + &(logvar.affine(0.5, 0.0)?.exp()? * input.posterior_noise)?)?;
        validate_tensor(
            &sample,
            "posterior sample",
            &latent_shape,
            dtype,
            device,
            None,
        )?;
        observer("posterior_sample", &sample)?;
        let original = sample.affine(self.config.vae_scaling_factor, 0.0)?;
        validate_tensor(
            &original,
            "original latent",
            &latent_shape,
            dtype,
            device,
            None,
        )?;
        observer("original_latent", &original)?;
        let mask = input.mask.max_pool2d(8)?.to_dtype(dtype)?;
        observer("pooled_mask", &mask)?;
        let noisy =
            self.scheduler
                .add_noise(&original, input.diffusion_noise, self.config.timestep)?;
        validate_tensor(&noisy, "noisy latent", &latent_shape, dtype, device, None)?;
        observer("noisy_latent", &noisy)?;
        let unet_input = Tensor::cat(&[&noisy, &mask, &original], 1)?;
        observer("unet_input", &unet_input)?;
        let sequence = if batch == 1 {
            self.sequence.clone()
        } else {
            self.sequence.repeat((batch, 1, 1))?
        };
        let pooled = if batch == 1 {
            self.pooled.clone()
        } else {
            self.pooled.repeat((batch, 1))?
        };
        let ids = Tensor::new(
            &[
                height as f32,
                width as f32,
                0.0,
                0.0,
                height as f32,
                width as f32,
            ],
            device,
        )?
        .to_dtype(dtype)?
        .unsqueeze(0)?;
        let ids = if batch == 1 {
            ids
        } else {
            ids.repeat((batch, 1))?
        };
        observer("sequence", &sequence)?;
        observer("pooled", &pooled)?;
        observer("time_ids", &ids)?;
        let output =
            self.backend
                .denoise(&unet_input, self.config.timestep, &sequence, &pooled, &ids)?;
        validate_tensor(
            &output,
            "denoiser output",
            &[batch, 5, height / 8, width / 8],
            dtype,
            device,
            None,
        )?;
        let denoising = output.narrow(1, 0, 4)?;
        let logits = output.narrow(1, 4, 1)?;
        observer("denoising_output", &denoising)?;
        observer("alpha_logits", &logits)?;
        let latent_alpha = candle_nn::ops::sigmoid(&logits)?;
        observer("latent_alpha", &latent_alpha)?;
        let predicted =
            self.scheduler
                .predicted_original_sample(&noisy, &denoising, self.config.timestep)?;
        validate_tensor(
            &predicted,
            "predicted original",
            &latent_shape,
            dtype,
            device,
            None,
        )?;
        observer("predicted_original", &predicted)?;
        let blend = (latent_alpha.broadcast_mul(&predicted)?
            + latent_alpha.affine(-1.0, 1.0)?.broadcast_mul(&original)?)?;
        validate_tensor(&blend, "blended latent", &latent_shape, dtype, device, None)?;
        observer("blended_latent", &blend)?;
        let decoded = self
            .backend
            .decode(&blend.affine(1.0 / self.config.vae_scaling_factor, 0.0)?)?;
        validate_tensor(
            &decoded,
            "decoded image",
            &[batch, 3, height, width],
            dtype,
            device,
            None,
        )?;
        // The reference promotes decoder output and alpha to float32 before
        // display-space mapping/interpolation, even for a half-precision model.
        let image = decoded
            .to_dtype(DType::F32)?
            .affine(0.5, 0.5)?
            .clamp(0.0, 1.0)?;
        observer("decoded_image", &image)?;
        let alpha = bilinear_8x(&latent_alpha.to_dtype(DType::F32)?, height, width)?;
        observer("alpha", &alpha)?;
        Ok(InferenceResult {
            image,
            alpha,
            latent_alpha,
        })
    }
}

// Separable bilinear interpolation with PyTorch's align_corners=false coordinates.
fn bilinear_8x(input: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    fn axis(t: &Tensor, dim: usize, target: usize, source: usize) -> Result<Tensor> {
        let mut lo = Vec::with_capacity(target);
        let mut hi = Vec::with_capacity(target);
        let mut weight = Vec::with_capacity(target);
        for i in 0..target {
            let x = ((i as f64 + 0.5) * source as f64 / target as f64 - 0.5)
                .max(0.0)
                .min((source - 1) as f64);
            let left = x.floor() as usize;
            lo.push(
                u32::try_from(left)
                    .map_err(|_| Error::InvalidInput("interpolation size exceeds u32".into()))?,
            );
            hi.push(
                u32::try_from((left + 1).min(source - 1))
                    .map_err(|_| Error::InvalidInput("interpolation size exceeds u32".into()))?,
            );
            weight.push((x - left as f64) as f32);
        }
        let device = t.device();
        let a = t.index_select(&Tensor::new(lo.as_slice(), device)?, dim)?;
        let b = t.index_select(&Tensor::new(hi.as_slice(), device)?, dim)?;
        let shape = if dim == 2 {
            (1, 1, target, 1)
        } else {
            (1, 1, 1, target)
        };
        let w = Tensor::new(weight.as_slice(), device)?
            .to_dtype(t.dtype())?
            .reshape(shape)?;
        Ok((a.broadcast_mul(&w.affine(-1.0, 1.0)?)? + b.broadcast_mul(&w)?)?)
    }
    let (_, _, h, w) = input.dims4()?;
    let vertical = axis(input, 2, height, h)?;
    axis(&vertical, 3, width, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::SchedulerConfig;
    use candle_core::Device;
    use std::cell::{Cell, RefCell};

    #[derive(Default)]
    struct Synthetic {
        decoded: RefCell<Option<Tensor>>,
        conditioning: RefCell<Option<(Tensor, Tensor, Tensor)>>,
        encodes: Cell<usize>,
    }
    impl ModelBackend for Synthetic {
        fn encode_moments(&self, image: &Tensor) -> Result<Tensor> {
            self.encodes.set(self.encodes.get() + 1);
            let (b, _, h, w) = image.dims4()?;
            let mean = Tensor::full(2f32, (b, 4, h / 8, w / 8), image.device())?;
            let logvar = Tensor::zeros((b, 4, h / 8, w / 8), DType::F32, image.device())?;
            Ok(Tensor::cat(&[&mean, &logvar], 1)?)
        }
        fn denoise(
            &self,
            input: &Tensor,
            timestep: usize,
            sequence: &Tensor,
            pooled: &Tensor,
            time_ids: &Tensor,
        ) -> Result<Tensor> {
            assert_eq!(timestep, 0);
            *self.conditioning.borrow_mut() =
                Some((sequence.clone(), pooled.clone(), time_ids.clone()));
            let (b, c, h, w) = input.dims4()?;
            assert_eq!(c, 9);
            let eps = Tensor::full(0.5f32, (b, 4, h, w), input.device())?;
            let logits = Tensor::zeros((b, 1, h, w), DType::F32, input.device())?;
            Ok(Tensor::cat(&[&eps, &logits], 1)?)
        }
        fn decode(&self, latents: &Tensor) -> Result<Tensor> {
            *self.decoded.borrow_mut() = Some(latents.clone());
            let (_, _, h, w) = latents.dims4()?;
            Ok(latents.narrow(1, 0, 3)?.upsample_nearest2d(h * 8, w * 8)?)
        }
    }
    fn pipeline() -> Result<Pipeline<Synthetic>> {
        pipeline_with_limits(InferenceLimits::default())
    }
    fn pipeline_with_limits(limits: InferenceLimits) -> Result<Pipeline<Synthetic>> {
        let scheduler = Scheduler::new(SchedulerConfig {
            num_train_timesteps: 1,
            trained_betas: Some(vec![0.36]),
            ..Default::default()
        })?;
        Pipeline::new(
            Synthetic::default(),
            scheduler,
            Tensor::ones((1, 77, 2048), DType::F32, &Device::Cpu)?,
            Tensor::full(2f32, (1, 1280), &Device::Cpu)?,
            PipelineConfig {
                timestep: 0,
                vae_scaling_factor: 1.0,
                limits,
            },
        )
    }
    fn first(t: &Tensor) -> Result<f32> {
        t.flatten_all()?
            .to_vec1::<f32>()?
            .first()
            .copied()
            .ok_or_else(|| Error::InvalidInput("empty test tensor".into()))
    }
    fn close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }

    #[test]
    fn half_precision_decoder_is_mapped_and_interpolated_in_f32() -> Result<()> {
        struct Half;
        impl ModelBackend for Half {
            fn encode_moments(&self, image: &Tensor) -> Result<Tensor> {
                let (b, _, h, w) = image.dims4()?;
                Ok(Tensor::zeros(
                    (b, 8, h / 8, w / 8),
                    DType::BF16,
                    image.device(),
                )?)
            }
            fn denoise(
                &self,
                input: &Tensor,
                _: usize,
                _: &Tensor,
                _: &Tensor,
                _: &Tensor,
            ) -> Result<Tensor> {
                let (b, _, h, w) = input.dims4()?;
                Ok(Tensor::zeros((b, 5, h, w), DType::BF16, input.device())?)
            }
            fn decode(&self, input: &Tensor) -> Result<Tensor> {
                let (b, _, h, w) = input.dims4()?;
                Ok(
                    Tensor::full(0.001_f32, (b, 3, h * 8, w * 8), input.device())?
                        .to_dtype(DType::BF16)?,
                )
            }
        }
        let device = &Device::Cpu;
        let pipeline = Pipeline::new(
            Half,
            Scheduler::new(SchedulerConfig::default())?,
            Tensor::zeros((1, 77, 2048), DType::BF16, device)?,
            Tensor::zeros((1, 1280), DType::BF16, device)?,
            PipelineConfig::default(),
        )?;
        let image = Tensor::full(0.5_f32, (1, 3, 32, 32), device)?;
        let mask = Tensor::zeros((1, 1, 32, 32), DType::F32, device)?;
        let noise = Tensor::zeros((1, 4, 4, 4), DType::BF16, device)?;
        let result = pipeline.infer(InferenceInput {
            image: &image,
            mask: &mask,
            posterior_noise: &noise,
            diffusion_noise: &noise,
        })?;
        assert_eq!(result.image.dtype(), DType::F32);
        assert_eq!(result.alpha.dtype(), DType::F32);
        assert_eq!(first(&result.image)?, 0.5004997_f32);
        close(first(&result.alpha)?, 0.5);
        Ok(())
    }

    #[test]
    fn explicit_noises_conditioning_and_blend_before_decode() -> Result<()> {
        let p = pipeline()?;
        let image = Tensor::full(0.5f32, (2, 3, 32, 32), &Device::Cpu)?;
        let mut mask_values = vec![0f32; 2 * 32 * 32];
        if let Some(first) = mask_values.first_mut() {
            *first = 1.0;
        }
        let mask = Tensor::from_vec(mask_values, (2, 1, 32, 32), &Device::Cpu)?;
        let posterior = Tensor::ones((2, 4, 4, 4), DType::F32, &Device::Cpu)?;
        let diffusion = Tensor::full(2f32, (2, 4, 4, 4), &Device::Cpu)?;
        let mut stages = Vec::new();
        let result = p.infer_with_observer(
            InferenceInput {
                image: &image,
                mask: &mask,
                posterior_noise: &posterior,
                diffusion_noise: &diffusion,
            },
            &mut |name, t| {
                stages.push(name.to_owned());
                match name {
                    "posterior_sample" => close(first(t)?, 3.0),
                    "pooled_mask" => close(first(t)?, 1.0),
                    "noisy_latent" => close(first(t)?, 3.6),
                    "predicted_original" => close(first(t)?, 4.125),
                    "blended_latent" => close(first(t)?, 3.5625),
                    _ => {}
                }
                Ok(())
            },
        )?;
        assert!(stages.iter().map(String::as_str).eq(STAGES.iter().copied()));
        close(first(&result.latent_alpha)?, 0.5);
        close(first(&result.alpha)?, 0.5);
        close(first(&result.image)?, 1.0);
        let decoded = p.backend.decoded.borrow();
        close(
            first(
                decoded
                    .as_ref()
                    .ok_or_else(|| Error::InvalidInput("decode missing".into()))?,
            )?,
            3.5625,
        );
        let conditioning = p.backend.conditioning.borrow();
        let (sequence, pooled, ids) = conditioning
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("conditioning missing".into()))?;
        assert_eq!(sequence.dims(), &[2, 77, 2048]);
        assert_eq!(pooled.dims(), &[2, 1280]);
        assert_eq!(
            ids.to_vec2::<f32>()?,
            vec![vec![32., 32., 0., 0., 32., 32.]; 2]
        );
        drop(conditioning);
        drop(decoded);
        let zero = Tensor::zeros((2, 4, 4, 4), DType::F32, &Device::Cpu)?;
        p.infer(InferenceInput {
            image: &image,
            mask: &mask,
            posterior_noise: &zero,
            diffusion_noise: &diffusion,
        })?;
        {
            let decoded = p.backend.decoded.borrow();
            close(
                first(
                    decoded
                        .as_ref()
                        .ok_or_else(|| Error::InvalidInput("decode missing".into()))?,
                )?,
                2.5625,
            );
        }
        p.infer(InferenceInput {
            image: &image,
            mask: &mask,
            posterior_noise: &posterior,
            diffusion_noise: &zero,
        })?;
        {
            let decoded = p.backend.decoded.borrow();
            close(
                first(
                    decoded
                        .as_ref()
                        .ok_or_else(|| Error::InvalidInput("decode missing".into()))?,
                )?,
                2.8125,
            );
        }
        p.infer(InferenceInput {
            image: &image,
            mask: &mask,
            posterior_noise: &zero,
            diffusion_noise: &zero,
        })?;
        let decoded = p.backend.decoded.borrow();
        close(
            first(
                decoded
                    .as_ref()
                    .ok_or_else(|| Error::InvalidInput("decode missing".into()))?,
            )?,
            1.8125,
        );
        Ok(())
    }

    #[test]
    fn rejects_bad_input_and_observer_can_cancel() -> Result<()> {
        let p = pipeline()?;
        let image = Tensor::zeros((1, 3, 32, 32), DType::F32, &Device::Cpu)?;
        let mask = Tensor::zeros((1, 1, 32, 32), DType::F32, &Device::Cpu)?;
        let noise = Tensor::zeros((1, 4, 4, 4), DType::F32, &Device::Cpu)?;
        let bad_shape = Tensor::zeros((1, 4, 5, 4), DType::F32, &Device::Cpu)?;
        let nonfinite = Tensor::full(f32::NAN, (1, 4, 4, 4), &Device::Cpu)?;
        let out_of_range = Tensor::full(1.1f32, (1, 1, 32, 32), &Device::Cpu)?;
        let wrong_dtype = noise.to_dtype(DType::F64)?;
        for bad in [&bad_shape, &nonfinite, &wrong_dtype] {
            assert!(matches!(
                p.infer(InferenceInput {
                    image: &image,
                    mask: &mask,
                    posterior_noise: bad,
                    diffusion_noise: &noise
                }),
                Err(Error::InvalidInput(_))
            ));
        }
        assert!(matches!(
            p.infer(InferenceInput {
                image: &image,
                mask: &out_of_range,
                posterior_noise: &noise,
                diffusion_noise: &noise
            }),
            Err(Error::InvalidInput(_))
        ));
        let mut count = 0;
        assert!(matches!(
            p.infer_with_observer(
                InferenceInput {
                    image: &image,
                    mask: &mask,
                    posterior_noise: &noise,
                    diffusion_noise: &noise
                },
                &mut |_, _| {
                    count += 1;
                    Err(Error::Cancelled)
                }
            ),
            Err(Error::Cancelled)
        ));
        assert_eq!(count, 1);
        assert!(p.backend.decoded.borrow().is_none());
        Ok(())
    }

    #[test]
    fn accepts_multiple_of_eight_and_rejects_axis_limit_before_backend() -> Result<()> {
        let p = pipeline()?;
        let image = Tensor::zeros((1, 3, 40, 56), DType::F32, &Device::Cpu)?;
        let mask = Tensor::zeros((1, 1, 40, 56), DType::F32, &Device::Cpu)?;
        let noise = Tensor::zeros((1, 4, 5, 7), DType::F32, &Device::Cpu)?;
        let result = p.infer(InferenceInput {
            image: &image,
            mask: &mask,
            posterior_noise: &noise,
            diffusion_noise: &noise,
        })?;
        assert_eq!(result.image.dims(), &[1, 3, 40, 56]);
        assert_eq!(result.latent_alpha.dims(), &[1, 1, 5, 7]);
        assert_eq!(p.backend.encodes.get(), 1);
        let conditioning = p.backend.conditioning.borrow();
        let (_, _, ids) = conditioning
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("conditioning missing".into()))?;
        assert_eq!(
            ids.to_vec2::<f32>()?,
            vec![vec![40., 56., 0., 0., 40., 56.]]
        );
        drop(conditioning);

        for (height, width) in [(1032, 32), (32, 1032), (31, 32), (32, 41)] {
            let invalid = Tensor::zeros((1, 3, height, width), DType::F32, &Device::Cpu)?;
            assert!(matches!(
                p.infer(InferenceInput {
                    image: &invalid,
                    mask: &mask,
                    posterior_noise: &noise,
                    diffusion_noise: &noise,
                }),
                Err(Error::InvalidInput(_))
            ));
            assert_eq!(p.backend.encodes.get(), 1);
        }
        Ok(())
    }

    #[test]
    fn pixel_and_batch_limits_are_enforced_before_backend() -> Result<()> {
        for limits in [
            InferenceLimits {
                max_batch: 5,
                max_pixels: 1024,
            },
            InferenceLimits {
                max_batch: 4,
                max_pixels: 4 * 1024 * 1024 + 1,
            },
        ] {
            assert!(matches!(
                pipeline_with_limits(limits),
                Err(Error::InvalidInput(_))
            ));
        }
        let p = pipeline_with_limits(InferenceLimits {
            max_batch: 2,
            max_pixels: 2048,
        })?;
        let image = Tensor::zeros((2, 3, 32, 40), DType::F32, &Device::Cpu)?;
        let mask = Tensor::zeros((2, 1, 32, 40), DType::F32, &Device::Cpu)?;
        let noise = Tensor::zeros((2, 4, 4, 5), DType::F32, &Device::Cpu)?;
        assert!(matches!(
            p.infer(InferenceInput {
                image: &image,
                mask: &mask,
                posterior_noise: &noise,
                diffusion_noise: &noise,
            }),
            Err(Error::InvalidInput(_))
        ));
        assert_eq!(p.backend.encodes.get(), 0);
        Ok(())
    }

    #[test]
    fn interpolation_uses_half_pixel_coordinates() -> Result<()> {
        let input = Tensor::new(&[[[[0f32, 1.], [0., 1.]]]], &Device::Cpu)?;
        let row = bilinear_8x(&input, 16, 16)?
            .narrow(0, 0, 1)?
            .narrow(1, 0, 1)?
            .narrow(2, 0, 1)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        close(
            *row.get(3)
                .ok_or_else(|| Error::InvalidInput("short row".into()))?,
            0.0,
        );
        close(
            *row.get(4)
                .ok_or_else(|| Error::InvalidInput("short row".into()))?,
            0.0625,
        );
        close(
            *row.get(7)
                .ok_or_else(|| Error::InvalidInput("short row".into()))?,
            0.4375,
        );
        close(
            *row.get(11)
                .ok_or_else(|| Error::InvalidInput("short row".into()))?,
            0.9375,
        );
        close(
            *row.get(12)
                .ok_or_else(|| Error::InvalidInput("short row".into()))?,
            1.0,
        );
        Ok(())
    }
}
