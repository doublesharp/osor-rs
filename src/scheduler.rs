//! The deterministic predicted-original portion of the DDPM scheduler.
use candle_core::Tensor;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

use crate::{Error, Result};

/// Diffusers DDPM configuration. Euler-only metadata from the base config is
/// accepted during deserialization but does not participate in DDPM math.
#[derive(Debug, Clone, Serialize)]
pub struct SchedulerConfig {
    /// Number of training timesteps.
    pub num_train_timesteps: usize,
    /// First beta of the schedule.
    pub beta_start: f64,
    /// Last beta of the schedule.
    pub beta_end: f64,
    /// Beta schedule name, such as `scaled_linear`.
    pub beta_schedule: String,
    /// Explicit betas overriding the schedule.
    pub trained_betas: Option<Vec<f64>>,
    /// `epsilon`, `sample` or `v_prediction`.
    pub prediction_type: String,
    /// Clamp predicted original samples.
    pub clip_sample: bool,
    /// Clamp bound when `clip_sample` is set.
    pub clip_sample_range: f64,
    /// Dynamic thresholding; must be false.
    pub thresholding: bool,
    /// Dynamic thresholding ratio (unused when disabled).
    pub dynamic_thresholding_ratio: f64,
    /// Dynamic thresholding bound (unused when disabled).
    pub sample_max_value: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            beta_schedule: "scaled_linear".into(),
            trained_betas: None,
            prediction_type: "epsilon".into(),
            clip_sample: false,
            clip_sample_range: 1.0,
            thresholding: false,
            dynamic_thresholding_ratio: 0.995,
            sample_max_value: 1.0,
        }
    }
}

impl<'de> Deserialize<'de> for SchedulerConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let mut object = serde_json::Map::<String, serde_json::Value>::deserialize(deserializer)?;
        // These fields are specific to the source Euler scheduler or diffusers metadata.
        for name in [
            "_class_name",
            "_diffusers_version",
            "interpolation_type",
            "set_alpha_to_one",
            "skip_prk_steps",
            "steps_offset",
            "timestep_spacing",
            "use_karras_sigmas",
        ] {
            object.remove(name);
        }
        #[derive(Deserialize)]
        #[serde(default, deny_unknown_fields)]
        struct Fields {
            num_train_timesteps: usize,
            beta_start: f64,
            beta_end: f64,
            beta_schedule: String,
            trained_betas: Option<Vec<f64>>,
            prediction_type: String,
            clip_sample: bool,
            clip_sample_range: f64,
            thresholding: bool,
            dynamic_thresholding_ratio: f64,
            sample_max_value: f64,
        }
        impl Default for Fields {
            fn default() -> Self {
                let c = SchedulerConfig::default();
                Self {
                    num_train_timesteps: c.num_train_timesteps,
                    beta_start: c.beta_start,
                    beta_end: c.beta_end,
                    beta_schedule: c.beta_schedule,
                    trained_betas: c.trained_betas,
                    prediction_type: c.prediction_type,
                    clip_sample: c.clip_sample,
                    clip_sample_range: c.clip_sample_range,
                    thresholding: c.thresholding,
                    dynamic_thresholding_ratio: c.dynamic_thresholding_ratio,
                    sample_max_value: c.sample_max_value,
                }
            }
        }
        let f: Fields =
            serde_json::from_value(serde_json::Value::Object(object)).map_err(D::Error::custom)?;
        Ok(Self {
            num_train_timesteps: f.num_train_timesteps,
            beta_start: f.beta_start,
            beta_end: f.beta_end,
            beta_schedule: f.beta_schedule,
            trained_betas: f.trained_betas,
            prediction_type: f.prediction_type,
            clip_sample: f.clip_sample,
            clip_sample_range: f.clip_sample_range,
            thresholding: f.thresholding,
            dynamic_thresholding_ratio: f.dynamic_thresholding_ratio,
            sample_max_value: f.sample_max_value,
        })
    }
}

/// Precomputed cumulative alphas; no stochastic scheduler step is performed.
pub struct Scheduler {
    config: SchedulerConfig,
    alpha_cumprod: Vec<f32>,
}

impl Scheduler {
    /// Validate the configuration and precompute cumulative alphas.
    pub fn new(config: SchedulerConfig) -> Result<Self> {
        let invalid = |message: &str| Error::InvalidInput(format!("scheduler: {message}"));
        if config.num_train_timesteps == 0 || config.num_train_timesteps > 100_000 {
            return Err(invalid("num_train_timesteps must be in 1..=100000"));
        }
        if !config.beta_start.is_finite()
            || !config.beta_end.is_finite()
            || !(0.0..1.0).contains(&config.beta_start)
            || !(0.0..1.0).contains(&config.beta_end)
        {
            return Err(invalid("betas must be finite and in [0, 1)"));
        }
        if !config.clip_sample_range.is_finite()
            || config.clip_sample_range <= 0.0
            || !config.dynamic_thresholding_ratio.is_finite()
            || !(0.0..=1.0).contains(&config.dynamic_thresholding_ratio)
            || !config.sample_max_value.is_finite()
            || config.sample_max_value < 1.0
        {
            return Err(invalid("invalid clipping or thresholding parameters"));
        }
        if config.thresholding {
            return Err(Error::Unavailable(
                "DDPM dynamic thresholding is not implemented".into(),
            ));
        }
        if !matches!(
            config.prediction_type.as_str(),
            "epsilon" | "sample" | "v_prediction"
        ) {
            return Err(invalid("unsupported prediction_type"));
        }
        if !matches!(config.beta_schedule.as_str(), "linear" | "scaled_linear") {
            return Err(invalid("unsupported beta_schedule"));
        }
        if let Some(betas) = &config.trained_betas
            && (betas.len() != config.num_train_timesteps
                || betas
                    .iter()
                    .any(|b| !b.is_finite() || !(0.0..1.0).contains(b)))
        {
            return Err(invalid(
                "trained_betas must match num_train_timesteps and be finite in [0, 1)",
            ));
        }
        // Diffusers constructs float32 betas. PyTorch's CPU linspace uses a
        // float32 step and fused endpoint interpolation; cumprod accumulates
        // those float32 alphas in f64 and stores float32 results.
        let (start, end) = if config.beta_schedule == "scaled_linear" {
            (
                config.beta_start.sqrt() as f32,
                config.beta_end.sqrt() as f32,
            )
        } else {
            (config.beta_start as f32, config.beta_end as f32)
        };
        let step = if config.num_train_timesteps == 1 {
            0.0
        } else {
            (end - start) / (config.num_train_timesteps - 1) as f32
        };
        let mut alpha = 1.0_f64;
        let mut alpha_cumprod = Vec::with_capacity(config.num_train_timesteps);
        for i in 0..config.num_train_timesteps {
            let beta = if let Some(betas) = &config.trained_betas {
                *betas
                    .get(i)
                    .ok_or_else(|| invalid("missing trained beta"))? as f32
            } else {
                let point = if i < config.num_train_timesteps / 2 || config.num_train_timesteps == 1
                {
                    step.mul_add(i as f32, start)
                } else {
                    (-step).mul_add((config.num_train_timesteps - i - 1) as f32, end)
                };
                if config.beta_schedule == "scaled_linear" {
                    point * point
                } else {
                    point
                }
            };
            alpha *= f64::from(1.0_f32 - beta);
            let stored = alpha as f32;
            if !stored.is_finite() || stored <= 0.0 {
                return Err(invalid("cumulative alpha underflowed"));
            }
            alpha_cumprod.push(stored);
        }
        Ok(Self {
            config,
            alpha_cumprod,
        })
    }

    /// Configuration used to build this scheduler.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    fn alpha(&self, timestep: usize) -> Result<f32> {
        self.alpha_cumprod
            .get(timestep)
            .copied()
            .ok_or_else(|| Error::InvalidInput("scheduler timestep out of range".into()))
    }

    /// Shape, dtype, device and size checks; no device readback.
    fn validate_layout(a: &Tensor, b: &Tensor) -> Result<()> {
        if a.dims() != b.dims() || a.dtype() != b.dtype() || !a.device().same_device(b.device()) {
            return Err(Error::InvalidInput(
                "scheduler tensors must have matching shape, dtype, and device".into(),
            ));
        }
        if !matches!(
            a.dtype(),
            candle_core::DType::F16
                | candle_core::DType::BF16
                | candle_core::DType::F32
                | candle_core::DType::F64
        ) {
            return Err(Error::InvalidInput(
                "scheduler tensors must have floating dtype".into(),
            ));
        }
        if a.elem_count() > 16 * 1024 * 1024 {
            return Err(Error::InvalidInput(
                "scheduler tensor exceeds safety limit".into(),
            ));
        }
        Ok(())
    }

    /// Layout checks plus a host readback rejecting non-finite values.
    fn validate_pair(a: &Tensor, b: &Tensor) -> Result<()> {
        Self::validate_layout(a, b)?;
        for tensor in [a, b] {
            let values = tensor
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::InvalidInput(
                    "scheduler tensors must contain finite values".into(),
                ));
            }
        }
        Ok(())
    }

    /// Noise `original` with explicit `noise` at `timestep`.
    pub fn add_noise(&self, original: &Tensor, noise: &Tensor, timestep: usize) -> Result<Tensor> {
        Self::validate_pair(original, noise)?;
        self.add_noise_prechecked(original, noise, timestep)
    }

    /// [`Self::add_noise`] for callers that already rejected non-finite inputs.
    pub(crate) fn add_noise_prechecked(
        &self,
        original: &Tensor,
        noise: &Tensor,
        timestep: usize,
    ) -> Result<Tensor> {
        Self::validate_layout(original, noise)?;
        let a = self.alpha(timestep)?;
        let a = Tensor::new(a, original.device())?.to_dtype(original.dtype())?;
        let original_scale = a.sqrt()?;
        let noise_scale = a.affine(-1.0, 1.0)?.sqrt()?;
        Ok((original.broadcast_mul(&original_scale)? + noise.broadcast_mul(&noise_scale)?)?)
    }

    /// Predicted original sample from a noisy input and model output at `timestep`.
    pub fn predicted_original_sample(
        &self,
        noisy: &Tensor,
        model_output: &Tensor,
        timestep: usize,
    ) -> Result<Tensor> {
        Self::validate_pair(noisy, model_output)?;
        self.predicted_original_sample_prechecked(noisy, model_output, timestep)
    }

    /// [`Self::predicted_original_sample`] for callers that already rejected
    /// non-finite inputs.
    pub(crate) fn predicted_original_sample_prechecked(
        &self,
        noisy: &Tensor,
        model_output: &Tensor,
        timestep: usize,
    ) -> Result<Tensor> {
        Self::validate_layout(noisy, model_output)?;
        let a = self.alpha(timestep)?;
        let original_scale = Tensor::new(a.sqrt(), noisy.device())?.to_dtype(noisy.dtype())?;
        let noise_scale =
            Tensor::new((1.0_f32 - a).sqrt(), noisy.device())?.to_dtype(noisy.dtype())?;
        let prediction = match self.config.prediction_type.as_str() {
            "epsilon" => (noisy - &model_output.broadcast_mul(&noise_scale)?)?
                .broadcast_div(&original_scale)?,
            "sample" => model_output.clone(),
            "v_prediction" => {
                (noisy.broadcast_mul(&original_scale)?
                    - model_output.broadcast_mul(&noise_scale)?)?
            }
            _ => return Err(Error::InvalidInput("unsupported prediction_type".into())),
        };
        Ok(if self.config.clip_sample {
            prediction.clamp(
                -self.config.clip_sample_range,
                self.config.clip_sample_range,
            )?
        } else {
            prediction
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn pinned_fp32_scheduler_matches_captured_reference() -> Result<()> {
        // PyTorch 2.7.1 / diffusers 0.35.1, pinned base scheduler, CPU fp32.
        // Values capture the reference's coefficient precision and operation order.
        let scheduler = Scheduler::new(SchedulerConfig::default())?;
        let sample = Tensor::new(&[[2_f32, -3., 0.01, 100.]], &Device::Cpu)?;
        let noise = Tensor::new(&[[0.5_f32, 0.125, 2., -4.]], &Device::Cpu)?;
        let noisy = scheduler
            .add_noise(&sample, &noise, 400)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            noisy.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![0x3fd7579c, 0xbfee0c04, 0x3fc30af8, 0x427878a8]
        );
        let predicted = scheduler
            .predicted_original_sample(&sample, &noise, 400)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(
            predicted.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vec![0x401f33bd, 0xc0980129, 0xc0140f51, 0x431e24dd]
        );
        Ok(())
    }

    #[test]
    fn ddpm_predictions_and_clipping() -> Result<()> {
        let x = Tensor::new(&[2f32], &Device::Cpu)?;
        let n = Tensor::new(&[0.5f32], &Device::Cpu)?;
        for (kind, expected) in [
            ("epsilon", 2.0 / 0.9f64.sqrt() - 0.5 * (0.1f64 / 0.9).sqrt()),
            ("sample", 0.5),
            ("v_prediction", 2.0 * 0.9f64.sqrt() - 0.5 * 0.1f64.sqrt()),
        ] {
            let config = SchedulerConfig {
                num_train_timesteps: 1,
                trained_betas: Some(vec![0.1]),
                prediction_type: kind.into(),
                ..Default::default()
            };
            let scheduler = Scheduler::new(config)?;
            let noisy = scheduler.add_noise(&x, &n, 0)?.to_vec1::<f32>()?;
            assert!(
                (f64::from(noisy[0]) - (2.0 * 0.9f64.sqrt() + 0.5 * 0.1f64.sqrt())).abs() < 1e-6
            );
            let value = scheduler
                .predicted_original_sample(&x, &n, 0)?
                .to_vec1::<f32>()?;
            assert!((f64::from(value[0]) - expected).abs() < 1e-6);
        }
        let scheduler = Scheduler::new(SchedulerConfig {
            num_train_timesteps: 1,
            trained_betas: Some(vec![0.1]),
            clip_sample: true,
            clip_sample_range: 0.4,
            ..Default::default()
        })?;
        assert!(
            (scheduler
                .predicted_original_sample(&x, &n, 0)?
                .to_vec1::<f32>()?[0]
                - 0.4)
                .abs()
                < 1e-6
        );
        assert!(scheduler.add_noise(&x, &n, 1).is_err());
        Ok(())
    }

    #[test]
    fn prechecked_paths_match_public_results_and_check_layout() -> Result<()> {
        let scheduler = Scheduler::new(SchedulerConfig::default())?;
        let device = &candle_core::Device::Cpu;
        let x = Tensor::new(&[0.5_f32, -1.0, 2.0, 0.25], device)?;
        let n = Tensor::new(&[1.0_f32, 0.0, -0.5, 3.0], device)?;
        let same = |a: &Tensor, b: &Tensor| -> Result<bool> {
            Ok(a.to_vec1::<f32>()? == b.to_vec1::<f32>()?)
        };
        assert!(same(
            &scheduler.add_noise(&x, &n, 400)?,
            &scheduler.add_noise_prechecked(&x, &n, 400)?
        )?);
        assert!(same(
            &scheduler.predicted_original_sample(&x, &n, 400)?,
            &scheduler.predicted_original_sample_prechecked(&x, &n, 400)?
        )?);
        let short = Tensor::new(&[1.0_f32], device)?;
        assert!(scheduler.add_noise_prechecked(&x, &short, 400).is_err());
        assert!(
            scheduler
                .predicted_original_sample_prechecked(&x, &short, 400)
                .is_err()
        );
        let nan = Tensor::new(&[f32::NAN, 0.0, 0.0, 0.0], device)?;
        assert!(scheduler.add_noise(&x, &nan, 400).is_err());
        Ok(())
    }

    #[test]
    fn metadata_and_unsupported_configuration() -> Result<()> {
        let config: SchedulerConfig = serde_json::from_str(
            r#"{"_class_name":"EulerDiscreteScheduler","interpolation_type":"linear","clip_sample":false}"#,
        )?;
        assert_eq!(config.beta_schedule, "scaled_linear");
        assert!(serde_json::from_str::<SchedulerConfig>(r#"{"mystery":true}"#).is_err());
        assert!(matches!(
            Scheduler::new(SchedulerConfig {
                thresholding: true,
                ..Default::default()
            }),
            Err(Error::Unavailable(_))
        ));
        Ok(())
    }
}
