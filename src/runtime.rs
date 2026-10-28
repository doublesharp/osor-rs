//! Application-facing model ownership and explicit backend selection.

use std::path::Path;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};

use crate::{
    Error, Result,
    artifact::{ArtifactLimits, Manifest, ModelBundle},
    native::NativeModel,
    pipeline::{
        InferenceInput, InferenceLimits, InferenceResult, ModelBackend, Pipeline, PipelineConfig,
    },
    scheduler::Scheduler,
};

/// Devices are explicit: selecting an unavailable accelerator returns an error.
#[derive(Debug, Clone, Copy, Default)]
pub enum Backend {
    #[default]
    Cpu,
    Metal {
        ordinal: usize,
    },
    Cuda {
        ordinal: usize,
    },
    Wgpu {
        ordinal: usize,
    },
}

impl Backend {
    /// Device for application input tensors. WGPU uses CPU staging around its
    /// resident GPU neural graph; its compute adapter is initialized on load.
    pub fn device(self) -> Result<Device> {
        match self {
            Self::Cpu => Ok(Device::Cpu),
            Self::Metal { ordinal } => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    Ok(Device::new_metal(ordinal)?)
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    let _ = ordinal;
                    Err(Error::Unavailable(
                        "Metal requires macOS and the metal Cargo feature".into(),
                    ))
                }
            }
            Self::Cuda { ordinal } => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Device::new_cuda(ordinal)?)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = ordinal;
                    Err(Error::Unavailable(
                        "CUDA requires the cuda Cargo feature and a supported driver".into(),
                    ))
                }
            }
            Self::Wgpu { ordinal } => {
                #[cfg(feature = "wgpu")]
                {
                    if ordinal != 0 {
                        return Err(Error::Unavailable(
                            "WGPU currently supports ordinal 0".into(),
                        ));
                    }
                    Ok(Device::Cpu)
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    let _ = ordinal;
                    Err(Error::Unavailable(
                        "WGPU requires the wgpu Cargo feature".into(),
                    ))
                }
            }
        }
    }
}

enum RuntimeModel {
    Native(Box<NativeModel>),
    #[cfg(feature = "wgpu")]
    Wgpu(Box<crate::wgpu::WgpuModel>),
}

impl ModelBackend for RuntimeModel {
    fn encode_moments(&self, image: &Tensor) -> Result<Tensor> {
        match self {
            Self::Native(model) => model.encode_moments(image),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(model) => model.encode_moments(image),
        }
    }

    fn denoise(
        &self,
        input: &Tensor,
        timestep: usize,
        sequence: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
    ) -> Result<Tensor> {
        match self {
            Self::Native(model) => model.denoise(input, timestep, sequence, pooled, time_ids),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(model) => model.denoise(input, timestep, sequence, pooled, time_ids),
        }
    }

    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        match self {
            Self::Native(model) => model.decode(latents),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(model) => model.decode(latents),
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub backend: Backend,
    pub dtype: DType,
    pub limits: InferenceLimits,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            backend: Backend::Cpu,
            dtype: DType::F32,
            limits: InferenceLimits::default(),
        }
    }
}

/// Loaded OSOR model. Inference uses caller-supplied noise for reproducibility.
///
/// Applications own model installation, job admission, and threading. Inference
/// is synchronous. The observer variant supports cancellation between stages.
pub struct Osor {
    pipeline: Pipeline<RuntimeModel>,
    manifest: Manifest,
    device: Device,
    dtype: DType,
    backend: Backend,
    compute_device: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StageTiming {
    pub stage: String,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InferenceProfile {
    pub total_ms: f64,
    pub stages: Vec<StageTiming>,
}

impl Osor {
    /// Load a pinned bundle with caller-specified artifact limits.
    /// The aggregate tensor limit counts decoded bytes in the selected dtype,
    /// not peak memory use.
    pub fn load_with_options_and_manifest_sha256_and_limits(
        path: impl AsRef<Path>,
        options: LoadOptions,
        expected_manifest_sha256: &str,
        limits: &ArtifactLimits,
    ) -> Result<Self> {
        Self::load_internal(
            path.as_ref(),
            options,
            Some(expected_manifest_sha256),
            limits,
        )
    }

    /// Load using a caller-trusted SHA-256 of the exact bundle manifest bytes.
    pub fn load_with_options_and_manifest_sha256(
        path: impl AsRef<Path>,
        options: LoadOptions,
        expected_manifest_sha256: &str,
    ) -> Result<Self> {
        Self::load_with_options_and_manifest_sha256_and_limits(
            path,
            options,
            expected_manifest_sha256,
            &ArtifactLimits::default(),
        )
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_options(path, LoadOptions::default())
    }

    pub fn load_with_options(path: impl AsRef<Path>, options: LoadOptions) -> Result<Self> {
        Self::load_internal(path.as_ref(), options, None, &ArtifactLimits::default())
    }

    fn load_internal(
        path: &Path,
        options: LoadOptions,
        manifest_sha256: Option<&str>,
        artifact_limits: &ArtifactLimits,
    ) -> Result<Self> {
        if !matches!(options.dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::InvalidInput(
                "inference dtype must be F32, F16, or BF16".into(),
            ));
        }
        if matches!(options.backend, Backend::Cpu | Backend::Wgpu { .. })
            && options.dtype != DType::F32
        {
            return Err(Error::Unavailable(
                "CPU and WGPU inference currently require F32".into(),
            ));
        }
        if options.limits.max_batch == 0
            || options.limits.max_batch > 16
            || options.limits.max_pixels == 0
            || options.limits.max_pixels > 16 * 1024 * 1024
        {
            return Err(Error::InvalidInput(
                "invalid inference admission limits".into(),
            ));
        }
        let device = options.backend.device()?;
        let bundle = match manifest_sha256 {
            Some(hash) => ModelBundle::load_with_limits_and_manifest_sha256(
                path,
                &device,
                options.dtype,
                artifact_limits,
                hash,
            )?,
            None => ModelBundle::load_with_limits(path, &device, options.dtype, artifact_limits)?,
        };
        let mut limits = options.limits;
        let (model, compute_device) = match options.backend {
            Backend::Wgpu { ordinal } => {
                #[cfg(feature = "wgpu")]
                {
                    limits.max_batch = limits.max_batch.min(1);
                    limits.max_pixels = limits.max_pixels.min(512 * 512);
                    let model = crate::wgpu::WgpuModel::new(bundle.unet, bundle.vae, ordinal)?;
                    let description = model.device_description();
                    (RuntimeModel::Wgpu(Box::new(model)), description)
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    let _ = ordinal;
                    return Err(Error::Unavailable(
                        "WGPU requires the wgpu Cargo feature".into(),
                    ));
                }
            }
            _ => {
                limits.max_batch = limits.max_batch.min(4);
                (
                    RuntimeModel::Native(Box::new(NativeModel::new(
                        bundle.unet,
                        bundle.vae,
                        options.dtype,
                        &device,
                    )?)),
                    format!("{:?}", options.backend),
                )
            }
        };
        let scheduler = Scheduler::new(bundle.manifest.scheduler.clone())?;
        let config = PipelineConfig {
            timestep: bundle.manifest.timestep,
            vae_scaling_factor: bundle.manifest.vae_scaling_factor,
            limits,
        };
        let pipeline = Pipeline::new(model, scheduler, bundle.sequence, bundle.pooled, config)?;
        Ok(Self {
            pipeline,
            manifest: bundle.manifest,
            device,
            dtype: options.dtype,
            backend: options.backend,
            compute_device,
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Actual compute adapter, distinct from WGPU's CPU input staging device.
    pub fn compute_device_description(&self) -> &str {
        &self.compute_device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Return the full decoded image and predicted edit alpha. Effects may extend
    /// beyond the supplied removal mask; there is no implicit RGB paste-back.
    pub fn infer(&self, input: InferenceInput<'_>) -> Result<InferenceResult> {
        self.pipeline.infer(input)
    }

    pub fn infer_with_observer(
        &self,
        input: InferenceInput<'_>,
        observer: &mut dyn FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<InferenceResult> {
        self.pipeline.infer_with_observer(input, observer)
    }

    /// Measure completed device work between reference stages. Synchronization
    /// adds profiling overhead; use `infer` when measuring normal throughput.
    pub fn infer_profiled(
        &self,
        input: InferenceInput<'_>,
    ) -> Result<(InferenceResult, InferenceProfile)> {
        self.device.synchronize()?;
        let start = Instant::now();
        let mut previous = start;
        let mut stages = Vec::new();
        let result = self.infer_with_observer(input, &mut |stage, _tensor| {
            self.device.synchronize()?;
            let now = Instant::now();
            stages.push(StageTiming {
                stage: stage.into(),
                elapsed_ms: now.duration_since(previous).as_secs_f64() * 1000.0,
            });
            previous = now;
            Ok(())
        })?;
        self.device.synchronize()?;
        Ok((
            result,
            InferenceProfile {
                total_ms: start.elapsed().as_secs_f64() * 1000.0,
                stages,
            },
        ))
    }
}

impl ModelBackend for NativeModel {
    fn encode_moments(&self, image: &Tensor) -> Result<Tensor> {
        NativeModel::encode_moments(self, image)
    }

    fn denoise(
        &self,
        input: &Tensor,
        timestep: usize,
        sequence: &Tensor,
        pooled: &Tensor,
        time_ids: &Tensor,
    ) -> Result<Tensor> {
        NativeModel::denoise(self, input, timestep, sequence, pooled, time_ids)
    }

    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        NativeModel::decode(self, latents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_is_default_and_missing_gpu_features_return_unavailable() -> Result<()> {
        assert!(Backend::default().device()?.is_cpu());
        #[cfg(not(feature = "metal"))]
        assert!(matches!(
            Backend::Metal { ordinal: 0 }.device(),
            Err(Error::Unavailable(_))
        ));
        #[cfg(not(feature = "cuda"))]
        assert!(matches!(
            Backend::Cuda { ordinal: 0 }.device(),
            Err(Error::Unavailable(_))
        ));
        #[cfg(not(feature = "wgpu"))]
        assert!(matches!(
            Backend::Wgpu { ordinal: 0 }.device(),
            Err(Error::Unavailable(_))
        ));
        Ok(())
    }

    #[test]
    fn unsupported_cpu_precision_fails_before_model_io() {
        let result = Osor::load_with_options(
            "missing-model",
            LoadOptions {
                dtype: DType::BF16,
                ..LoadOptions::default()
            },
        );
        assert!(matches!(result, Err(Error::Unavailable(_))));
    }
}
