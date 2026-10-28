//! Application-facing model ownership and explicit backend selection.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};

use crate::{
    Error, Result,
    artifact::{ArtifactLimits, Manifest, ModelBundle},
    native::NativeModel,
    pipeline::{
        InferenceInput, InferenceLimits, InferenceResult, ModelBackend, Pipeline, PipelineConfig,
        Stage,
    },
    scheduler::Scheduler,
};

/// Devices are explicit: selecting an unavailable accelerator returns an error.
#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub enum Backend {
    #[default]
    /// Portable Candle CPU execution; F32 only.
    Cpu,
    /// Candle Metal on macOS; requires the `metal` feature.
    Metal {
        /// Metal device index.
        ordinal: usize,
    },
    /// Candle CUDA; requires the `cuda` feature and an NVIDIA device.
    Cuda {
        /// CUDA device index.
        ordinal: usize,
    },
    /// Burn WGPU with CPU staging; requires the `wgpu` feature; F32 only.
    Wgpu {
        /// Adapter index; only 0 is supported.
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

    fn denoise(&self, input: &Tensor, timestep: usize, time_ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Native(model) => model.denoise(input, timestep, time_ids),
            #[cfg(feature = "wgpu")]
            Self::Wgpu(model) => model.denoise(input, timestep, time_ids),
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
/// Backend, dtype and admission limits for loading a model. See also [`Osor::builder`].
pub struct LoadOptions {
    /// Compute backend; defaults to CPU.
    pub backend: Backend,
    /// Model dtype; defaults to F32. CPU and WGPU require F32.
    pub dtype: DType,
    /// Requested job admission limits, validated before any bundle I/O.
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
/// Synchronized time spent reaching one reference stage.
pub struct StageTiming {
    /// Pinned stage name, as in [`Stage::as_str`].
    pub stage: String,
    /// Milliseconds since the previous stage completed.
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
/// Stage timings for one profiled inference.
pub struct InferenceProfile {
    /// Wall-clock milliseconds for the whole job, including synchronization.
    pub total_ms: f64,
    /// Per-stage timings in observer order.
    pub stages: Vec<StageTiming>,
}

/// Configures and loads an [`Osor`]; created by [`Osor::builder`].
///
/// ```no_run
/// # fn main() -> osor_rs::Result<()> {
/// use osor_rs::{Backend, Osor};
///
/// let model = Osor::builder("/path/to/bundle")
///     .backend(Backend::Cpu)
///     .trusted_manifest_sha256("75324653945a3c74b69fdfd2bde8b852e858bae461a3d9992ce48168c6987a0f")
///     .load()?;
/// # let _ = model;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[must_use = "call `load` to read the bundle"]
pub struct OsorBuilder {
    path: PathBuf,
    options: LoadOptions,
    manifest_sha256: Option<String>,
    artifact_limits: ArtifactLimits,
}

impl OsorBuilder {
    /// Compute backend; defaults to CPU.
    pub fn backend(mut self, backend: Backend) -> Self {
        self.options.backend = backend;
        self
    }

    /// Model dtype; defaults to F32. CPU and WGPU require F32.
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.options.dtype = dtype;
        self
    }

    /// Requested job admission limits, validated before any bundle I/O.
    pub fn inference_limits(mut self, limits: InferenceLimits) -> Self {
        self.options.limits = limits;
        self
    }

    /// Require the exact manifest bytes to match a SHA-256 retained from a
    /// trusted import, checked before any weight file is opened.
    pub fn trusted_manifest_sha256(mut self, digest: impl Into<String>) -> Self {
        self.manifest_sha256 = Some(digest.into());
        self
    }

    /// Artifact admission limits, including the aggregate decoded-weight cap.
    pub fn artifact_limits(mut self, limits: ArtifactLimits) -> Self {
        self.artifact_limits = limits;
        self
    }

    /// Validate options, then read, verify and load the bundle.
    pub fn load(self) -> Result<Osor> {
        Osor::load_internal(
            &self.path,
            self.options,
            self.manifest_sha256.as_deref(),
            &self.artifact_limits,
        )
    }
}

impl Osor {
    /// Start configuring a load of the bundle directory at `path`.
    pub fn builder(path: impl Into<PathBuf>) -> OsorBuilder {
        OsorBuilder {
            path: path.into(),
            options: LoadOptions::default(),
            manifest_sha256: None,
            artifact_limits: ArtifactLimits::default(),
        }
    }

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

    /// Load the bundle at `path` on CPU in F32 with default limits.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::load_with_options(path, LoadOptions::default())
    }

    /// Load the bundle at `path` with explicit backend, dtype and limits.
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
        // Reject before reading or hashing any bundle bytes.
        options.limits.validate()?;
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
        let (model, compute_device, limits) = match options.backend {
            Backend::Wgpu { ordinal } => {
                #[cfg(feature = "wgpu")]
                {
                    // WGPU admits less than the shared bound; see `Osor::inference_limits`.
                    let limits = InferenceLimits {
                        max_batch: options.limits.max_batch.min(1),
                        max_pixels: options.limits.max_pixels.min(crate::wgpu::MAX_PIXELS),
                    };
                    let model = crate::wgpu::WgpuModel::new(
                        bundle.unet,
                        bundle.vae,
                        &bundle.sequence,
                        &bundle.pooled,
                        ordinal,
                    )?;
                    let description = model.device_description();
                    (RuntimeModel::Wgpu(Box::new(model)), description, limits)
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    let _ = ordinal;
                    return Err(Error::Unavailable(
                        "WGPU requires the wgpu Cargo feature".into(),
                    ));
                }
            }
            _ => (
                RuntimeModel::Native(Box::new(NativeModel::new(
                    bundle.unet,
                    bundle.vae,
                    &bundle.sequence,
                    &bundle.pooled,
                    options.dtype,
                    &device,
                )?)),
                format!("{:?}", options.backend),
                options.limits,
            ),
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

    /// Validated manifest of the loaded bundle.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Device that inference inputs must be on. For WGPU this is the CPU staging device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Backend selected at load.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Actual compute adapter, distinct from WGPU's CPU input staging device.
    pub fn compute_device_description(&self) -> &str {
        &self.compute_device
    }

    /// Model dtype; posterior and diffusion noise must use it.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Effective admission limits. WGPU lowers requested limits to batch one
    /// and its own pixel bound; other backends use the requested limits.
    pub fn inference_limits(&self) -> InferenceLimits {
        self.pipeline.limits()
    }

    /// Return the full decoded image and predicted edit alpha. Effects may extend
    /// beyond the supplied removal mask; there is no implicit RGB paste-back.
    pub fn infer(&self, input: InferenceInput<'_>) -> Result<InferenceResult> {
        self.pipeline.infer(input)
    }

    /// Run one job, reporting each reference stage in order. Return
    /// [`Error::Cancelled`] from `observer` to stop between stages.
    pub fn infer_with_observer(
        &self,
        input: InferenceInput<'_>,
        observer: &mut dyn FnMut(Stage, &Tensor) -> Result<()>,
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
                stage: stage.as_str().into(),
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

    fn denoise(&self, input: &Tensor, timestep: usize, time_ids: &Tensor) -> Result<Tensor> {
        NativeModel::denoise(self, input, timestep, time_ids)
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
    fn builder_validates_options_and_pin_before_bundle_io() {
        assert!(matches!(
            Osor::builder("missing-model").dtype(DType::BF16).load(),
            Err(Error::Unavailable(_))
        ));
        assert!(matches!(
            Osor::builder("missing-model")
                .inference_limits(InferenceLimits {
                    max_batch: 5,
                    max_pixels: 1024,
                })
                .load(),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            Osor::builder("missing-model")
                .trusted_manifest_sha256("bad")
                .load(),
            Err(Error::InvalidArtifact(message)) if message.contains("64 hexadecimal")
        ));
    }

    #[test]
    fn loaded_models_can_be_shared_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Osor>();
    }

    #[test]
    fn out_of_range_limits_fail_before_bundle_io() {
        for limits in [
            InferenceLimits {
                max_batch: 5,
                max_pixels: 1024,
            },
            InferenceLimits {
                max_batch: 1,
                max_pixels: crate::pipeline::MAX_PIXELS + 1,
            },
            InferenceLimits {
                max_batch: 0,
                max_pixels: 1024,
            },
        ] {
            // A missing bundle would report an I/O or artifact error if loading began.
            let result = Osor::load_with_options(
                "missing-model",
                LoadOptions {
                    limits,
                    ..LoadOptions::default()
                },
            );
            assert!(
                matches!(result, Err(Error::InvalidInput(_))),
                "limits {limits:?} were not rejected before bundle I/O"
            );
        }
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
