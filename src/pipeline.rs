//! Explicit-noise OSOR inference, independent of a particular model backend.
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::{Error, Result, scheduler::Scheduler};
use candle_core::{DType, Tensor};

/// Called by backends between neural blocks; returns [`Error::Cancelled`] to
/// stop the job.
pub type Checkpoint<'a> = &'a dyn Fn() -> Result<()>;

/// Shared flag that stops an inference at its next checkpoint.
///
/// Clone it to another thread and call [`Self::cancel`]. Native CPU work stops
/// between encoder, UNet and decoder blocks. GPU backends queue work ahead of
/// the CPU, so already queued GPU work still completes before the job returns
/// [`Error::Cancelled`].
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    /// A token that has not been cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation; every clone observes it.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// Return [`Error::Cancelled`] once cancellation was requested.
    pub fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Model operations used by the deterministic pipeline.
///
/// Backends bind the bundle's fixed prompt and pooled embeddings when they are
/// constructed; `denoise` receives only per-job inputs. Each method calls
/// `checkpoint` between blocks it can interrupt.
pub trait ModelBackend {
    /// Return the raw eight-channel posterior mean and log-variance for a
    /// normalized `[-1, 1]` image.
    fn encode_moments(&self, image: &Tensor, checkpoint: Checkpoint<'_>) -> Result<Tensor>;
    /// Return four denoising channels followed by the alpha logit.
    fn denoise(
        &self,
        input: &Tensor,
        timestep: usize,
        time_ids: &Tensor,
        checkpoint: Checkpoint<'_>,
    ) -> Result<Tensor>;
    /// Decode unscaled four-channel latents into raw RGB.
    fn decode(&self, latents: &Tensor, checkpoint: Checkpoint<'_>) -> Result<Tensor>;
}

/// Smallest accepted image height or width.
pub const MIN_IMAGE_AXIS: usize = 32;
/// Largest accepted image height or width.
pub const MAX_IMAGE_AXIS: usize = 1024;
/// Image height and width must be multiples of the VAE downsampling factor.
pub const IMAGE_AXIS_MULTIPLE: usize = 8;
/// Hard batch bound for every backend; backends may admit less.
pub const MAX_BATCH: usize = 4;
/// Hard bound on batch-inclusive image pixels; backends may admit less.
pub const MAX_PIXELS: usize = MAX_BATCH * MAX_IMAGE_AXIS * MAX_IMAGE_AXIS;

/// Whether an image axis is inside the shared admission contract.
pub(crate) fn valid_image_axis(axis: usize) -> bool {
    (MIN_IMAGE_AXIS..=MAX_IMAGE_AXIS).contains(&axis) && axis.is_multiple_of(IMAGE_AXIS_MULTIPLE)
}

/// Limits apply before model execution and include the batch in pixel counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceLimits {
    /// Largest accepted batch, in `1..=MAX_BATCH`.
    pub max_batch: usize,
    /// Largest accepted `batch * height * width`, in `1..=MAX_PIXELS`.
    pub max_pixels: usize,
}
impl Default for InferenceLimits {
    fn default() -> Self {
        Self {
            max_batch: MAX_BATCH,
            max_pixels: MAX_PIXELS,
        }
    }
}
impl InferenceLimits {
    /// Reject limits outside `1..=MAX_BATCH` and `1..=MAX_PIXELS`.
    pub fn validate(&self) -> Result<()> {
        if !(1..=MAX_BATCH).contains(&self.max_batch)
            || !(1..=MAX_PIXELS).contains(&self.max_pixels)
        {
            return Err(Error::InvalidInput(format!(
                "inference limits must have max_batch in 1..={MAX_BATCH} and max_pixels in 1..={MAX_PIXELS}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
/// Pinned recipe constants and admission limits for a [`Pipeline`].
pub struct PipelineConfig {
    /// DDPM timestep used for noising and prediction (400 for Phase II).
    pub timestep: usize,
    /// Latent scaling factor applied after posterior sampling.
    pub vae_scaling_factor: f64,
    /// Job admission limits.
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

/// One inference job. All tensors must be on the model device and satisfy
/// Candle's storage/layout invariants. Callers using `Tensor::from_storage`
/// must ensure its shape is compatible with the logical storage extent.
pub struct InferenceInput<'a> {
    /// RGB `[B, 3, H, W]` F32 in `[0, 1]`.
    pub image: &'a Tensor,
    /// Removal mask `[B, 1, H, W]` F32 in `[0, 1]`; white means remove.
    pub mask: &'a Tensor,
    /// Standard-normal VAE posterior noise `[B, 4, H/8, W/8]` in the model dtype.
    pub posterior_noise: &'a Tensor,
    /// Standard-normal diffusion noise `[B, 4, H/8, W/8]` in the model dtype.
    pub diffusion_noise: &'a Tensor,
}
/// Output of one inference job.
pub struct InferenceResult {
    /// Full decoded RGB `[B, 3, H, W]` F32 in `[0, 1]`, alpha-blended in latent space.
    pub image: Tensor,
    /// Predicted edit alpha upsampled to `[B, 1, H, W]` F32.
    pub alpha: Tensor,
    /// Alpha `[B, 1, H/8, W/8]` used for latent blending.
    pub latent_alpha: Tensor,
}

/// Reference stages, reported to observers in this order, synchronously,
/// without retaining tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Stage {
    /// Raw eight-channel VAE encoder output.
    PosteriorMoments,
    /// First four posterior channels.
    PosteriorMean,
    /// Last four posterior channels, clamped to `[-30, 20]`.
    PosteriorLogvar,
    /// Mean plus standard deviation times posterior noise.
    PosteriorSample,
    /// Posterior sample times the VAE scaling factor.
    OriginalLatent,
    /// Mask max-pooled by 8.
    PooledMask,
    /// Original latent noised at the pinned timestep.
    NoisyLatent,
    /// Nine-channel noisy latent, pooled mask and original latent.
    UnetInput,
    /// Fixed prompt embeddings, repeated per batch entry.
    Sequence,
    /// Fixed pooled prompt embedding, repeated per batch entry.
    Pooled,
    /// SDXL time IDs `[H, W, 0, 0, H, W]`.
    TimeIds,
    /// Four denoising channels of the UNet output.
    DenoisingOutput,
    /// Fifth UNet output channel.
    AlphaLogits,
    /// Sigmoid of the alpha logits.
    LatentAlpha,
    /// Scheduler predicted original sample.
    PredictedOriginal,
    /// Alpha blend of predicted and original latents.
    BlendedLatent,
    /// Decoded RGB mapped to `[0, 1]`.
    DecodedImage,
    /// Alpha upsampled to image resolution.
    Alpha,
}

impl Stage {
    /// Every stage in observer order.
    pub const ALL: [Stage; 18] = [
        Self::PosteriorMoments,
        Self::PosteriorMean,
        Self::PosteriorLogvar,
        Self::PosteriorSample,
        Self::OriginalLatent,
        Self::PooledMask,
        Self::NoisyLatent,
        Self::UnetInput,
        Self::Sequence,
        Self::Pooled,
        Self::TimeIds,
        Self::DenoisingOutput,
        Self::AlphaLogits,
        Self::LatentAlpha,
        Self::PredictedOriginal,
        Self::BlendedLatent,
        Self::DecodedImage,
        Self::Alpha,
    ];

    /// The pinned stage name used in captures, references and tolerances.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PosteriorMoments => "posterior_moments",
            Self::PosteriorMean => "posterior_mean",
            Self::PosteriorLogvar => "posterior_logvar",
            Self::PosteriorSample => "posterior_sample",
            Self::OriginalLatent => "original_latent",
            Self::PooledMask => "pooled_mask",
            Self::NoisyLatent => "noisy_latent",
            Self::UnetInput => "unet_input",
            Self::Sequence => "sequence",
            Self::Pooled => "pooled",
            Self::TimeIds => "time_ids",
            Self::DenoisingOutput => "denoising_output",
            Self::AlphaLogits => "alpha_logits",
            Self::LatentAlpha => "latent_alpha",
            Self::PredictedOriginal => "predicted_original",
            Self::BlendedLatent => "blended_latent",
            Self::DecodedImage => "decoded_image",
            Self::Alpha => "alpha",
        }
    }
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Stage names in observer order; identical to `Stage::ALL` names.
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

/// Backend-independent explicit-noise OSOR pipeline.
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
    validate_tensor_with_readback(
        t,
        name,
        shape,
        dtype,
        device,
        range,
        ValidationReadback {
            reduce_on_device: !device.is_cpu(),
            read: |values: &Tensor| Ok(values.flatten_all()?.to_vec1::<f32>()?),
        },
    )
}

struct ValidationReadback<F> {
    reduce_on_device: bool,
    read: F,
}

fn validate_backing_storage(tensor: &Tensor) -> Result<()> {
    let invalid = || Error::InvalidInput("tensor layout exceeds its backing storage".into());
    let (storage, layout) = tensor.storage_and_layout();
    let elements = match &*storage {
        candle_core::Storage::Cpu(storage) => match storage {
            candle_core::CpuStorage::F32(values) => values.len(),
            candle_core::CpuStorage::F64(values) => values.len(),
            candle_core::CpuStorage::F16(values) => values.len(),
            candle_core::CpuStorage::BF16(values) => values.len(),
            _ => return Err(invalid()),
        },
        candle_core::Storage::Cuda(storage) => {
            #[cfg(feature = "cuda")]
            {
                use candle_core::cuda_backend::CudaStorageSlice;
                match &storage.slice {
                    CudaStorageSlice::F32(values) => values.len(),
                    CudaStorageSlice::F64(values) => values.len(),
                    CudaStorageSlice::F16(values) => values.len(),
                    CudaStorageSlice::BF16(values) => values.len(),
                    _ => return Err(invalid()),
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = storage;
                return Err(invalid());
            }
        }
        candle_core::Storage::Metal(storage) => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                // Candle exposes Metal's physical capacity, not its private
                // logical element count. Its from_storage precondition still
                // applies when a pooled buffer has unused padding.
                storage.buffer().length() / tensor.dtype().size_in_bytes()
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                let _ = storage;
                return Err(invalid());
            }
        }
    };
    if layout.dims().len() != layout.stride().len() {
        return Err(invalid());
    }
    let end = if layout.dims().contains(&0) {
        Some(layout.start_offset())
    } else {
        layout
            .dims()
            .iter()
            .zip(layout.stride())
            .try_fold(layout.start_offset(), |offset, (&dim, &stride)| {
                offset.checked_add((dim - 1).checked_mul(stride)?)
            })
            .and_then(|last| last.checked_add(1))
    };
    if end.is_none_or(|end| end > elements) {
        return Err(invalid());
    }
    Ok(())
}

fn validate_tensor_with_readback(
    t: &Tensor,
    name: &str,
    shape: &[usize],
    dtype: DType,
    device: &candle_core::Device,
    range: Option<(f32, f32)>,
    mut readback: ValidationReadback<impl FnMut(&Tensor) -> Result<Vec<f32>>>,
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
    validate_backing_storage(t)?;
    // Preserve the original F64-through-F32 and half-to-F32 validation rules.
    // Tensor sizes are bounded by the caller before this conversion.
    let values = t.to_dtype(DType::F32)?;
    let invalid = if readback.reduce_on_device && values.elem_count() != 0 {
        // Intersect finite F32 bounds with the optional range. Comparisons
        // against NaN range endpoints in the original scalar check are false.
        let (lo, hi) = range.unwrap_or((f32::NEG_INFINITY, f32::INFINITY));
        let lo = if lo > -f32::MAX { lo } else { -f32::MAX };
        let hi = if hi < f32::MAX { hi } else { f32::MAX };
        let valid = validation_flags(&values, lo, hi)?;
        // Min is over integer flags, so NaNs cannot be hidden by reduction
        // order. The cast and readback operate on a newly allocated scalar.
        let scalar = valid.min_all()?.to_dtype(DType::F32)?;
        (readback.read)(&scalar)?.as_slice() != [1.0]
    } else if readback.reduce_on_device {
        false
    } else {
        (readback.read)(&values)?
            .iter()
            .any(|v| !v.is_finite() || range.is_some_and(|(lo, hi)| *v < lo || *v > hi))
    };
    if invalid {
        return Err(Error::InvalidInput(format!(
            "{name} contains non-finite or out-of-range values"
        )));
    }
    Ok(())
}

fn validation_flags(values: &Tensor, lo: f32, hi: f32) -> Result<Tensor> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if values.device().is_metal() {
        return Ok(values
            .contiguous()?
            .apply_op1_no_bwd(&metal_validation::Flags {
                lo: lo.to_bits(),
                hi: hi.to_bits(),
            })?);
    }
    // Candle's CUDA kernels use ordered comparisons without fast-math. The
    // CPU also exercises this path in tests through the private dispatch seam.
    Ok(values.ge(lo)?.mul(&values.le(hi)?)?)
}

// Candle compiles its Metal comparison kernels with fast math by default.
// Classify F32 bit patterns and compare monotonic integer keys instead, so
// NaNs, infinities, signed zero and subnormals do not depend on that mode.
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_validation {
    use std::sync::{Arc, Mutex};

    use candle_core::{
        CpuStorage, CustomOp1, DType, Layout, MetalStorage, Result, Shape, backend::BackendStorage,
    };
    use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline, Device};
    use objc2_metal::MTLSize;

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Params { uint count, lo, hi; };
inline uint order_key(uint bits) {
    if ((bits & 0x7fffffffu) == 0u) bits = 0u;
    return (bits & 0x80000000u) != 0u ? ~bits : bits ^ 0x80000000u;
}
kernel void osor_validation_flags(
    device const uint* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.count) return;
    uint bits = input[index];
    uint key = order_key(bits);
    bool finite = (bits & 0x7fffffffu) < 0x7f800000u;
    output[index] = uchar(finite && key >= order_key(p.lo) && key <= order_key(p.hi));
}
"#;

    #[repr(C)]
    struct Params {
        count: u32,
        lo: u32,
        hi: u32,
    }

    pub(super) struct Flags {
        pub(super) lo: u32,
        pub(super) hi: u32,
    }

    static PIPELINES: Mutex<Vec<(u64, Arc<ComputePipeline>)>> = Mutex::new(Vec::new());

    fn kernel_error(error: impl std::fmt::Display) -> candle_core::Error {
        candle_core::Error::Msg(format!("validation kernel: {error}"))
    }

    fn pipeline(device: &Device) -> Result<Arc<ComputePipeline>> {
        let mut cached = PIPELINES
            .lock()
            .map_err(|_| candle_core::Error::Msg("validation pipeline cache poisoned".into()))?;
        let id = device.registry_id();
        if let Some((_, pipeline)) = cached.iter().find(|(key, _)| *key == id) {
            return Ok(Arc::clone(pipeline));
        }
        let library = device
            .new_library_with_source(SOURCE, None)
            .map_err(kernel_error)?;
        let function = library
            .get_function("osor_validation_flags", None)
            .map_err(kernel_error)?;
        let pipeline = Arc::new(
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(kernel_error)?,
        );
        cached.push((id, Arc::clone(&pipeline)));
        Ok(pipeline)
    }

    impl CustomOp1 for Flags {
        fn name(&self) -> &'static str {
            "osor-validation-flags"
        }

        fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("validation flags kernel requires Metal")
        }

        fn metal_fwd(
            &self,
            storage: &MetalStorage,
            layout: &Layout,
        ) -> Result<(MetalStorage, Shape)> {
            if storage.dtype() != DType::F32 || !layout.is_contiguous() {
                candle_core::bail!("Metal validation requires contiguous F32 input")
            }
            let count = layout.shape().elem_count();
            let count_u32 = u32::try_from(count).map_err(|_| {
                candle_core::Error::Msg("validation tensor exceeds Metal index range".into())
            })?;
            let offset = layout
                .start_offset()
                .checked_mul(DType::F32.size_in_bytes())
                .ok_or_else(|| candle_core::Error::Msg("validation offset overflow".into()))?;
            let end = count
                .checked_mul(DType::F32.size_in_bytes())
                .and_then(|bytes| offset.checked_add(bytes))
                .ok_or_else(|| candle_core::Error::Msg("validation extent overflow".into()))?;
            if end > storage.buffer().length() {
                candle_core::bail!("validation layout exceeds its Metal buffer")
            }
            let device = storage.device();
            let output = device.new_buffer(count, DType::U8, "osor-validation-flags")?;
            let params = Params {
                count: count_u32,
                lo: self.lo,
                hi: self.hi,
            };
            let pipeline = pipeline(device.metal_device())?;
            let guard = device.command_encoder()?;
            let encoder: &ComputeCommandEncoder = guard.as_ref();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_input_buffer(0, Some(storage.buffer()), offset);
            encoder.set_output_buffer(1, Some(output.as_ref()), 0);
            encoder.set_bytes(2, &params);
            let group = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
            encoder.dispatch_threads(
                MTLSize {
                    width: count,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: group.min(count),
                    height: 1,
                    depth: 1,
                },
            );
            drop(guard);
            Ok((
                MetalStorage::new(output, device.clone(), count, DType::U8),
                layout.shape().clone(),
            ))
        }
    }
}

impl<B: ModelBackend> Pipeline<B> {
    /// Effective admission limits after backend-specific bounds.
    pub fn limits(&self) -> InferenceLimits {
        self.config.limits
    }

    /// Validate the recipe constants, limits and fixed prompt embeddings.
    pub fn new(
        backend: B,
        scheduler: Scheduler,
        sequence: Tensor,
        pooled: Tensor,
        config: PipelineConfig,
    ) -> Result<Self> {
        config.limits.validate()?;
        if !config.vae_scaling_factor.is_finite()
            || config.vae_scaling_factor <= 0.0
            || config.timestep >= scheduler.config().num_train_timesteps
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

    /// Run one job without observing stages.
    pub fn infer(&self, input: InferenceInput<'_>) -> Result<InferenceResult> {
        self.infer_with_observer(input, &mut |_, _| Ok(()))
    }

    /// Run one job, reporting each [`Stage`] in order. Return
    /// [`Error::Cancelled`] from `observer` to stop between stages.
    pub fn infer_with_observer(
        &self,
        input: InferenceInput<'_>,
        observer: &mut dyn FnMut(Stage, &Tensor) -> Result<()>,
    ) -> Result<InferenceResult> {
        self.run(input, observer, &|| Ok(()))
    }

    /// Run one job that stops with [`Error::Cancelled`] at the next stage or
    /// backend block checkpoint after `cancellation` is cancelled.
    pub fn infer_with_cancellation(
        &self,
        input: InferenceInput<'_>,
        cancellation: &CancellationToken,
        observer: &mut dyn FnMut(Stage, &Tensor) -> Result<()>,
    ) -> Result<InferenceResult> {
        let checkpoint = || cancellation.check();
        self.run(
            input,
            &mut |stage, tensor| {
                cancellation.check()?;
                observer(stage, tensor)
            },
            &checkpoint,
        )
    }

    fn run(
        &self,
        input: InferenceInput<'_>,
        observer: &mut dyn FnMut(Stage, &Tensor) -> Result<()>,
        checkpoint: Checkpoint<'_>,
    ) -> Result<InferenceResult> {
        checkpoint()?;
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
            || !valid_image_axis(height)
            || !valid_image_axis(width)
        {
            return Err(Error::InvalidInput(format!(
                "image dimensions or batch exceed limits; H and W must be multiples of {IMAGE_AXIS_MULTIPLE} in {MIN_IMAGE_AXIS}..={MAX_IMAGE_AXIS}"
            )));
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
        let moments = self.backend.encode_moments(&normalized, checkpoint)?;
        validate_tensor(
            &moments,
            "posterior moments",
            &[batch, 8, height / 8, width / 8],
            dtype,
            device,
            None,
        )?;
        observer(Stage::PosteriorMoments, &moments)?;
        let mean = moments.narrow(1, 0, 4)?;
        let logvar = moments.narrow(1, 4, 4)?.clamp(-30.0, 20.0)?;
        observer(Stage::PosteriorMean, &mean)?;
        observer(Stage::PosteriorLogvar, &logvar)?;
        let sample = (&mean + &(logvar.affine(0.5, 0.0)?.exp()? * input.posterior_noise)?)?;
        validate_tensor(
            &sample,
            "posterior sample",
            &latent_shape,
            dtype,
            device,
            None,
        )?;
        observer(Stage::PosteriorSample, &sample)?;
        let original = sample.affine(self.config.vae_scaling_factor, 0.0)?;
        validate_tensor(
            &original,
            "original latent",
            &latent_shape,
            dtype,
            device,
            None,
        )?;
        observer(Stage::OriginalLatent, &original)?;
        let mask = input.mask.max_pool2d(8)?.to_dtype(dtype)?;
        observer(Stage::PooledMask, &mask)?;
        // Both operands were checked finite above; skip the scheduler's readback.
        let noisy = self.scheduler.add_noise_prechecked(
            &original,
            input.diffusion_noise,
            self.config.timestep,
        )?;
        validate_tensor(&noisy, "noisy latent", &latent_shape, dtype, device, None)?;
        observer(Stage::NoisyLatent, &noisy)?;
        let unet_input = Tensor::cat(&[&noisy, &mask, &original], 1)?;
        observer(Stage::UnetInput, &unet_input)?;
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
        observer(Stage::Sequence, &sequence)?;
        observer(Stage::Pooled, &pooled)?;
        observer(Stage::TimeIds, &ids)?;
        let output = self
            .backend
            .denoise(&unet_input, self.config.timestep, &ids, checkpoint)?;
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
        observer(Stage::DenoisingOutput, &denoising)?;
        observer(Stage::AlphaLogits, &logits)?;
        let latent_alpha = candle_nn::ops::sigmoid(&logits)?;
        observer(Stage::LatentAlpha, &latent_alpha)?;
        // `noisy` and the full denoiser output were checked finite above.
        let predicted = self.scheduler.predicted_original_sample_prechecked(
            &noisy,
            &denoising,
            self.config.timestep,
        )?;
        validate_tensor(
            &predicted,
            "predicted original",
            &latent_shape,
            dtype,
            device,
            None,
        )?;
        observer(Stage::PredictedOriginal, &predicted)?;
        let blend = (latent_alpha.broadcast_mul(&predicted)?
            + latent_alpha.affine(-1.0, 1.0)?.broadcast_mul(&original)?)?;
        validate_tensor(&blend, "blended latent", &latent_shape, dtype, device, None)?;
        observer(Stage::BlendedLatent, &blend)?;
        let decoded = self.backend.decode(
            &blend.affine(1.0 / self.config.vae_scaling_factor, 0.0)?,
            checkpoint,
        )?;
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
        observer(Stage::DecodedImage, &image)?;
        let alpha = bilinear_8x(&latent_alpha.to_dtype(DType::F32)?, height, width)?;
        observer(Stage::Alpha, &alpha)?;
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

    #[test]
    fn validation_rejects_a_declared_shape_larger_than_storage() -> Result<()> {
        for storage in [
            candle_core::CpuStorage::F32(vec![0.5]),
            candle_core::CpuStorage::F64(vec![0.5]),
        ] {
            let malformed = Tensor::from_storage(
                candle_core::Storage::Cpu(storage),
                (2,),
                candle_core::op::BackpropOp::none(),
                false,
            );
            for reduce_on_device in [false, true] {
                let read = Cell::new(false);
                let result = validate_tensor_with_readback(
                    &malformed,
                    "test tensor",
                    &[2],
                    malformed.dtype(),
                    malformed.device(),
                    None,
                    ValidationReadback {
                        reduce_on_device,
                        read: |_values: &Tensor| {
                            read.set(true);
                            Ok(vec![0.5, 0.5])
                        },
                    },
                );
                assert!(
                    matches!(result, Err(Error::InvalidInput(message)) if message.contains("backing storage"))
                );
                assert!(
                    !read.get(),
                    "malformed storage must be rejected before readback"
                );
            }
        }
        Ok(())
    }

    #[derive(Default)]
    struct Synthetic {
        decoded: RefCell<Option<Tensor>>,
        time_ids: RefCell<Option<Tensor>>,
        encodes: Cell<usize>,
    }
    impl ModelBackend for Synthetic {
        fn encode_moments(&self, image: &Tensor, checkpoint: Checkpoint<'_>) -> Result<Tensor> {
            checkpoint()?;
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
            time_ids: &Tensor,
            _: Checkpoint<'_>,
        ) -> Result<Tensor> {
            assert_eq!(timestep, 0);
            *self.time_ids.borrow_mut() = Some(time_ids.clone());
            let (b, c, h, w) = input.dims4()?;
            assert_eq!(c, 9);
            let eps = Tensor::full(0.5f32, (b, 4, h, w), input.device())?;
            let logits = Tensor::zeros((b, 1, h, w), DType::F32, input.device())?;
            Ok(Tensor::cat(&[&eps, &logits], 1)?)
        }
        fn decode(&self, latents: &Tensor, _: Checkpoint<'_>) -> Result<Tensor> {
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

    fn scalar_oracle(values: &[f32], range: Option<(f32, f32)>) -> bool {
        values.iter().all(|value| {
            value.is_finite()
                && range.is_none_or(|(lo, _)| *value >= lo || lo.is_nan())
                && range.is_none_or(|(_, hi)| *value <= hi || hi.is_nan())
        })
    }

    fn assert_validation_matches_oracle(
        t: &Tensor,
        values: &[f32],
        range: Option<(f32, f32)>,
        label: &str,
    ) -> Result<()> {
        let expected = scalar_oracle(values, range);
        for reduced in [None, Some(false), Some(true)] {
            let result = if let Some(reduced) = reduced {
                validate_tensor_with_readback(
                    t,
                    "test tensor",
                    t.dims(),
                    t.dtype(),
                    t.device(),
                    range,
                    ValidationReadback {
                        reduce_on_device: reduced,
                        read: |values: &Tensor| Ok(values.flatten_all()?.to_vec1::<f32>()?),
                    },
                )
            } else {
                validate_tensor(t, "test tensor", t.dims(), t.dtype(), t.device(), range)
            };
            match result {
                Ok(()) => assert!(
                    expected,
                    "{label}, reduced={reduced:?}: accepted invalid values"
                ),
                Err(Error::InvalidInput(_)) => {
                    assert!(
                        !expected,
                        "{label}, reduced={reduced:?}: rejected valid values"
                    )
                }
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    // Every value used here is exactly representable by both F16 and BF16.
    // The expected verdict is computed from these independent scalar values.
    fn exercise_validation_cases(device: &Device, dtype: DType) -> Result<()> {
        let min_half_subnormal = 2f32.powi(-24);
        let cases = [
            (
                "endpoints and signed zero",
                vec![0., -0., 0.5, 1.],
                Some((0., 1.)),
            ),
            ("below range", vec![-0.125, 0., 1.], Some((0., 1.))),
            ("above range", vec![0., 1., 1. + 1. / 128.], Some((0., 1.))),
            (
                "subnormals",
                vec![-min_half_subnormal, min_half_subnormal],
                None,
            ),
            (
                "negative subnormal below zero",
                vec![-min_half_subnormal, -0.],
                Some((0., 1.)),
            ),
            ("signed zero endpoints", vec![-0., 0.], Some((-0., 0.))),
            ("positive infinity", vec![0., f32::INFINITY, 1.], None),
            ("negative infinity", vec![f32::NEG_INFINITY, 0., 1.], None),
            (
                "both infinities",
                vec![f32::INFINITY, f32::NEG_INFINITY],
                Some((0., 1.)),
            ),
            ("nan lower bound", vec![-0.125, 0.5], Some((f32::NAN, 1.))),
            ("nan upper bound", vec![0.5, 2.], Some((0., f32::NAN))),
            (
                "infinite bounds",
                vec![-1., 1.],
                Some((f32::NEG_INFINITY, f32::INFINITY)),
            ),
            (
                "positive infinite lower bound",
                vec![0.5],
                Some((f32::INFINITY, f32::INFINITY)),
            ),
            (
                "negative infinite upper bound",
                vec![0.5],
                Some((f32::NEG_INFINITY, f32::NEG_INFINITY)),
            ),
            ("inverted bounds", vec![0.5], Some((1., 0.))),
        ];
        for (label, values, range) in cases {
            let t = Tensor::from_vec(values.clone(), values.len(), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(device)?;
            assert_validation_matches_oracle(&t, &values, range, label)?;
        }

        if dtype == DType::F32 {
            for (values, range) in [
                (vec![-f32::from_bits(1), 0.], Some((0., 1.))),
                (vec![f32::from_bits(1), 0.], Some((0., 0.))),
                (vec![1.0f32.next_down(), 1.], Some((0., 1.))),
                (vec![1.0f32.next_up(), 1.], Some((0., 1.))),
                (vec![f32::from_bits(0xffc0_0001), 0.5], None),
                (vec![f32::from_bits(0x7f80_0001), 0.5], None),
            ] {
                let t = Tensor::from_vec(values.clone(), values.len(), &Device::Cpu)?
                    .to_device(device)?;
                assert_validation_matches_oracle(&t, &values, range, "f32 bits")?;
            }
        }

        // Cover several reduction groups and both ends; invalid padding must
        // not be lost to the ordering of a floating-point min/max reduction.
        for position in [0, 1, 63, 127, 128, 255, 256, 1023, 4096] {
            let mut values = vec![0.5f32; 4097];
            if let Some(value) = values.get_mut(position) {
                *value = f32::NAN;
            }
            let t = Tensor::from_vec(values.clone(), values.len(), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(device)?;
            assert_validation_matches_oracle(&t, &values, None, "nan position")?;
            assert_validation_matches_oracle(&t, &values, Some((0., 1.)), "ranged nan position")?;
        }

        // The first row is outside the view and must not affect validation.
        let base = Tensor::new(
            &[
                [f32::NAN, f32::INFINITY, -1., 2.],
                [0., 0.5, 1., 0.],
                [1., 0., 0.5, 1.],
            ],
            &Device::Cpu,
        )?
        .to_dtype(dtype)?
        .to_device(device)?;
        let view = base.narrow(0, 1, 2)?.transpose(0, 1)?.narrow(0, 1, 2)?;
        assert!(!view.is_contiguous());
        assert_validation_matches_oracle(&view, &[0.5, 0., 1., 0.5], Some((0., 1.)), "view")?;
        let bad_view = base.transpose(0, 1)?.narrow(0, 0, 2)?;
        assert_validation_matches_oracle(
            &bad_view,
            &[f32::NAN, 0., 1., f32::INFINITY, 0.5, 0.],
            None,
            "invalid view",
        )?;
        let repeated = base
            .narrow(0, 1, 1)?
            .narrow(1, 1, 1)?
            .broadcast_as((3, 5))?;
        assert_validation_matches_oracle(&repeated, &[0.5; 15], Some((0., 1.)), "broadcast")?;
        Ok(())
    }

    #[test]
    fn validation_matches_scalar_oracle_for_float_dtypes_and_views() -> Result<()> {
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::F64] {
            exercise_validation_cases(&Device::Cpu, dtype)?;
        }
        exercise_f64_conversion(&Device::Cpu)?;

        for (dtype, source, converted) in [
            (DType::F16, 65_504f32, 65_504f32),
            (DType::F16, 65_536f32, f32::INFINITY),
            (
                DType::BF16,
                f32::from_bits(0x7f7f_0000),
                f32::from_bits(0x7f7f_0000),
            ),
            (DType::BF16, f32::MAX, f32::INFINITY),
            (DType::F32, f32::MAX, f32::MAX),
        ] {
            let t = Tensor::new(&[source], &Device::Cpu)?.to_dtype(dtype)?;
            assert_validation_matches_oracle(&t, &[converted], None, "dtype overflow")?;
        }
        Ok(())
    }

    fn exercise_f64_conversion(device: &Device) -> Result<()> {
        // F64 remains judged after conversion to F32, including overflow and
        // values rounded onto a range endpoint.
        for values in [
            vec![0.0, 1.0 + f64::EPSILON],
            vec![-f64::MIN_POSITIVE, 0.5],
            vec![f32::MAX as f64, -(f32::MAX as f64)],
            vec![f64::MAX, 0.5],
            vec![-f64::MAX, 0.5],
            vec![f64::NAN, 0.5],
        ] {
            let expected: Vec<f32> = values.iter().map(|value| *value as f32).collect();
            let t = Tensor::from_vec(values, expected.len(), &Device::Cpu)?.to_device(device)?;
            for range in [None, Some((0., 1.))] {
                assert_validation_matches_oracle(&t, &expected, range, "f64 conversion")?;
            }
        }
        Ok(())
    }

    #[test]
    fn validation_checks_metadata_and_empty_tensors() -> Result<()> {
        let t = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
        for (shape, dtype) in [(&[3, 2][..], DType::F32), (&[2, 3][..], DType::F64)] {
            assert!(matches!(
                validate_tensor(&t, "test tensor", shape, dtype, &Device::Cpu, None),
                Err(Error::InvalidInput(_))
            ));
        }
        let integer = Tensor::zeros((2, 3), DType::U8, &Device::Cpu)?;
        assert!(matches!(
            validate_tensor(
                &integer,
                "test tensor",
                &[2, 3],
                DType::U8,
                &Device::Cpu,
                None
            ),
            Err(Error::InvalidInput(_))
        ));
        let empty = Tensor::zeros(0, DType::F32, &Device::Cpu)?;
        assert_validation_matches_oracle(&empty, &[], Some((0., 1.)), "empty")?;
        let scalar = Tensor::new(1f32, &Device::Cpu)?;
        assert_validation_matches_oracle(&scalar, &[1.], Some((0., 1.)), "scalar")?;
        Ok(())
    }

    #[test]
    fn reduced_validation_reads_only_fresh_scalar_storage() -> Result<()> {
        for count in [4, 64, 4097] {
            let t = Tensor::full(0.5f32, (count, 2), &Device::Cpu)?.transpose(0, 1)?;
            let calls = Cell::new(0);
            validate_tensor_with_readback(
                &t,
                "test tensor",
                t.dims(),
                t.dtype(),
                t.device(),
                Some((0., 1.)),
                ValidationReadback {
                    reduce_on_device: true,
                    read: |value: &Tensor| {
                        calls.set(calls.get() + 1);
                        assert_eq!(value.rank(), 0, "readback must receive a scalar reduction");
                        let (storage, _) = value.storage_and_layout();
                        if let candle_core::Storage::Cpu(cpu) = &*storage {
                            assert_eq!(
                                cpu.as_slice::<f32>()?.len(),
                                1,
                                "scalar must own scalar storage"
                            );
                        }
                        Ok(value.flatten_all()?.to_vec1::<f32>()?)
                    },
                },
            )?;
            assert_eq!(calls.get(), 1);
        }
        Ok(())
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a real Metal device"]
    fn metal_validation_matches_scalar_oracle() -> Result<()> {
        let device = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16] {
            exercise_validation_cases(&device, dtype)?;
            // Inspect malformed metadata only; never dispatch a kernel or
            // conversion against the deliberately undersized backing buffer.
            let base = Tensor::zeros(1, dtype, &device)?;
            let (storage, _) = base.storage_and_layout();
            let candle_core::Storage::Metal(storage) = &*storage else {
                return Err(Error::InvalidInput("expected Metal test storage".into()));
            };
            let count = (storage.buffer().length() / dtype.size_in_bytes())
                .checked_add(1)
                .ok_or_else(|| Error::InvalidInput("test storage overflow".into()))?;
            let malformed = Tensor::from_storage(
                candle_core::Storage::Metal(storage.clone()),
                (count,),
                candle_core::op::BackpropOp::none(),
                false,
            );
            assert!(matches!(
                validate_backing_storage(&malformed),
                Err(Error::InvalidInput(_))
            ));
        }
        let t = Tensor::zeros(2, DType::F32, &device)?;
        assert!(matches!(
            validate_tensor(&t, "test tensor", &[2], DType::F32, &Device::Cpu, None),
            Err(Error::InvalidInput(_))
        ));
        Ok(())
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device with BF16 support"]
    fn metal_bf16_validation_matches_scalar_oracle() -> Result<()> {
        exercise_validation_cases(&Device::new_metal(0)?, DType::BF16)
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA device with F16 and BF16 support"]
    fn cuda_validation_matches_scalar_oracle() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16, DType::F64] {
            exercise_validation_cases(&device, dtype)?;
        }
        exercise_f64_conversion(&device)
    }

    #[test]
    fn half_precision_decoder_is_mapped_and_interpolated_in_f32() -> Result<()> {
        struct Half;
        impl ModelBackend for Half {
            fn encode_moments(&self, image: &Tensor, _: Checkpoint<'_>) -> Result<Tensor> {
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
                _: Checkpoint<'_>,
            ) -> Result<Tensor> {
                let (b, _, h, w) = input.dims4()?;
                Ok(Tensor::zeros((b, 5, h, w), DType::BF16, input.device())?)
            }
            fn decode(&self, input: &Tensor, _: Checkpoint<'_>) -> Result<Tensor> {
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
        let mut conditioning = Vec::new();
        let result = p.infer_with_observer(
            InferenceInput {
                image: &image,
                mask: &mask,
                posterior_noise: &posterior,
                diffusion_noise: &diffusion,
            },
            &mut |stage, t| {
                stages.push(stage);
                if matches!(stage, Stage::Sequence | Stage::Pooled) {
                    conditioning.push(t.dims().to_vec());
                }
                match stage {
                    Stage::PosteriorSample => close(first(t)?, 3.0),
                    Stage::PooledMask => close(first(t)?, 1.0),
                    Stage::NoisyLatent => close(first(t)?, 3.6),
                    Stage::PredictedOriginal => close(first(t)?, 4.125),
                    Stage::BlendedLatent => close(first(t)?, 3.5625),
                    _ => {}
                }
                Ok(())
            },
        )?;
        assert_eq!(stages, Stage::ALL);
        assert!(
            Stage::ALL
                .iter()
                .map(|stage| stage.as_str())
                .eq(STAGES.iter().copied())
        );
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
        assert_eq!(conditioning, vec![vec![2, 77, 2048], vec![2, 1280]]);
        let time_ids = p.backend.time_ids.borrow();
        let ids = time_ids
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("time IDs missing".into()))?;
        assert_eq!(
            ids.to_vec2::<f32>()?,
            vec![vec![32., 32., 0., 0., 32., 32.]; 2]
        );
        drop(time_ids);
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
    fn cancellation_token_stops_before_and_between_stages() -> Result<()> {
        let p = pipeline()?;
        let image = Tensor::zeros((1, 3, 32, 32), DType::F32, &Device::Cpu)?;
        let mask = Tensor::zeros((1, 1, 32, 32), DType::F32, &Device::Cpu)?;
        let noise = Tensor::zeros((1, 4, 4, 4), DType::F32, &Device::Cpu)?;
        let input = || InferenceInput {
            image: &image,
            mask: &mask,
            posterior_noise: &noise,
            diffusion_noise: &noise,
        };
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert!(matches!(
            p.infer_with_cancellation(input(), &cancelled, &mut |_, _| Ok(())),
            Err(Error::Cancelled)
        ));
        assert_eq!(p.backend.encodes.get(), 0);
        // Cancelling from another holder during the job stops at the next stage.
        let token = CancellationToken::new();
        let remote = token.clone();
        let mut seen = Vec::new();
        let result = p.infer_with_cancellation(input(), &token, &mut |stage, _| {
            seen.push(stage);
            if stage == Stage::PooledMask {
                remote.cancel();
            }
            Ok(())
        });
        assert!(matches!(result, Err(Error::Cancelled)));
        assert_eq!(seen.last(), Some(&Stage::PooledMask));
        assert!(p.backend.time_ids.borrow().is_none());
        // An uncancelled token completes normally.
        p.infer_with_cancellation(input(), &CancellationToken::new(), &mut |_, _| Ok(()))?;
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
        let time_ids = p.backend.time_ids.borrow();
        let ids = time_ids
            .as_ref()
            .ok_or_else(|| Error::InvalidInput("time IDs missing".into()))?;
        assert_eq!(
            ids.to_vec2::<f32>()?,
            vec![vec![40., 56., 0., 0., 40., 56.]]
        );
        drop(time_ids);

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
