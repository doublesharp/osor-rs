//! WGPU neural graph with CPU-staged pipeline inputs and outputs.
//!
//! Neural weights and intermediate activations stay on the selected GPU. The
//! surrounding explicit-noise pipeline remains on the CPU.
mod model;

use std::{
    any::Any,
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use burn::tensor::wgpu::{WgpuSetup, wgpu};
use burn::tensor::{Device, Tensor as BurnTensor, TensorData};
use candle_core::{DType, Device as CandleDevice, Tensor};

use crate::{
    Error, Result,
    pipeline::{Checkpoint, ModelBackend, valid_image_axis},
};

/// WGPU admission bound on image pixels; batch is limited to one.
pub(crate) const MAX_PIXELS: usize = 512 * 512;
// Larger than every tensor in the pinned architecture, while bounding a single
// conversion even if this interface is called outside the artifact loader.
const MAX_WEIGHT_ELEMENTS: usize = 64 * 1024 * 1024;
static DEVICE: Mutex<Option<Arc<DeviceGeneration>>> = Mutex::new(None);

struct DeviceGeneration {
    device: Device,
    raw: wgpu::Device,
    description: String,
    lost: Arc<AtomicBool>,
}

impl DeviceGeneration {
    fn ensure_available(&self) -> Result<()> {
        if self.lost.load(Ordering::Acquire) {
            return Err(Error::Unavailable(
                "WGPU device generation is lost; load a new model to retry".into(),
            ));
        }
        Ok(())
    }

    fn mark_lost(&self) {
        self.lost.store(true, Ordering::Release);
    }
}

fn guarded<T>(context: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        let detail = panic_message(&*payload);
        Error::Unavailable(format!("WGPU {context} failed: {detail}"))
    })?
}

fn panic_message(payload: &(dyn Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<String>() {
        message
    } else if let Some(message) = payload.downcast_ref::<&str>() {
        message
    } else {
        "dependency panicked"
    }
}

fn create_generation() -> Result<Arc<DeviceGeneration>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|e| Error::Unavailable(format!("WGPU adapter selection failed: {e}")))?;
    let info = adapter.get_info();
    if !matches!(
        info.device_type,
        wgpu::DeviceType::DiscreteGpu
            | wgpu::DeviceType::IntegratedGpu
            | wgpu::DeviceType::VirtualGpu
    ) {
        return Err(Error::Unavailable(format!(
            "WGPU selected a non-GPU or unclassified adapter: {} ({:?})",
            info.name, info.device_type
        )));
    }
    // The up-path ResNet convolution [1280, 2560, 3, 3] is larger
    // than the attention projection weights (all stored as F32).
    const MIN_WEIGHT_BYTES: u64 = 1280 * 2560 * 3 * 3 * 4;
    let limits = adapter.limits();
    if limits.max_storage_buffer_binding_size < MIN_WEIGHT_BYTES
        || limits.max_buffer_size < MIN_WEIGHT_BYTES
    {
        return Err(Error::Unavailable(format!(
            "WGPU adapter {} cannot bind the largest model weight (requires at least {MIN_WEIGHT_BYTES} bytes)",
            info.name
        )));
    }
    let features = adapter.features()
        - wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
        - wgpu::Features::all_experimental_mask();
    let (raw, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("OSOR WGPU"),
        required_features: features,
        required_limits: limits,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        ..Default::default()
    }))
    .map_err(|e| Error::Unavailable(format!("WGPU device request failed: {e}")))?;
    let setup = WgpuSetup {
        instance,
        adapter,
        device: raw.clone(),
        queue,
        backend: info.backend,
    };
    let device = Device::wgpu_options()
        .setup(setup)
        .init()
        .map_err(|e| Error::Unavailable(format!("WGPU runtime registration failed: {e}")))?;
    let description = format!("{} ({:?}, {:?})", info.name, info.device_type, info.backend);
    Ok(Arc::new(DeviceGeneration {
        device,
        raw,
        description,
        lost: Arc::new(AtomicBool::new(false)),
    }))
}

fn acquire_generation(ordinal: usize) -> Result<Arc<DeviceGeneration>> {
    if ordinal != 0 {
        return Err(Error::Unavailable(format!(
            "WGPU supports only the default GPU at ordinal 0; ordinal {ordinal} is unsupported"
        )));
    }
    let mut cached = DEVICE
        .lock()
        .map_err(|_| Error::Unavailable("WGPU device initialization lock poisoned".into()))?;
    if let Some(existing) = cached.as_ref() {
        let health = check_health(existing);
        if health.is_ok() {
            return Ok(Arc::clone(existing));
        }
        existing.mark_lost();
        // A failed generation must release GPU resources even while old graphs live.
        let _ = guarded("retiring device", || {
            existing.raw.destroy();
            Ok(())
        });
        *cached = None;
    }
    // Catch dependency unwinds before releasing the cache mutex, avoiding poisoning it.
    let generation = guarded("device initialization", create_generation)?;
    *cached = Some(Arc::clone(&generation));
    Ok(generation)
}

#[cfg(test)]
fn acquire_device(ordinal: usize) -> Result<(Device, String)> {
    let generation = acquire_generation(ordinal)?;
    Ok((generation.device.clone(), generation.description.clone()))
}

fn guarded_generation<T>(
    generation: &DeviceGeneration,
    context: &str,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    generation.ensure_available()?;
    let result = catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        generation.mark_lost();
        Error::Unavailable(format!(
            "WGPU {context} failed: {}",
            panic_message(&*payload)
        ))
    })?;
    generation.ensure_available()?;
    result
}

// CubeCL owns the one device-lost callback supported by wgpu. Its callback
// feeds sync errors; replacing it would disable Burn's poison detection.
fn check_health(generation: &DeviceGeneration) -> Result<()> {
    let health = guarded("device health check", || {
        generation.ensure_available()?;
        generation
            .raw
            .poll(wgpu::PollType::Poll)
            .map_err(|e| Error::Unavailable(format!("WGPU device poll failed: {e}")))?;
        generation
            .device
            .sync()
            .map_err(|e| Error::Unavailable(format!("WGPU device synchronization failed: {e}")))?;
        generation.ensure_available()
    });
    if health.is_err() {
        generation.mark_lost();
    }
    health
}

fn guarded_model<T>(
    generation: &DeviceGeneration,
    context: &str,
    f: impl FnOnce() -> Result<T>,
) -> Result<T> {
    check_health(generation)?;
    let result = guarded_generation(generation, context, f)?;
    check_health(generation)?;
    Ok(result)
}

fn checked_elements(shape: &[usize], max: usize) -> Option<usize> {
    shape.iter().try_fold(1usize, |count, &dim| {
        if dim == 0 {
            return None;
        }
        count.checked_mul(dim).filter(|&n| n <= max)
    })
}

fn cpu_values(tensor: &Tensor, name: &str, shape: &[usize], artifact: bool) -> Result<Vec<f32>> {
    let invalid = |message: String| {
        if artifact {
            Error::InvalidArtifact(message)
        } else {
            Error::InvalidInput(message)
        }
    };
    if tensor.dtype() != DType::F32 || !tensor.device().is_cpu() || tensor.dims() != shape {
        return Err(invalid(format!(
            "{name} must be a CPU F32 tensor with shape {shape:?}"
        )));
    }
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid(format!("{name} contains non-finite values")));
    }
    Ok(values)
}

/// Owns CPU checkpoint tensors until the graph consumes and uploads each weight.
pub struct Weights {
    tensors: HashMap<String, Tensor>,
    device: Device,
    generation: Option<Arc<DeviceGeneration>>,
}

impl Weights {
    #[cfg(test)]
    fn new(tensors: HashMap<String, Tensor>, device: &Device) -> Self {
        Self {
            tensors,
            device: device.clone(),
            generation: None,
        }
    }

    fn for_generation(
        tensors: HashMap<String, Tensor>,
        generation: &Arc<DeviceGeneration>,
    ) -> Self {
        Self {
            tensors,
            device: generation.device.clone(),
            generation: Some(Arc::clone(generation)),
        }
    }

    /// Consume one named F32 CPU tensor after validating its shape and bounds.
    pub fn take<const D: usize>(&mut self, name: &str, shape: [usize; D]) -> Result<BurnTensor<D>> {
        let generation = self.generation.clone();
        let upload_weight = || {
            if checked_elements(&shape, MAX_WEIGHT_ELEMENTS).is_none() {
                return Err(Error::InvalidArtifact(format!(
                    "weight {name} has invalid or oversized shape {shape:?}"
                )));
            }
            let tensor = self
                .tensors
                .get(name)
                .ok_or_else(|| Error::InvalidArtifact(format!("missing weight {name}")))?;
            let values = cpu_values(tensor, name, &shape, true)?;
            let tensor = self.tensors.remove(name);
            drop(tensor);
            Ok(BurnTensor::from_data(
                TensorData::new(values, shape),
                (&self.device, burn::tensor::DType::F32),
            ))
        };
        match generation {
            Some(generation) => guarded_generation(&generation, "weight upload", upload_weight),
            None => guarded("weight upload", upload_weight),
        }
    }
}

fn image_shape(tensor: &Tensor, channels: usize, latent: bool, name: &str) -> Result<[usize; 4]> {
    let (batch, actual_channels, height, width) = tensor
        .dims4()
        .map_err(|_| Error::InvalidInput(format!("{name} must be rank four")))?;
    let scale = if latent { 8 } else { 1 };
    let image_height = height.checked_mul(scale);
    let image_width = width.checked_mul(scale);
    let valid_axes = image_height.zip(image_width).is_some_and(|(h, w)| {
        valid_image_axis(h)
            && valid_image_axis(w)
            && h.checked_mul(w).is_some_and(|pixels| pixels <= MAX_PIXELS)
    });
    if batch != 1 || actual_channels != channels || !valid_axes {
        return Err(Error::InvalidInput(format!(
            "{name} requires batch 1, {channels} channels, image axes 32..=1024 in multiples of 8, and at most {MAX_PIXELS} pixels"
        )));
    }
    Ok([batch, actual_channels, height, width])
}

fn upload<const D: usize>(
    tensor: &Tensor,
    name: &str,
    shape: [usize; D],
    device: &Device,
) -> Result<BurnTensor<D>> {
    let values = cpu_values(tensor, name, &shape, false)?;
    Ok(BurnTensor::from_data(
        TensorData::new(values, shape),
        (device, burn::tensor::DType::F32),
    ))
}

#[cfg(test)]
fn readback<const D: usize>(
    tensor: BurnTensor<D>,
    shape: [usize; D],
    name: &str,
) -> Result<Tensor> {
    readback_impl(tensor, shape, name, None)
}

fn readback_impl<const D: usize>(
    tensor: BurnTensor<D>,
    shape: [usize; D],
    name: &str,
    generation: Option<&DeviceGeneration>,
) -> Result<Tensor> {
    if tensor.dims() != shape {
        return Err(Error::Unavailable(format!(
            "WGPU {name} produced shape {:?}, expected {shape:?}",
            tensor.dims()
        )));
    }
    let values = tensor
        .try_into_data()
        .map_err(|e| {
            if let Some(generation) = generation {
                generation.mark_lost();
            }
            Error::Unavailable(format!("WGPU {name} readback failed: {e}"))
        })?
        .try_into_vec::<f32>()
        .map_err(|e| Error::Unavailable(format!("WGPU {name} output is not F32: {e}")))?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::Unavailable(format!(
            "WGPU {name} produced non-finite values"
        )));
    }
    Ok(Tensor::from_vec(
        values,
        shape.as_slice(),
        &CandleDevice::Cpu,
    )?)
}

/// A model whose neural graph executes on a WGPU adapter.
pub struct WgpuModel {
    graph: model::Graph,
    generation: Arc<DeviceGeneration>,
}

impl WgpuModel {
    /// Load validated checkpoint tensors and the fixed CPU F32 prompt
    /// embeddings (`[1, 77, 2048]`, `[1, 1280]`), then wait for GPU uploads.
    pub fn new(
        unet: HashMap<String, Tensor>,
        vae: HashMap<String, Tensor>,
        sequence: &Tensor,
        pooled: &Tensor,
        ordinal: usize,
    ) -> Result<Self> {
        let generation = acquire_generation(ordinal)?;
        guarded_model(&generation, "model initialization", || {
            let sequence = upload(sequence, "sequence", [1, 77, 2048], &generation.device)?;
            let pooled = upload(pooled, "pooled", [1, 1280], &generation.device)?;
            let mut unet = Weights::for_generation(unet, &generation);
            let mut vae = Weights::for_generation(vae, &generation);
            let graph =
                model::Graph::new(&mut unet, &mut vae, &generation.device, &sequence, pooled)?;
            generation.device.sync().map_err(|e| {
                generation.mark_lost();
                Error::Unavailable(format!("WGPU weight upload synchronization failed: {e}"))
            })?;
            Ok(Self {
                graph,
                generation: Arc::clone(&generation),
            })
        })
    }

    /// Description reported by the selected WGPU adapter.
    pub fn device_description(&self) -> String {
        self.generation.description.clone()
    }
}

// The WGPU graph is queued as a whole, so checkpoints run once per call.
impl ModelBackend for WgpuModel {
    fn encode_moments(&self, image: &Tensor, checkpoint: Checkpoint<'_>) -> Result<Tensor> {
        checkpoint()?;
        guarded_model(&self.generation, "VAE encode", || {
            let [b, _, h, w] = image_shape(image, 3, false, "image")?;
            let values = cpu_values(image, "image", &[b, 3, h, w], false)?;
            if values.iter().any(|&v| !(-1.0..=1.0).contains(&v)) {
                return Err(Error::InvalidInput(
                    "normalized image values must be in [-1, 1]".into(),
                ));
            }
            let input = BurnTensor::from_data(
                TensorData::new(values, [b, 3, h, w]),
                (&self.generation.device, burn::tensor::DType::F32),
            );
            readback_impl(
                self.graph.encode_moments(input),
                [b, 8, h / 8, w / 8],
                "VAE moments",
                Some(&self.generation),
            )
        })
    }

    fn denoise(
        &self,
        input: &Tensor,
        timestep: usize,
        time_ids: &Tensor,
        checkpoint: Checkpoint<'_>,
    ) -> Result<Tensor> {
        checkpoint()?;
        guarded_model(&self.generation, "UNet", || {
            let [b, _, h, w] = image_shape(input, 9, true, "UNet input")?;
            if timestep >= 100_000 {
                return Err(Error::InvalidInput(
                    "timestep exceeds scheduler limits".into(),
                ));
            }
            let input = upload(input, "UNet input", [b, 9, h, w], &self.generation.device)?;
            let time_ids = upload(time_ids, "time_ids", [1, 6], &self.generation.device)?;
            readback_impl(
                self.graph.denoise(input, timestep, time_ids)?,
                [b, 5, h, w],
                "UNet output",
                Some(&self.generation),
            )
        })
    }

    fn decode(&self, latents: &Tensor, checkpoint: Checkpoint<'_>) -> Result<Tensor> {
        checkpoint()?;
        guarded_model(&self.generation, "VAE decode", || {
            let [b, _, h, w] = image_shape(latents, 4, true, "latents")?;
            let input = upload(latents, "latents", [b, 4, h, w], &self.generation.device)?;
            readback_impl(
                self.graph.decode(input),
                [b, 3, h * 8, w * 8],
                "VAE decoded image",
                Some(&self.generation),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimensions_allow_odd_latents_and_reject_unsafe_sizes() -> Result<()> {
        let odd = Tensor::zeros((1, 9, 5, 7), DType::F32, &CandleDevice::Cpu)?;
        assert_eq!(image_shape(&odd, 9, true, "input")?, [1, 9, 5, 7]);
        for shape in [(2, 9, 4, 4), (1, 9, 3, 4), (1, 9, 65, 65), (1, 9, 129, 4)] {
            let input = Tensor::zeros(shape, DType::F32, &CandleDevice::Cpu)?;
            assert!(matches!(
                image_shape(&input, 9, true, "input"),
                Err(Error::InvalidInput(_))
            ));
        }
        Ok(())
    }

    #[test]
    fn validation_rejects_nonfinite_dtype_and_weight_overflow() -> Result<()> {
        let nan = Tensor::full(f32::NAN, (1, 2), &CandleDevice::Cpu)?;
        let f64 = Tensor::zeros((1, 2), DType::F64, &CandleDevice::Cpu)?;
        assert!(matches!(
            cpu_values(&nan, "input", &[1, 2], false),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            cpu_values(&f64, "weight", &[1, 2], true),
            Err(Error::InvalidArtifact(_))
        ));
        assert_eq!(
            checked_elements(&[usize::MAX, 2], MAX_WEIGHT_ELEMENTS),
            None
        );
        assert_eq!(checked_elements(&[0, 2], MAX_WEIGHT_ELEMENTS), None);
        assert!(matches!(acquire_device(1), Err(Error::Unavailable(_))));
        Ok(())
    }

    #[test]
    #[ignore = "requires a real WGPU adapter"]
    fn gpu_roundtrip() -> Result<()> {
        guarded("roundtrip", || {
            let (device, _) = acquire_device(0)?;
            let source =
                Tensor::from_vec(vec![1f32, -2., 3., 4.], (1, 1, 2, 2), &CandleDevice::Cpu)?;
            let gpu = upload(&source, "roundtrip", [1, 1, 2, 2], &device)?;
            let result = readback(gpu, [1, 1, 2, 2], "roundtrip")?;
            assert_eq!(
                result.flatten_all()?.to_vec1::<f32>()?,
                vec![1., -2., 3., 4.]
            );
            Ok(())
        })
    }

    #[test]
    #[ignore = "requires a real WGPU adapter; run this test in a fresh process"]
    fn gpu_host_default_initialized_before_osor() -> Result<()> {
        guarded("host default coexistence", || {
            let host = Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
            let source =
                Tensor::from_vec(vec![1f32, -2., 3., 4.], (1, 1, 2, 2), &CandleDevice::Cpu)?;
            let host_tensor = upload(&source, "host", [1, 1, 2, 2], &host)?;
            assert_eq!(
                readback(host_tensor, [1, 1, 2, 2], "host")?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                vec![1., -2., 3., 4.]
            );
            let (device, _) = acquire_device(0)?;
            let tensor = upload(&source, "OSOR", [1, 1, 2, 2], &device)?;
            assert_eq!(
                readback(tensor, [1, 1, 2, 2], "OSOR")?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                vec![1., -2., 3., 4.]
            );
            Ok(())
        })
    }

    #[test]
    #[ignore = "requires a real WGPU adapter; destroys an OSOR device generation"]
    fn gpu_destroyed_generation_is_replaced_and_old_callback_is_fenced() -> Result<()> {
        guarded("generation replacement", || {
            let old = acquire_generation(0)?;
            old.raw.destroy();
            let _ = old.raw.poll(wgpu::PollType::Poll);
            let fresh = acquire_generation(0)?;
            assert!(!Arc::ptr_eq(&old, &fresh));
            assert!(matches!(old.ensure_available(), Err(Error::Unavailable(_))));
            // Dispatch any pending callback from the retired device after replacement.
            let _ = old.raw.poll(wgpu::PollType::Poll);
            fresh.ensure_available()?;
            let source =
                Tensor::from_vec(vec![1f32, -2., 3., 4.], (1, 1, 2, 2), &CandleDevice::Cpu)?;
            let gpu = upload(&source, "replacement", [1, 1, 2, 2], &fresh.device)?;
            let result = readback_impl(gpu, [1, 1, 2, 2], "replacement", Some(&fresh))?;
            assert_eq!(
                result.flatten_all()?.to_vec1::<f32>()?,
                vec![1., -2., 3., 4.]
            );
            assert!(Arc::ptr_eq(&fresh, &acquire_generation(0)?));
            Ok(())
        })
    }
}
