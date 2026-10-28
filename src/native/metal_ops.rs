//! F32 Metal operations for VAE tensors. All Metal encoding uses Candle's
//! safe wrappers; other devices and dtypes retain the existing tensor paths.

use std::sync::{Arc, Mutex};

use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, CustomOp3, DType, Layout, MetalStorage, Result, Shape,
    Tensor, backend::BackendStorage,
};
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline, Device};
use objc2_metal::{MTLCompileOptions, MTLSize};

const THREADS: usize = 256;
const CHUNK: usize = THREADS * 16;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

struct NormParams { uint count, chunks, groups, channels, spatial; };

// Subtract a per-group anchor before summing. This avoids discarding small
// deviations when a group has a large common offset. Variance is computed in
// a separate pass over those centered deviations, never from E[x*x]-E[x]^2.
kernel void norm_mean_part(device const float* x [[buffer(0)]],
                           device float* partial [[buffer(1)]],
                           constant NormParams& p [[buffer(2)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float sums[256];
    uint lane = tid.x;
    ulong base = ulong(group.y) * p.count;
    ulong start = ulong(group.x) * 4096 + lane;
    float anchor = x[base];
    float sum = 0.0f;
    for (uint j = 0; j < 16; ++j) {
        ulong i = start + ulong(j) * 256;
        if (i < p.count) sum += x[base + i] - anchor;
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) partial[ulong(group.y) * p.chunks + group.x] = sums[0];
}

kernel void norm_reduce(device const float* partial [[buffer(0)]],
                        device float* stats [[buffer(1)]],
                        constant NormParams& p [[buffer(2)]],
                        constant uint& variance [[buffer(3)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_position_in_threadgroup]]) {
    threadgroup float sums[256];
    float sum = 0.0f;
    for (ulong i = lane; i < p.chunks; i += 256)
        sum += partial[ulong(group) * p.chunks + i];
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
        float value = sums[0] / float(p.count);
        stats[group] = variance != 0 ? sqrt(value + 1.0e-6f) : value;
    }
}

kernel void norm_var_part(device const float* x [[buffer(0)]],
                          device const float* means [[buffer(1)]],
                          device float* partial [[buffer(2)]],
                          constant NormParams& p [[buffer(3)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float sums[256];
    uint lane = tid.x;
    ulong base = ulong(group.y) * p.count;
    ulong start = ulong(group.x) * 4096 + lane;
    float anchor = x[base];
    float mean = means[group.y];
    float sum = 0.0f;
    for (uint j = 0; j < 16; ++j) {
        ulong i = start + ulong(j) * 256;
        if (i < p.count) {
            float centered = (x[base + i] - anchor) - mean;
            sum += centered * centered;
        }
    }
    sums[lane] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) partial[ulong(group.y) * p.chunks + group.x] = sums[0];
}

kernel void norm_apply(device const float* x [[buffer(0)]],
                       device const float* means [[buffer(1)]],
                       device const float* denom [[buffer(2)]],
                       device const float* weight [[buffer(3)]],
                       device const float* bias [[buffer(4)]],
                       device float* out [[buffer(5)]],
                       constant NormParams& p [[buffer(6)]],
                       uint i [[thread_position_in_grid]]) {
    ulong count = ulong(p.groups) * p.count;
    if (ulong(i) >= count) return;
    uint group = i / p.count;
    uint channel = (i / p.spatial) % p.channels;
    float centered = (x[i] - x[ulong(group) * p.count]) - means[group];
    float value = (centered / denom[group]) * weight[channel] + bias[channel];
    out[i] = value;
}

struct UpsampleParams { uint width, height, elements; };
kernel void upsample_2x(device const float* x [[buffer(0)]],
                        device float2* out [[buffer(1)]],
                        constant UpsampleParams& p [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= p.elements) return;
    uint column = i % p.width;
    ulong row = ulong(i) / p.width;
    ulong out_row = (row / p.height) * (ulong(p.height) * 2) + (row % p.height) * 2;
    ulong dst = out_row * p.width + column;
    float value = x[i];
    out[dst] = float2(value);
    out[dst + p.width] = float2(value);
}

struct BiasParams { uint plane_vectors, channels, vectors; };
kernel void channel_bias(device const float4* x [[buffer(0)]],
                         device const float* bias [[buffer(1)]],
                         device float4* out [[buffer(2)]],
                         constant BiasParams& p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.vectors) return;
    uint channel = (i / p.plane_vectors) % p.channels;
    out[i] = x[i] + float4(bias[channel]);
}
"#;

#[repr(C)]
struct NormParams {
    count: u32,
    chunks: u32,
    groups: u32,
    channels: u32,
    spatial: u32,
}

#[repr(C)]
struct UpsampleParams {
    width: u32,
    height: u32,
    elements: u32,
}

#[repr(C)]
struct BiasParams {
    plane_vectors: u32,
    channels: u32,
    vectors: u32,
}

struct Pipelines {
    mean: ComputePipeline,
    reduce: ComputePipeline,
    variance: ComputePipeline,
    apply: ComputePipeline,
    upsample: ComputePipeline,
    bias: ComputePipeline,
}

static PIPELINES: Mutex<Vec<(u64, Arc<Pipelines>)>> = Mutex::new(Vec::new());

fn kernel_error(error: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("VAE Metal kernel: {error}"))
}

fn pipelines(device: &Device) -> Result<Arc<Pipelines>> {
    let mut cached = PIPELINES
        .lock()
        .map_err(|_| kernel_error("pipeline cache poisoned"))?;
    let id = device.registry_id();
    if let Some((_, pipelines)) = cached.iter().find(|(key, _)| *key == id) {
        return Ok(Arc::clone(pipelines));
    }
    let options = MTLCompileOptions::new();
    // This works on supported macOS versions, including those before mathMode.
    // Disable reassociation/fast approximations for the normalization recipe.
    #[allow(deprecated)]
    options.setFastMathEnabled(false);
    let library = device
        .new_library_with_source(SOURCE, Some(&options))
        .map_err(kernel_error)?;
    let compile = |name| -> Result<ComputePipeline> {
        let function = library.get_function(name, None).map_err(kernel_error)?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(kernel_error)
    };
    let result = Arc::new(Pipelines {
        mean: compile("norm_mean_part")?,
        reduce: compile("norm_reduce")?,
        variance: compile("norm_var_part")?,
        apply: compile("norm_apply")?,
        upsample: compile("upsample_2x")?,
        bias: compile("channel_bias")?,
    });
    if [&result.mean, &result.reduce, &result.variance]
        .iter()
        .any(|pipeline| pipeline.max_total_threads_per_threadgroup() < THREADS)
    {
        candle_core::bail!("VAE Metal reduction requires 256 threads per threadgroup")
    }
    cached.push((id, Arc::clone(&result)));
    Ok(result)
}

fn u32_of(value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| kernel_error("dimension exceeds u32"))
}

fn checked_elements(dims: &[usize]) -> Result<usize> {
    let count = dims
        .iter()
        .try_fold(1usize, |total, &dim| total.checked_mul(dim))
        .ok_or_else(|| kernel_error("tensor size overflows"))?;
    count
        .checked_mul(DType::F32.size_in_bytes())
        .ok_or_else(|| kernel_error("tensor byte size overflows"))?;
    Ok(count)
}

fn offset(layout: &Layout) -> Result<usize> {
    layout
        .start_offset()
        .checked_mul(DType::F32.size_in_bytes())
        .ok_or_else(|| kernel_error("tensor offset overflows"))
}

fn contiguous_f32(storage: &MetalStorage, layout: &Layout) -> Result<()> {
    if storage.dtype() != DType::F32 || !layout.is_contiguous() {
        candle_core::bail!("VAE Metal kernel requires contiguous F32 tensors")
    }
    Ok(())
}

fn threads(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

fn dispatch_linear(encoder: &ComputeCommandEncoder, pipeline: &ComputePipeline, count: usize) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.dispatch_threads(
        threads(count),
        threads(
            pipeline
                .max_total_threads_per_threadgroup()
                .clamp(1, THREADS),
        ),
    );
}

fn dispatch_reduction(
    encoder: &ComputeCommandEncoder,
    pipeline: &ComputePipeline,
    x: usize,
    y: usize,
) {
    encoder.set_compute_pipeline_state(pipeline);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: x,
            height: y,
            depth: 1,
        },
        threads(THREADS),
    );
}

struct GroupNorm;

impl CustomOp3 for GroupNorm {
    fn name(&self) -> &'static str {
        "osor-metal-vae-group-norm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("VAE Metal group normalization is unavailable on CPU")
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        il: &Layout,
        weight: &MetalStorage,
        wl: &Layout,
        bias: &MetalStorage,
        bl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        contiguous_f32(input, il)?;
        contiguous_f32(weight, wl)?;
        contiguous_f32(bias, bl)?;
        let (batch, channels, height, width) = il.shape().dims4()?;
        if batch == 0
            || channels == 0
            || !channels.is_multiple_of(32)
            || height == 0
            || width == 0
            || wl.shape().dims1()? != channels
            || bl.shape().dims1()? != channels
        {
            candle_core::bail!("invalid VAE Metal group normalization shape")
        }
        let spatial = checked_elements(&[height, width])?;
        let count = checked_elements(&[channels / 32, spatial])?;
        let groups = checked_elements(&[batch, 32])?;
        let total = checked_elements(&[groups, count])?;
        let chunks = count.div_ceil(CHUNK);
        let partial_count = checked_elements(&[groups, chunks])?;
        let params = NormParams {
            count: u32_of(count)?,
            chunks: u32_of(chunks)?,
            groups: u32_of(groups)?,
            channels: u32_of(channels)?,
            spatial: u32_of(spatial)?,
        };
        u32_of(total)?;
        let device = input.device();
        let partial = device.new_buffer(partial_count, DType::F32, "osor-norm-partial")?;
        let means = device.new_buffer(groups, DType::F32, "osor-norm-means")?;
        let denom = device.new_buffer(groups, DType::F32, "osor-norm-denom")?;
        let output = device.new_buffer(total, DType::F32, "osor-norm-output")?;
        let pipelines = pipelines(device.metal_device())?;
        let guard = device.command_encoder()?;
        let encoder: &ComputeCommandEncoder = guard.as_ref();

        encoder.set_input_buffer(0, Some(input.buffer()), offset(il)?);
        encoder.set_output_buffer(1, Some(partial.as_ref()), 0);
        encoder.set_bytes(2, &params);
        dispatch_reduction(encoder, &pipelines.mean, chunks, groups);

        encoder.set_input_buffer(0, Some(partial.as_ref()), 0);
        encoder.set_output_buffer(1, Some(means.as_ref()), 0);
        encoder.set_bytes(2, &params);
        encoder.set_bytes(3, &0u32);
        dispatch_reduction(encoder, &pipelines.reduce, groups, 1);

        encoder.set_input_buffer(0, Some(input.buffer()), offset(il)?);
        encoder.set_input_buffer(1, Some(means.as_ref()), 0);
        encoder.set_output_buffer(2, Some(partial.as_ref()), 0);
        encoder.set_bytes(3, &params);
        dispatch_reduction(encoder, &pipelines.variance, chunks, groups);

        encoder.set_input_buffer(0, Some(partial.as_ref()), 0);
        encoder.set_output_buffer(1, Some(denom.as_ref()), 0);
        encoder.set_bytes(2, &params);
        encoder.set_bytes(3, &1u32);
        dispatch_reduction(encoder, &pipelines.reduce, groups, 1);

        encoder.set_input_buffer(0, Some(input.buffer()), offset(il)?);
        encoder.set_input_buffer(1, Some(means.as_ref()), 0);
        encoder.set_input_buffer(2, Some(denom.as_ref()), 0);
        encoder.set_input_buffer(3, Some(weight.buffer()), offset(wl)?);
        encoder.set_input_buffer(4, Some(bias.buffer()), offset(bl)?);
        encoder.set_output_buffer(5, Some(output.as_ref()), 0);
        encoder.set_bytes(6, &params);
        dispatch_linear(encoder, &pipelines.apply, total);
        drop(guard);
        Ok((
            MetalStorage::new(output, device.clone(), total, DType::F32),
            il.shape().clone(),
        ))
    }
}

pub(super) fn group_norm(input: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    input
        .contiguous()?
        .apply_op3_no_bwd(&weight.contiguous()?, &bias.contiguous()?, &GroupNorm)
}

struct Upsample;

impl CustomOp1 for Upsample {
    fn name(&self) -> &'static str {
        "osor-metal-vae-upsample"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("VAE Metal upsample is unavailable on CPU")
    }
    fn metal_fwd(&self, input: &MetalStorage, layout: &Layout) -> Result<(MetalStorage, Shape)> {
        contiguous_f32(input, layout)?;
        let (batch, channels, height, width) = layout.shape().dims4()?;
        if [batch, channels, height, width].contains(&0) {
            candle_core::bail!("empty VAE Metal upsample")
        }
        let elements = checked_elements(&[batch, channels, height, width])?;
        let out_height = height
            .checked_mul(2)
            .ok_or_else(|| kernel_error("upsample height overflows"))?;
        let out_width = width
            .checked_mul(2)
            .ok_or_else(|| kernel_error("upsample width overflows"))?;
        let total = checked_elements(&[batch, channels, out_height, out_width])?;
        let params = UpsampleParams {
            width: u32_of(width)?,
            height: u32_of(height)?,
            elements: u32_of(elements)?,
        };
        let device = input.device();
        let output = device.new_buffer(total, DType::F32, "osor-upsample")?;
        let pipelines = pipelines(device.metal_device())?;
        let guard = device.command_encoder()?;
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_input_buffer(0, Some(input.buffer()), offset(layout)?);
        encoder.set_output_buffer(1, Some(output.as_ref()), 0);
        encoder.set_bytes(2, &params);
        dispatch_linear(encoder, &pipelines.upsample, elements);
        drop(guard);
        Ok((
            MetalStorage::new(output, device.clone(), total, DType::F32),
            Shape::from((batch, channels, out_height, out_width)),
        ))
    }
}

pub(in crate::native) fn upsample_2x(input: &Tensor) -> Result<Tensor> {
    input.contiguous()?.apply_op1_no_bwd(&Upsample)
}

struct ChannelBias;

impl CustomOp2 for ChannelBias {
    fn name(&self) -> &'static str {
        "osor-metal-vae-channel-bias"
    }
    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("VAE Metal channel bias is unavailable on CPU")
    }
    fn metal_fwd(
        &self,
        input: &MetalStorage,
        il: &Layout,
        bias: &MetalStorage,
        bl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        contiguous_f32(input, il)?;
        contiguous_f32(bias, bl)?;
        let (batch, channels, height, width) = il.shape().dims4()?;
        let spatial = checked_elements(&[height, width])?;
        if [batch, channels, height, width].contains(&0)
            || !spatial.is_multiple_of(4)
            || !il.start_offset().is_multiple_of(4)
            || bl.shape().dims1()? != channels
        {
            candle_core::bail!("invalid VAE Metal channel bias shape or alignment")
        }
        let total = checked_elements(&[batch, channels, spatial])?;
        let vectors = total / 4;
        let params = BiasParams {
            plane_vectors: u32_of(spatial / 4)?,
            channels: u32_of(channels)?,
            vectors: u32_of(vectors)?,
        };
        let device = input.device();
        let output = device.new_buffer(total, DType::F32, "osor-channel-bias")?;
        let pipelines = pipelines(device.metal_device())?;
        let guard = device.command_encoder()?;
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_input_buffer(0, Some(input.buffer()), offset(il)?);
        encoder.set_input_buffer(1, Some(bias.buffer()), offset(bl)?);
        encoder.set_output_buffer(2, Some(output.as_ref()), 0);
        encoder.set_bytes(3, &params);
        dispatch_linear(encoder, &pipelines.bias, vectors);
        drop(guard);
        Ok((
            MetalStorage::new(output, device.clone(), total, DType::F32),
            il.shape().clone(),
        ))
    }
}

pub(in crate::native) fn channel_bias(input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    input
        .contiguous()?
        .apply_op2_no_bwd(&bias.contiguous()?, &ChannelBias)
}
