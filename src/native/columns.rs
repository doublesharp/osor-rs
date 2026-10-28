//! Channel-major im2col convolution for Metal.
//!
//! Candle (0.9.1 through 0.11.0) lowers a Metal convolution to a pixel-major im2col buffer, a
//! GEMM, and a strided copy back to NCHW. At VAE shapes its im2col kernel and
//! final copy take most of the time while the GEMM runs at 6-9 TFLOPS. This
//! module writes `[batch, channels * k * k, pixels]` columns with coalesced
//! stores, so `weight [out, channels * k * k] x columns` is already NCHW.
//! Columns are produced in bands of output rows to bound scratch memory.
//! Padding is applied by bounds checks inside the column kernel.

use candle_core::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Tensor};

/// Columns for output rows `row_start..row_start + rows`.
#[derive(Debug, Clone, Copy)]
struct Im2Col {
    kernel: usize,
    stride: usize,
    padding: usize,
    out_width: usize,
    row_start: usize,
    rows: usize,
}

impl Im2Col {
    fn shape(&self, batch: usize, channels: usize) -> Result<(usize, usize, usize)> {
        let rows = channels
            .checked_mul(self.kernel * self.kernel)
            .ok_or_else(|| candle_core::Error::Msg("im2col rows overflow".into()))?;
        let pixels = self
            .rows
            .checked_mul(self.out_width)
            .ok_or_else(|| candle_core::Error::Msg("im2col pixels overflow".into()))?;
        rows.checked_mul(pixels)
            .and_then(|n| n.checked_mul(batch))
            .ok_or_else(|| candle_core::Error::Msg("im2col size overflows".into()))?;
        Ok((batch, rows, pixels))
    }
}

impl Im2Col {
    /// Fill column row `index` (`batch * rows + c * k * k + ky * k + kx`).
    fn fill_row(
        &self,
        source: &[f32],
        (channels, height, width): (usize, usize, usize),
        rows: usize,
        index: usize,
        row: &mut [f32],
    ) {
        let k = self.kernel;
        let (b, tap) = (index / rows, index % rows);
        let (c, ky, kx) = (tap / (k * k), (tap / k) % k, tap % k);
        let plane = (b * channels + c) * height;
        for (oy_local, out_row) in row.chunks_mut(self.out_width).enumerate() {
            let Some(iy) =
                ((self.row_start + oy_local) * self.stride + ky).checked_sub(self.padding)
            else {
                continue;
            };
            if iy >= height {
                continue;
            }
            let Some(input_row) = source.get((plane + iy) * width..(plane + iy + 1) * width) else {
                continue;
            };
            for (ox, value) in out_row.iter_mut().enumerate() {
                if let Some(ix) = (ox * self.stride + kx).checked_sub(self.padding)
                    && let Some(&input) = input_row.get(ix)
                {
                    *value = input;
                }
            }
        }
    }
}

impl Im2Col {
    fn cpu_fwd_with_spawn_check(
        &self,
        storage: &CpuStorage,
        layout: &Layout,
        check_spawn: impl Fn() -> std::io::Result<()>,
    ) -> Result<(CpuStorage, Shape)> {
        let (batch, channels, height, width) = layout.shape().dims4()?;
        if !layout.is_contiguous() {
            candle_core::bail!("im2col input must be contiguous")
        }
        let source = storage
            .as_slice::<f32>()?
            .get(layout.start_offset()..)
            .ok_or_else(|| candle_core::Error::Msg("im2col input offset out of range".into()))?;
        let (_, rows, pixels) = self.shape(batch, channels)?;
        let mut columns = vec![0f32; batch * rows * pixels];
        if pixels == 0 {
            return Ok((CpuStorage::F32(columns), Shape::from((batch, rows, pixels))));
        }
        // Each column row is one (batch, channel, ky, kx) tap over the band's
        // output pixels; rows are filled in parallel with whole-row copies.
        let threads = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .clamp(1, 16);
        let rows_per_thread = (batch * rows).div_ceil(threads).max(1);
        std::thread::scope(|scope| -> Result<()> {
            for (chunk_index, chunk) in columns.chunks_mut(rows_per_thread * pixels).enumerate() {
                // The check lets the regression inject the same failure that
                // fallible OS thread creation returns, without exhausting the host.
                check_spawn()
                    .and_then(|()| {
                        std::thread::Builder::new().spawn_scoped(scope, move || {
                            for (offset, row) in chunk.chunks_mut(pixels).enumerate() {
                                self.fill_row(
                                    source,
                                    (channels, height, width),
                                    rows,
                                    chunk_index * rows_per_thread + offset,
                                    row,
                                );
                            }
                        })
                    })
                    .map_err(|error| {
                        candle_core::Error::Msg(format!(
                            "unable to create convolution worker; reduce concurrent inference jobs or raise the OS thread limit: {error}"
                        ))
                    })?;
            }
            Ok(())
        })?;
        Ok((CpuStorage::F32(columns), Shape::from((batch, rows, pixels))))
    }
}

impl CustomOp1 for Im2Col {
    fn name(&self) -> &'static str {
        "osor-im2col"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        self.cpu_fwd_with_spawn_check(storage, layout, || Ok(()))
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn metal_fwd(
        &self,
        storage: &candle_core::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal_kernel::im2col(self, storage, layout)
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_kernel {
    use std::sync::{Arc, Mutex};

    use candle_core::{DType, Layout, MetalStorage, Result, Shape, backend::BackendStorage};
    use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline, Device};
    use objc2_metal::MTLSize;

    use super::Im2Col;

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
struct Params {
    uint batch, channels, height, width, ksize, stride, padding, out_width, row_start, rows;
};
kernel void osor_im2col_f32(
    device const float* source [[buffer(0)]],
    device float* columns [[buffer(1)]],
    constant Params& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
    uint pixels = p.rows * p.out_width;
    uint kk = p.ksize * p.ksize;
    if (gid.x >= pixels || gid.y >= p.channels * kk || gid.z >= p.batch) return;
    uint c = gid.y / kk;
    uint r = gid.y % kk;
    uint oy = p.row_start + gid.x / p.out_width;
    uint ox = gid.x % p.out_width;
    int iy = int(oy * p.stride + r / p.ksize) - int(p.padding);
    int ix = int(ox * p.stride + r % p.ksize) - int(p.padding);
    float value = 0.0f;
    if (iy >= 0 && iy < int(p.height) && ix >= 0 && ix < int(p.width)) {
        value = source[((ulong(gid.z) * p.channels + c) * p.height + uint(iy)) * p.width + uint(ix)];
    }
    columns[(ulong(gid.z) * p.channels * kk + gid.y) * pixels + gid.x] = value;
}
"#;

    #[repr(C)]
    struct Params {
        batch: u32,
        channels: u32,
        height: u32,
        width: u32,
        kernel: u32,
        stride: u32,
        padding: u32,
        out_width: u32,
        row_start: u32,
        rows: u32,
    }

    /// Compiled pipelines keyed by Metal device registry ID.
    static PIPELINES: Mutex<Vec<(u64, Arc<ComputePipeline>)>> = Mutex::new(Vec::new());

    fn kernel_error(error: impl std::fmt::Display) -> candle_core::Error {
        candle_core::Error::Msg(format!("im2col kernel: {error}"))
    }

    fn pipeline(device: &Device) -> Result<Arc<ComputePipeline>> {
        let mut cached = PIPELINES
            .lock()
            .map_err(|_| candle_core::Error::Msg("im2col pipeline cache poisoned".into()))?;
        let id = device.registry_id();
        if let Some((_, pipeline)) = cached.iter().find(|(key, _)| *key == id) {
            return Ok(Arc::clone(pipeline));
        }
        // Pure data movement: default compile options cannot change results.
        let library = device
            .new_library_with_source(SOURCE, None)
            .map_err(kernel_error)?;
        let function = library
            .get_function("osor_im2col_f32", None)
            .map_err(kernel_error)?;
        let pipeline = Arc::new(
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(kernel_error)?,
        );
        cached.push((id, Arc::clone(&pipeline)));
        Ok(pipeline)
    }

    fn u32_of(value: usize) -> Result<u32> {
        u32::try_from(value)
            .map_err(|_| candle_core::Error::Msg("im2col dimension exceeds u32".into()))
    }

    pub(super) fn im2col(
        op: &Im2Col,
        storage: &MetalStorage,
        layout: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        if storage.dtype() != DType::F32 || !layout.is_contiguous() {
            candle_core::bail!("Metal im2col requires a contiguous F32 input")
        }
        let (batch, channels, height, width) = layout.shape().dims4()?;
        let (_, rows, pixels) = op.shape(batch, channels)?;
        let device = storage.device();
        let count = batch * rows * pixels;
        let columns = device.new_buffer(count, DType::F32, "osor-im2col")?;
        let params = Params {
            batch: u32_of(batch)?,
            channels: u32_of(channels)?,
            height: u32_of(height)?,
            width: u32_of(width)?,
            kernel: u32_of(op.kernel)?,
            stride: u32_of(op.stride)?,
            padding: u32_of(op.padding)?,
            out_width: u32_of(op.out_width)?,
            row_start: u32_of(op.row_start)?,
            rows: u32_of(op.rows)?,
        };
        let pipeline = pipeline(device.metal_device())?;
        let guard = device.command_encoder()?;
        let encoder: &ComputeCommandEncoder = guard.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        let offset = layout.start_offset() * DType::F32.size_in_bytes();
        encoder.set_input_buffer(0, Some(storage.buffer()), offset);
        encoder.set_output_buffer(1, Some(columns.as_ref()), 0);
        encoder.set_bytes(2, &params);
        let group = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
        encoder.dispatch_threads(
            MTLSize {
                width: pixels,
                height: rows,
                depth: batch,
            },
            MTLSize {
                width: group.min(pixels).max(1),
                height: 1,
                depth: 1,
            },
        );
        drop(guard);
        Ok((
            MetalStorage::new(columns, device.clone(), count, DType::F32),
            Shape::from((batch, rows, pixels)),
        ))
    }
}

/// `weight [out, channels, k, k]` convolution through banded column GEMMs.
/// `max_scratch_elements` bounds one band of columns.
pub(super) fn conv2d_checked(
    input: &Tensor,
    weight: &Tensor,
    stride: usize,
    padding: usize,
    max_scratch_elements: usize,
    checkpoint: crate::pipeline::Checkpoint<'_>,
) -> crate::Result<Tensor> {
    checkpoint()?;
    let (batch, channels, height, width) = input.dims4()?;
    let (out_channels, weight_channels, kernel, kernel_width) = weight.dims4()?;
    if weight_channels != channels || kernel != kernel_width || kernel == 0 || stride == 0 {
        return Err(candle_core::Error::Msg(
            "column convolution needs a square kernel matching the input".into(),
        )
        .into());
    }
    if input.dtype() != DType::F32 || weight.dtype() != DType::F32 {
        return Err(candle_core::Error::Msg("column convolution requires F32".into()).into());
    }
    let padded_height = height + 2 * padding;
    let padded_width = width + 2 * padding;
    if padded_height < kernel || padded_width < kernel {
        return Err(
            candle_core::Error::Msg("convolution input is smaller than its kernel".into()).into(),
        );
    }
    let out_height = (padded_height - kernel) / stride + 1;
    let out_width = (padded_width - kernel) / stride + 1;
    let column_rows = channels * kernel * kernel;
    let row_elements = batch
        .checked_mul(column_rows)
        .and_then(|n| n.checked_mul(out_width))
        .ok_or_else(|| candle_core::Error::Msg("column scratch size overflows".into()))?;
    let rows_per_band = (max_scratch_elements / row_elements).clamp(1, out_height);
    let input = input.contiguous()?;
    let weight = weight.reshape((out_channels, column_rows))?;
    let mut bands = Vec::with_capacity(out_height.div_ceil(rows_per_band));
    for row_start in (0..out_height).step_by(rows_per_band) {
        checkpoint()?;
        let rows = rows_per_band.min(out_height - row_start);
        let columns = input.apply_op1_no_bwd(&Im2Col {
            kernel,
            stride,
            padding,
            out_width,
            row_start,
            rows,
        })?;
        let mut outputs = Vec::with_capacity(batch);
        for b in 0..batch {
            checkpoint()?;
            outputs.push(weight.matmul(&columns.get(b)?)?);
        }
        let band = if batch == 1 {
            outputs
                .pop()
                .ok_or_else(|| candle_core::Error::Msg("missing column output".into()))?
                .unsqueeze(0)?
        } else {
            Tensor::stack(&outputs, 0)?
        };
        bands.push(band.reshape((batch, out_channels, rows, out_width))?);
    }
    checkpoint()?;
    Ok(if bands.len() == 1 {
        bands
            .pop()
            .ok_or_else(|| candle_core::Error::Msg("missing column band".into()))?
    } else {
        Tensor::cat(&bands, 2)?
    })
}

#[cfg(test)]
pub(super) fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    stride: usize,
    padding: usize,
    max_scratch_elements: usize,
) -> Result<Tensor> {
    conv2d_checked(
        input,
        weight,
        stride,
        padding,
        max_scratch_elements,
        &|| Ok(()),
    )
    .map_err(|error| candle_core::Error::Msg(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn native_convolution_cancels_between_output_bands() -> crate::Result<()> {
        use std::cell::Cell;
        let device = &Device::Cpu;
        let input = Tensor::ones((1, 2, 8, 8), DType::F32, device)?;
        let weight = Tensor::ones((2, 2, 3, 3), DType::F32, device)?;
        let calls = Cell::new(0);
        let result = conv2d_checked(&input, &weight, 1, 1, 2 * 3 * 3 * 8, &|| {
            calls.set(calls.get() + 1);
            // The first band has completed: entry, first-band, and its
            // batch GEMM checkpoints precede the next-band checkpoint.
            if calls.get() == 4 {
                Err(crate::Error::Cancelled)
            } else {
                Ok(())
            }
        });
        assert!(matches!(result, Err(crate::Error::Cancelled)));
        assert_eq!(calls.get(), 4);
        Ok(())
    }

    #[test]
    fn thread_creation_failure_returns_an_actionable_error() -> Result<()> {
        let op = Im2Col {
            kernel: 1,
            stride: 1,
            padding: 0,
            out_width: 4,
            row_start: 0,
            rows: 4,
        };
        let storage = CpuStorage::F32(vec![1.; 2 * 3 * 4 * 4]);
        let layout = Layout::contiguous((2, 3, 4, 4));
        let result = op.cpu_fwd_with_spawn_check(&storage, &layout, || {
            Err(std::io::Error::other("injected thread limit"))
        });
        match result {
            Err(error) => {
                let message = error.to_string();
                assert!(message.contains("convolution worker"), "{message}");
                assert!(message.contains("injected thread limit"), "{message}");
            }
            Ok(_) => candle_core::bail!("thread creation failure must return an error"),
        }
        Ok(())
    }

    fn max_difference(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()
    }

    #[test]
    fn column_convolution_matches_candle_on_cpu() -> Result<()> {
        let device = &Device::Cpu;
        // (batch, channels, height, width, out, kernel, stride, padding, scratch rows)
        for (index, (batch, channels, height, width, out, kernel, stride, padding, rows)) in [
            (1, 6, 37, 20, 5, 3, 1, 1, 100),
            (2, 6, 37, 20, 5, 3, 1, 1, 4),
            (1, 4, 33, 17, 3, 3, 2, 0, 3),
            (3, 8, 19, 11, 7, 1, 1, 0, 2),
        ]
        .into_iter()
        .enumerate()
        {
            let input = Tensor::randn(0f32, 1f32, (batch, channels, height, width), device)?;
            let weight = Tensor::randn(0f32, 0.2f32, (out, channels, kernel, kernel), device)?;
            let out_width = (width + 2 * padding - kernel) / stride + 1;
            let expected = input.conv2d(&weight, padding, stride, 1, 1)?;
            let actual = conv2d(
                &input,
                &weight,
                stride,
                padding,
                batch * channels * kernel * kernel * out_width * rows,
            )?;
            assert_eq!(actual.dims(), expected.dims(), "case {index}");
            let difference = max_difference(&actual, &expected)?;
            assert!(difference <= 1e-5, "case {index} differs by {difference}");
        }
        Ok(())
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a Metal device"]
    fn metal_columns_match_cpu_exactly_and_convolution_matches_candle() -> Result<()> {
        let metal = Device::new_metal(0)?;
        for (batch, channels, height, width, kernel, stride, padding) in [
            (1, 5, 37, 21, 3, 1, 1),
            (2, 3, 33, 17, 3, 2, 0),
            (1, 7, 9, 13, 1, 1, 0),
        ] {
            let input = Tensor::randn(0f32, 1f32, (batch, channels, height, width), &Device::Cpu)?;
            let out_width = (width + 2 * padding - kernel) / stride + 1;
            let op = Im2Col {
                kernel,
                stride,
                padding,
                out_width,
                row_start: 1,
                rows: 3,
            };
            let cpu = input.apply_op1_no_bwd(&op)?;
            let gpu = input.to_device(&metal)?.apply_op1_no_bwd(&op)?;
            assert_eq!(
                cpu.flatten_all()?.to_vec1::<f32>()?,
                gpu.flatten_all()?.to_vec1::<f32>()?
            );
            let weight = Tensor::randn(0f32, 0.2f32, (4, channels, kernel, kernel), &metal)?;
            let input = input.to_device(&metal)?;
            let expected = input.conv2d(&weight, padding, stride, 1, 1)?;
            let actual = conv2d(&input, &weight, stride, padding, 64)?;
            let difference = max_difference(&actual, &expected)?;
            assert!(
                difference <= 1e-5,
                "Metal column convolution differs by {difference}"
            );
        }
        Ok(())
    }

    #[test]
    fn rejects_mismatched_weights_and_non_f32_inputs() -> Result<()> {
        let device = &Device::Cpu;
        let input = Tensor::zeros((1, 4, 8, 8), DType::F32, device)?;
        let weight = Tensor::zeros((2, 3, 3, 3), DType::F32, device)?;
        assert!(conv2d(&input, &weight, 1, 1, 1024).is_err());
        let weight = Tensor::zeros((2, 4, 3, 3), DType::F64, device)?;
        assert!(conv2d(&input, &weight, 1, 1, 1024).is_err());
        Ok(())
    }
}
