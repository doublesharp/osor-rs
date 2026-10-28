//! Explicit Metal convolution timing at real VAE shapes; never runs by default.

use std::time::Instant;

use candle_core::{DType, Device, Tensor};

use crate::Result;

fn time(device: &Device, runs: usize, mut f: impl FnMut() -> Result<Tensor>) -> Result<f64> {
    let _ = f()?;
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..runs {
        let _ = f()?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1000.0 / runs as f64)
}

#[test]
#[ignore = "requires a Metal device; prints timings"]
fn metal_convolution_breakdown_at_vae_shapes() -> Result<()> {
    let device = Device::new_metal(0)?;
    for (channels, height, width) in [
        (128, 384, 640),
        (256, 384, 640),
        (512, 192, 320),
        (512, 96, 160),
    ] {
        let input = Tensor::randn(0f32, 1f32, (1, channels, height, width), &device)?;
        let weight = Tensor::randn(0f32, 0.05f32, (channels, channels, 3, 3), &device)?;
        let conv = time(&device, 5, || Ok(input.conv2d(&weight, 1, 1, 1, 1)?))?;
        let pixels = height * width;
        let k = channels * 9;
        let columns = Tensor::randn(0f32, 1f32, (1, pixels, k), &device)?;
        let kernel = weight.reshape((1, channels, k))?.transpose(1, 2)?;
        let gemm = time(&device, 5, || Ok(columns.matmul(&kernel)?))?;
        let nhwc = Tensor::randn(0f32, 1f32, (1, height, width, channels), &device)?;
        let transpose = time(&device, 5, || {
            Ok(nhwc.permute((0, 3, 1, 2))?.contiguous()?)
        })?;
        let columns_conv = time(&device, 5, || {
            Ok(super::columns::conv2d(
                &input,
                &weight,
                1,
                1,
                super::conv::MAX_SCRATCH_ELEMENTS,
            )?)
        })?;
        let flops = 2.0 * pixels as f64 * channels as f64 * k as f64;
        println!(
            "C={channels} {height}x{width}: candle conv {conv:.1} ms ({:.2} TFLOPS), column conv {columns_conv:.1} ms ({:.2} TFLOPS), gemm {gemm:.1} ms ({:.2} TFLOPS), transpose {transpose:.1} ms",
            flops / conv / 1e9,
            flops / columns_conv / 1e9,
            flops / gemm / 1e9
        );
        drop(columns);
    }
    let _ = DType::F32;
    Ok(())
}
