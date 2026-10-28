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

#[test]
#[ignore = "requires a Metal device; prints timings"]
fn metal_decoder_elementwise_and_norm_costs() -> Result<()> {
    use candle_nn::{Module, VarBuilder};
    use std::collections::HashMap;
    let device = Device::new_metal(0)?;
    for (channels, height, width) in [(128, 384, 640), (256, 384, 640), (512, 192, 320)] {
        let xs = Tensor::randn(0f32, 1f32, (1, channels, height, width), &device)?;
        let other = Tensor::randn(0f32, 1f32, (1, channels, height, width), &device)?;
        let bias = Tensor::randn(0f32, 1f32, channels, &device)?;
        let weights = HashMap::from([
            (
                "weight".to_owned(),
                Tensor::ones(channels, DType::F32, &device)?,
            ),
            (
                "bias".to_owned(),
                Tensor::zeros(channels, DType::F32, &device)?,
            ),
        ]);
        let norm = super::vae_encoder::StableGroupNorm::new(
            channels,
            VarBuilder::from_tensors(weights, DType::F32, &device),
        )?;
        let old_group_norm = time(&device, 5, || Ok(norm.forward_reference(&xs)?))?;
        let group_norm = time(&device, 5, || Ok(norm.forward(&xs)?))?;
        let old_norm_silu = time(&device, 5, || {
            Ok(candle_nn::ops::silu(&norm.forward_reference(&xs)?)?)
        })?;
        let norm_silu = time(&device, 5, || Ok(norm.forward_silu(&xs)?))?;
        let silu = time(&device, 5, || Ok(candle_nn::ops::silu(&xs)?))?;
        let add = time(&device, 5, || Ok((&xs + &other)?))?;
        let old_bias_add = time(&device, 5, || {
            Ok(xs.broadcast_add(&bias.reshape((1, channels, 1, 1))?)?)
        })?;
        let bias_add = time(&device, 5, || {
            Ok(super::vae_encoder::metal_ops::channel_bias(&xs, &bias)?)
        })?;
        let half = xs
            .narrow(2, 0, height / 2)?
            .narrow(3, 0, width / 2)?
            .contiguous()?;
        let old_upsample = time(&device, 5, || Ok(half.upsample_nearest2d(height, width)?))?;
        let upsample = time(&device, 5, || Ok(super::vae_decoder::upsample_2x(&half)?))?;
        let weight = Tensor::randn(0f32, 0.05f32, (channels, channels, 3, 3), &device)?;
        let banded = time(&device, 3, || {
            Ok(super::columns::conv2d(
                &xs,
                &weight,
                1,
                1,
                super::conv::MAX_SCRATCH_ELEMENTS,
            )?)
        })?;
        let whole = time(&device, 3, || {
            Ok(super::columns::conv2d(&xs, &weight, 1, 1, usize::MAX)?)
        })?;
        println!(
            "C={channels} {height}x{width}: group norm old {old_group_norm:.1} new {group_norm:.1} ms, norm+silu old {old_norm_silu:.1} new {norm_silu:.1} ms, silu {silu:.1} ms, add {add:.1} ms, bias old {old_bias_add:.1} new {bias_add:.1} ms, upsample old {old_upsample:.1} new {upsample:.1} ms, conv banded {banded:.1} ms, conv one band {whole:.1} ms"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Metal device; prints timings"]
fn metal_elementwise_dispatch_costs() -> Result<()> {
    let device = Device::new_metal(0)?;
    let xs = Tensor::randn(0f32, 1f32, (1, 128, 384, 640), &device)?;
    let other = Tensor::randn(0f32, 1f32, (1, 128, 384, 640), &device)?;
    for round in 0..2 {
        let silu = time(&device, 5, || Ok(candle_nn::ops::silu(&xs)?))?;
        let affine = time(&device, 5, || Ok(xs.affine(2.0, 1.0)?))?;
        let add = time(&device, 5, || Ok((&xs + &other)?))?;
        let mul = time(&device, 5, || Ok((&xs * &other)?))?;
        let sqr = time(&device, 5, || Ok(xs.sqr()?))?;
        let sum = time(&device, 5, || {
            Ok(xs.reshape((1, 32, 4 * 384 * 640))?.sum_keepdim(2)?)
        })?;
        println!(
            "round {round}: silu {silu:.1} affine {affine:.1} add {add:.1} mul {mul:.1} sqr {sqr:.1} sum {sum:.1} ms"
        );
    }
    Ok(())
}
