//! Deterministic standard-normal noise from a seed.
//!
//! Inference always takes explicit noise tensors; this module is a convenience
//! for applications that want to store a seed instead of the tensors. The
//! generator is part of the stored-seed contract, version [`NOISE_VERSION`]:
//!
//! - Stream 0 fills `posterior`, stream 1 fills `diffusion`.
//! - Each stream is a xoshiro256** generator whose four state words are the
//!   first four SplitMix64 outputs from `seed + stream * 0x9E37_79B9_7F4A_7C15`.
//! - Values are Box–Muller pairs: `u1` maps the top 53 bits of one output to
//!   `(0, 1]`, `u2` maps the next output to `[0, 1)`; the pair is
//!   `r * cos(2πu2), r * sin(2πu2)` with `r = sqrt(-2 ln u1)`, computed in F64
//!   with the pure-Rust `libm` functions so results do not depend on the
//!   platform math library, then rounded to F32.
//! - Values fill each `[B, 4, H/8, W/8]` tensor in row-major NCHW order.
//!
//! Changing any of these rules changes the noise for existing seeds and needs
//! a new version. A seed is not portable to other frameworks: cross-framework
//! parity still requires the captured tensors.

use candle_core::{DType, Device, Tensor};

use crate::{
    Error, Result,
    pipeline::{IMAGE_AXIS_MULTIPLE, MAX_BATCH, valid_image_axis},
};

/// Version of the seed-to-noise contract described in the module docs.
pub const NOISE_VERSION: u32 = 1;

const STREAM_STRIDE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Posterior and diffusion noise for one job.
#[derive(Debug, Clone)]
pub struct Noise {
    /// VAE posterior noise `[B, 4, H/8, W/8]`.
    pub posterior: Tensor,
    /// Diffusion noise `[B, 4, H/8, W/8]`.
    pub diffusion: Tensor,
}

fn split_mix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(STREAM_STRIDE);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Xoshiro256 {
    state: [u64; 4],
}

impl Xoshiro256 {
    fn new(seed: u64, stream: u64) -> Self {
        let mut mixer = seed.wrapping_add(stream.wrapping_mul(STREAM_STRIDE));
        Self {
            state: [
                split_mix64(&mut mixer),
                split_mix64(&mut mixer),
                split_mix64(&mut mixer),
                split_mix64(&mut mixer),
            ],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let [s0, s1, s2, s3] = &mut self.state;
        let result = s1.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = *s1 << 17;
        *s2 ^= *s0;
        *s3 ^= *s1;
        *s1 ^= *s2;
        *s0 ^= *s3;
        *s2 ^= t;
        *s3 = s3.rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }
}

fn standard_normal(seed: u64, stream: u64, count: usize) -> Vec<f32> {
    let mut generator = Xoshiro256::new(seed, stream);
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let u1 = 1.0 - generator.unit();
        let u2 = generator.unit();
        let radius = libm::sqrt(-2.0 * libm::log(u1));
        let angle = 2.0 * std::f64::consts::PI * u2;
        values.push((radius * libm::cos(angle)) as f32);
        if values.len() < count {
            values.push((radius * libm::sin(angle)) as f32);
        }
    }
    values
}

/// Generate both noise tensors for a `[batch, 3, height, width]` job.
///
/// `height` and `width` follow the image admission rules (multiples of eight
/// in `32..=1024`); `batch` is in `1..=MAX_BATCH`. Pass the model's dtype and
/// device, for example from [`crate::Osor::seeded_noise`].
pub fn seeded_noise(
    seed: u64,
    batch: usize,
    height: usize,
    width: usize,
    dtype: DType,
    device: &Device,
) -> Result<Noise> {
    if !(1..=MAX_BATCH).contains(&batch) || !valid_image_axis(height) || !valid_image_axis(width) {
        return Err(Error::InvalidInput(format!(
            "seeded noise needs batch 1..={MAX_BATCH} and image axes that the model admits"
        )));
    }
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(Error::InvalidInput(
            "seeded noise dtype must be F32, F16, or BF16".into(),
        ));
    }
    let shape = (
        batch,
        4,
        height / IMAGE_AXIS_MULTIPLE,
        width / IMAGE_AXIS_MULTIPLE,
    );
    let count = batch * 4 * (height / IMAGE_AXIS_MULTIPLE) * (width / IMAGE_AXIS_MULTIPLE);
    let tensor = |stream: u64| -> Result<Tensor> {
        Ok(
            Tensor::from_vec(standard_normal(seed, stream, count), shape, &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(device)?,
        )
    };
    Ok(Noise {
        posterior: tensor(0)?,
        diffusion: tensor(1)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(tensor: &Tensor) -> Result<Vec<f32>> {
        Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
    }

    #[test]
    fn version_one_values_are_pinned() -> Result<()> {
        let noise = seeded_noise(0, 1, 32, 32, DType::F32, &Device::Cpu)?;
        let posterior = values(&noise.posterior)?;
        let diffusion = values(&noise.diffusion)?;
        let head = |v: &[f32]| v.iter().take(4).copied().collect::<Vec<_>>();
        // Golden values for NOISE_VERSION 1; a change here breaks stored seeds.
        assert_eq!(head(&posterior), GOLDEN_POSTERIOR.to_vec());
        assert_eq!(head(&diffusion), GOLDEN_DIFFUSION.to_vec());
        Ok(())
    }

    // Cross-checked against an independent implementation of the documented
    // algorithm (Python integers for the generator, F64 Box–Muller).
    const GOLDEN_POSTERIOR: [f32; 4] = [-0.018964991, -1.3559302, -0.4037211, 0.23335098];
    const GOLDEN_DIFFUSION: [f32; 4] = [0.5520624, -0.5417131, 0.9305711, 0.027425194];

    #[test]
    fn deterministic_independent_streams_with_standard_moments() -> Result<()> {
        let device = &Device::Cpu;
        let a = seeded_noise(42, 4, 1024, 1024, DType::F32, device)?;
        let b = seeded_noise(42, 4, 1024, 1024, DType::F32, device)?;
        let c = seeded_noise(43, 4, 1024, 1024, DType::F32, device)?;
        assert_eq!(a.posterior.dims(), &[4, 4, 128, 128]);
        assert_eq!(values(&a.posterior)?, values(&b.posterior)?);
        assert_eq!(values(&a.diffusion)?, values(&b.diffusion)?);
        assert_ne!(values(&a.posterior)?, values(&a.diffusion)?);
        assert_ne!(values(&a.posterior)?, values(&c.posterior)?);
        for tensor in [&a.posterior, &a.diffusion, &c.posterior] {
            let v = values(tensor)?;
            let n = v.len() as f64;
            let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
            let variance = v
                .iter()
                .map(|&x| (f64::from(x) - mean).powi(2))
                .sum::<f64>()
                / n;
            assert!(v.iter().all(|x| x.is_finite()));
            // 262,144 samples: standard error of the mean is about 0.002.
            assert!(mean.abs() < 0.01, "mean {mean}");
            assert!((variance - 1.0).abs() < 0.02, "variance {variance}");
        }
        Ok(())
    }

    #[test]
    fn smaller_jobs_are_prefixes_of_the_same_stream_and_dtypes_convert() -> Result<()> {
        let device = &Device::Cpu;
        let small = seeded_noise(7, 1, 32, 40, DType::F32, device)?;
        let large = seeded_noise(7, 2, 32, 40, DType::F32, device)?;
        let small_values = values(&small.posterior)?;
        let large_values = values(&large.posterior)?;
        assert_eq!(small_values, large_values[..small_values.len()]);
        let half = seeded_noise(7, 1, 32, 40, DType::F16, device)?;
        assert_eq!(half.diffusion.dtype(), DType::F16);
        Ok(())
    }

    #[test]
    fn rejects_inadmissible_shapes_and_dtypes() {
        let device = &Device::Cpu;
        for (batch, height, width) in [(0, 32, 32), (5, 32, 32), (1, 33, 32), (1, 32, 1032)] {
            assert!(matches!(
                seeded_noise(1, batch, height, width, DType::F32, device),
                Err(Error::InvalidInput(_))
            ));
        }
        assert!(matches!(
            seeded_noise(1, 1, 32, 32, DType::F64, device),
            Err(Error::InvalidInput(_))
        ));
    }
}
