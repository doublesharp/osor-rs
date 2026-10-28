//! Safe Rust adaptation of Pillow 13.0.0's eight-bit resampling paths.
//! Pillow's copyright and permission notice is preserved in `LICENSE-PILLOW`.

use super::{ImageSize, MAX_PREPROCESS_AXIS, MAX_PREPROCESS_PIXELS};
use crate::{Error, Result};

const PRECISION_BITS: u32 = 22;
const ROUNDING_BIAS: i64 = 1 << (PRECISION_BITS - 1);
// Bicubic has four source taps per downscaled source pixel plus a bounded
// number of edge/upsampling taps per output pixel. Enforce an independent cap.
const MAX_COEFFICIENTS: usize = MAX_PREPROCESS_AXIS as usize * 10;

#[derive(Clone, Copy)]
pub(super) enum Filter {
    Bicubic,
    Bilinear,
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Self::Bicubic => 2.0,
            Self::Bilinear => 1.0,
        }
    }

    fn value(self, x: f64) -> f64 {
        let x = x.abs();
        match self {
            Self::Bilinear if x < 1.0 => 1.0 - x,
            Self::Bilinear => 0.0,
            Self::Bicubic if x < 1.0 => ((1.5 * x - 2.5) * x * x) + 1.0,
            Self::Bicubic if x < 2.0 => (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5,
            Self::Bicubic => 0.0,
        }
    }
}

pub(super) struct Layout {
    width: usize,
    height: usize,
    pub(super) pixels: usize,
    pub(super) bytes: usize,
}

pub(super) fn layout(size: ImageSize, channels: usize) -> Result<Layout> {
    if size.width == 0
        || size.height == 0
        || size.width > MAX_PREPROCESS_AXIS
        || size.height > MAX_PREPROCESS_AXIS
        || !matches!(channels, 1 | 3)
    {
        return Err(Error::InvalidInput(format!(
            "byte image width and height must be in 1..={MAX_PREPROCESS_AXIS}"
        )));
    }
    let width = usize::try_from(size.width)
        .map_err(|_| Error::InvalidInput("image width exceeds platform limits".into()))?;
    let height = usize::try_from(size.height)
        .map_err(|_| Error::InvalidInput("image height exceeds platform limits".into()))?;
    let pixels = width
        .checked_mul(height)
        .filter(|&pixels| pixels <= MAX_PREPROCESS_PIXELS)
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "byte image or intermediate exceeds {MAX_PREPROCESS_PIXELS} pixels"
            ))
        })?;
    let bytes = pixels
        .checked_mul(channels)
        .ok_or_else(|| Error::InvalidInput("byte image length overflows".into()))?;
    Ok(Layout {
        width,
        height,
        pixels,
        bytes,
    })
}

pub(super) fn validate(
    data: &[u8],
    size: ImageSize,
    channels: usize,
    name: &str,
) -> Result<Layout> {
    let layout = layout(size, channels)?;
    if data.len() != layout.bytes {
        return Err(Error::InvalidInput(format!(
            "{name} requires exactly {} bytes for {}x{} with {channels} channels; got {}",
            layout.bytes,
            size.width,
            size.height,
            data.len()
        )));
    }
    Ok(layout)
}

pub(super) fn zeroed<T: Default + Clone>(length: usize) -> Result<Vec<T>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(length)
        .map_err(|_| Error::InvalidInput("unable to allocate preprocessing buffer".into()))?;
    output.resize(length, T::default());
    Ok(output)
}

fn copy(data: &[u8]) -> Result<Vec<u8>> {
    let mut output = zeroed::<u8>(data.len())?;
    output.copy_from_slice(data);
    Ok(output)
}

struct Kernel {
    first: usize,
    weights: Vec<i32>,
}

fn coefficients(source: usize, target: usize, filter: Filter) -> Result<Vec<Kernel>> {
    let scale = source as f64 / target as f64;
    let filter_scale = scale.max(1.0);
    let support = filter.support() * filter_scale;
    let inverse_scale = 1.0 / filter_scale;
    let mut kernels = Vec::new();
    kernels
        .try_reserve_exact(target)
        .map_err(|_| Error::InvalidInput("unable to allocate resize kernels".into()))?;
    let mut total_coefficients = 0usize;
    for index in 0..target {
        let center = (index as f64 + 0.5) * scale;
        // Pillow converts to signed int by truncating toward zero before
        // clipping bounds. This matters at the first pixel near the edge.
        let first = ((center - support + 0.5) as isize).max(0) as usize;
        let end = ((center + support + 0.5) as usize).min(source);
        let count = end
            .checked_sub(first)
            .ok_or_else(|| Error::InvalidInput("invalid computed resize filter bounds".into()))?;
        total_coefficients = total_coefficients
            .checked_add(count)
            .filter(|&count| count <= MAX_COEFFICIENTS)
            .ok_or_else(|| Error::InvalidInput("resize filter exceeds coefficient limit".into()))?;
        let mut raw = Vec::new();
        raw.try_reserve_exact(count)
            .map_err(|_| Error::InvalidInput("unable to allocate resize weights".into()))?;
        let mut sum = 0.0;
        for offset in 0..count {
            let value = filter.value(((offset + first) as f64 - center + 0.5) * inverse_scale);
            raw.push(value);
            sum += value;
        }
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(count)
            .map_err(|_| Error::InvalidInput("unable to allocate fixed resize weights".into()))?;
        for value in raw {
            let normalized = if sum == 0.0 { value } else { value / sum };
            let scaled = normalized * f64::from(1 << PRECISION_BITS);
            weights.push(if scaled < 0.0 {
                (scaled - 0.5) as i32
            } else {
                (scaled + 0.5) as i32
            });
        }
        kernels.push(Kernel { first, weights });
    }
    Ok(kernels)
}

fn clip(value: i64) -> u8 {
    (value >> PRECISION_BITS).clamp(0, 255) as u8
}

fn horizontal(
    input: &[u8],
    source: ImageSize,
    width: u32,
    channels: usize,
    filter: Filter,
) -> Result<Vec<u8>> {
    let source_layout = validate(input, source, channels, "resize input")?;
    let target = ImageSize {
        width,
        height: source.height,
    };
    let target_layout = layout(target, channels)?;
    let kernels = coefficients(source_layout.width, target_layout.width, filter)?;
    let mut output = zeroed::<u8>(target_layout.bytes)?;
    let source_stride = source_layout.width * channels;
    let target_stride = target_layout.width * channels;
    for (row, output_row) in input
        .chunks_exact(source_stride)
        .zip(output.chunks_exact_mut(target_stride))
    {
        for (kernel, output_pixel) in kernels.iter().zip(output_row.chunks_exact_mut(channels)) {
            let first = kernel.first * channels;
            let end = (kernel.first + kernel.weights.len()) * channels;
            let samples = row.get(first..end).ok_or_else(|| {
                Error::InvalidInput("computed horizontal resize bounds are invalid".into())
            })?;
            let mut sums = [ROUNDING_BIAS; 3];
            for (sample, &weight) in samples.chunks_exact(channels).zip(&kernel.weights) {
                for (sum, &value) in sums.iter_mut().zip(sample) {
                    *sum += i64::from(value) * i64::from(weight);
                }
            }
            for (byte, sum) in output_pixel.iter_mut().zip(sums) {
                *byte = clip(sum);
            }
        }
    }
    Ok(output)
}

fn vertical(
    input: &[u8],
    source: ImageSize,
    height: u32,
    channels: usize,
    filter: Filter,
) -> Result<Vec<u8>> {
    let source_layout = validate(input, source, channels, "resize input")?;
    let target = ImageSize {
        width: source.width,
        height,
    };
    let target_layout = layout(target, channels)?;
    let kernels = coefficients(source_layout.height, target_layout.height, filter)?;
    let mut output = zeroed::<u8>(target_layout.bytes)?;
    let stride = source_layout.width * channels;
    let mut sums = zeroed::<i64>(stride)?;
    for (kernel, output_row) in kernels.iter().zip(output.chunks_exact_mut(stride)) {
        sums.fill(ROUNDING_BIAS);
        let first = kernel.first * stride;
        let end = (kernel.first + kernel.weights.len()) * stride;
        let rows = input.get(first..end).ok_or_else(|| {
            Error::InvalidInput("computed vertical resize bounds are invalid".into())
        })?;
        for (row, &weight) in rows.chunks_exact(stride).zip(&kernel.weights) {
            for (sum, &value) in sums.iter_mut().zip(row) {
                *sum += i64::from(value) * i64::from(weight);
            }
        }
        for (byte, &sum) in output_row.iter_mut().zip(&sums) {
            *byte = clip(sum);
        }
    }
    Ok(output)
}

pub(super) fn filtered(
    input: &[u8],
    source: ImageSize,
    target: ImageSize,
    channels: usize,
    filter: Filter,
) -> Result<Vec<u8>> {
    validate(input, source, channels, "resize input")?;
    layout(target, channels)?;
    let horizontal_needed = source.width != target.width;
    let vertical_needed = source.height != target.height;
    match (horizontal_needed, vertical_needed) {
        (false, false) => copy(input),
        (true, false) => horizontal(input, source, target.width, channels, filter),
        (false, true) => vertical(input, source, target.height, channels, filter),
        (true, true) => {
            // Pillow 13 selects pass order based on the absolute reductions,
            // and rounds to bytes between passes, so order is observable.
            let vertical_reduction = i64::from(source.height) - i64::from(target.height);
            let horizontal_reduction = i64::from(source.width) - i64::from(target.width);
            let horizontal_first =
                !(vertical_reduction > 0 && vertical_reduction > horizontal_reduction * 2);
            if horizontal_first {
                let intermediate = ImageSize {
                    width: target.width,
                    height: source.height,
                };
                layout(intermediate, channels)?;
                let first = horizontal(input, source, target.width, channels, filter)?;
                vertical(&first, intermediate, target.height, channels, filter)
            } else {
                let intermediate = ImageSize {
                    width: source.width,
                    height: target.height,
                };
                layout(intermediate, channels)?;
                let first = vertical(input, source, target.height, channels, filter)?;
                horizontal(&first, intermediate, target.width, channels, filter)
            }
        }
    }
}

pub(super) fn nearest(input: &[u8], source: ImageSize, target: ImageSize) -> Result<Vec<u8>> {
    let source_layout = validate(input, source, 1, "mask")?;
    let target_layout = layout(target, 1)?;
    if source == target {
        return copy(input);
    }
    let x_step = source_layout.width as f64 / target_layout.width as f64;
    let y_step = source_layout.height as f64 / target_layout.height as f64;
    let mut x = x_step * 0.5;
    let mut columns = Vec::new();
    columns
        .try_reserve_exact(target_layout.width)
        .map_err(|_| Error::InvalidInput("unable to allocate nearest-neighbor columns".into()))?;
    // Pillow's affine nearest path advances the double coordinate at each
    // pixel rather than recomputing it; ties can differ for recurring ratios.
    for _ in 0..target_layout.width {
        columns.push(x as usize);
        x += x_step;
    }
    let mut output = zeroed::<u8>(target_layout.bytes)?;
    let mut y = y_step * 0.5;
    for output_row in output.chunks_exact_mut(target_layout.width) {
        let start = (y as usize) * source_layout.width;
        let row = input
            .get(start..start + source_layout.width)
            .ok_or_else(|| Error::InvalidInput("invalid nearest-neighbor row".into()))?;
        for (byte, &column) in output_row.iter_mut().zip(&columns) {
            *byte = *row
                .get(column)
                .ok_or_else(|| Error::InvalidInput("invalid nearest-neighbor column".into()))?;
        }
        y += y_step;
    }
    Ok(output)
}
