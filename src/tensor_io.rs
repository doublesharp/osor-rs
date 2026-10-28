//! Bounded safetensors I/O for explicit inputs, noise, and parity captures.

use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{BufWriter, Read, Write},
    path::Path,
};

use candle_core::{DType, Device, Tensor};

use crate::{Error, Result};

const MAX_BYTES: u64 = 512 * 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 8 * 1024 * 1024;
const MAX_ELEMENTS: usize = 64 * 1024 * 1024;
const MAX_TENSORS: usize = 64;

fn layout(name: &str, shape: &[usize]) -> Result<usize> {
    if name.is_empty() || name.len() > 128 || shape.len() > 8 {
        return Err(Error::InvalidInput(
            "capture tensor name or rank exceeds limits".into(),
        ));
    }
    shape.iter().try_fold(1_usize, |count, dimension| {
        if *dimension == 0 || *dimension > 16_384 {
            return Err(Error::InvalidInput(format!(
                "capture tensor {name} dimension exceeds limits"
            )));
        }
        count
            .checked_mul(*dimension)
            .ok_or_else(|| Error::InvalidInput("capture dimensions overflow".into()))
    })
}

/// Read small inference inputs or captured stages. Model weights use the bundle loader.
pub fn load(path: impl AsRef<Path>, device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() < 8 || metadata.len() > MAX_BYTES {
        return Err(Error::InvalidInput(format!(
            "tensor capture must be a regular safetensors file of 8..={MAX_BYTES} bytes"
        )));
    }
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)?;
    let header_length = u64::from_le_bytes(prefix);
    if header_length > MAX_HEADER_BYTES || header_length > metadata.len().saturating_sub(8) {
        return Err(Error::InvalidInput(
            "invalid or oversized safetensors header".into(),
        ));
    }
    let length = usize::try_from(metadata.len())
        .map_err(|_| Error::InvalidInput("tensor capture is too large for this platform".into()))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|error| Error::InvalidInput(format!("cannot reserve capture memory: {error}")))?;
    bytes.extend_from_slice(&prefix);
    file.take(MAX_BYTES - 7).read_to_end(&mut bytes)?;
    if bytes.len() != length {
        return Err(Error::InvalidInput(
            "tensor capture changed while being read".into(),
        ));
    }
    let archive = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|error| Error::InvalidInput(format!("invalid safetensors capture: {error}")))?;
    if archive.is_empty() || archive.len() > MAX_TENSORS {
        return Err(Error::InvalidInput(format!(
            "capture must contain 1..={MAX_TENSORS} tensors"
        )));
    }
    let mut total = 0_usize;
    for (name, view) in archive.tensors() {
        if !matches!(
            view.dtype(),
            safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16
        ) {
            return Err(Error::InvalidInput(format!(
                "capture tensor {name} must be F32, F16, or BF16"
            )));
        }
        total = total
            .checked_add(layout(&name, view.shape())?)
            .ok_or_else(|| Error::InvalidInput("capture size overflows".into()))?;
        if total > MAX_ELEMENTS {
            return Err(Error::InvalidInput(format!(
                "capture exceeds {MAX_ELEMENTS} elements"
            )));
        }
    }
    Ok(candle_core::safetensors::load_buffer(&bytes, device)?)
}

#[derive(serde::Serialize)]
struct TensorInfo<'a> {
    dtype: &'static str,
    shape: &'a [usize],
    data_offsets: (usize, usize),
}

/// Write a new F32 capture without replacing an existing file.
///
/// F16/BF16 inputs are losslessly widened to F32. Transfers are fallible; this
/// deliberately avoids Candle's `safetensors::View`, whose readback can panic.
/// On a write/readback error, a partial file can remain and must not be used.
pub fn save_new(tensors: &HashMap<String, Tensor>, path: impl AsRef<Path>) -> Result<()> {
    if tensors.is_empty() || tensors.len() > MAX_TENSORS {
        return Err(Error::InvalidInput("invalid capture tensor count".into()));
    }
    let sorted: BTreeMap<_, _> = tensors.iter().collect();
    let mut metadata = BTreeMap::new();
    let mut elements = 0_usize;
    let mut offset = 0_usize;
    for (name, tensor) in &sorted {
        if !matches!(tensor.dtype(), DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::InvalidInput(
                "capture tensors must be F32, F16, or BF16".into(),
            ));
        }
        let count = layout(name, tensor.dims())?;
        elements = elements
            .checked_add(count)
            .ok_or_else(|| Error::InvalidInput("capture size overflows".into()))?;
        if elements > MAX_ELEMENTS {
            return Err(Error::InvalidInput("capture exceeds element limit".into()));
        }
        let end = count
            .checked_mul(4)
            .and_then(|bytes| offset.checked_add(bytes))
            .ok_or_else(|| Error::InvalidInput("capture byte size overflows".into()))?;
        metadata.insert(
            name.as_str(),
            TensorInfo {
                dtype: "F32",
                shape: tensor.dims(),
                data_offsets: (offset, end),
            },
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&metadata)?;
    header.extend(std::iter::repeat_n(b' ', (8 - header.len() % 8) % 8));
    let header_len = u64::try_from(header.len())
        .map_err(|_| Error::InvalidInput("capture header is too large".into()))?;
    let total_bytes = u64::try_from(offset)
        .ok()
        .and_then(|data| data.checked_add(header_len))
        .and_then(|bytes| bytes.checked_add(8))
        .ok_or_else(|| Error::InvalidInput("capture byte budget overflows".into()))?;
    if header_len > MAX_HEADER_BYTES || total_bytes > MAX_BYTES {
        return Err(Error::InvalidInput(
            "encoded capture exceeds byte limits".into(),
        ));
    }
    let file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(&header)?;
    let mut buffer = Vec::with_capacity(64 * 1024);
    for tensor in sorted.values() {
        let values = tensor
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for chunk in values.chunks(16 * 1024) {
            buffer.clear();
            for value in chunk {
                buffer.extend_from_slice(&value.to_le_bytes());
            }
            writer.write_all(&buffer)?;
        }
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    static NEXT: AtomicUsize = AtomicUsize::new(0);
    fn scratch_file() -> Result<PathBuf> {
        let root = match std::env::var_os("OSOR_SCRATCH_ROOT") {
            Some(path) => PathBuf::from(path),
            None if Path::new("/Volumes/Scratch").is_dir() => {
                PathBuf::from("/Volumes/Scratch/osor-rs")
            }
            None => {
                return Err(Error::Unavailable(
                    "Scratch is missing; set OSOR_SCRATCH_ROOT on other hosts".into(),
                ));
            }
        };
        let parent = root.join("tmp/capture-tests");
        std::fs::create_dir_all(&parent)?;
        Ok(parent.join(format!(
            "{}-{}.safetensors",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }

    #[test]
    fn bounded_capture_round_trip_and_exclusive_creation() -> Result<()> {
        let path = scratch_file()?;
        let tensors = HashMap::from([(
            "alpha".into(),
            Tensor::new(&[0.25_f32, 0.75], &Device::Cpu)?.to_dtype(DType::BF16)?,
        )]);
        save_new(&tensors, &path)?;
        let loaded = load(&path, &Device::Cpu)?;
        let alpha = loaded
            .get("alpha")
            .ok_or_else(|| Error::InvalidInput("missing alpha".into()))?;
        assert_eq!(alpha.dtype(), DType::F32);
        assert_eq!(alpha.to_vec1::<f32>()?, vec![0.25, 0.75]);
        assert!(save_new(&tensors, &path).is_err());
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn writer_rejects_invalid_metadata_before_creating_output() -> Result<()> {
        let path = scratch_file()?;
        let tensor = Tensor::new(&[1_f32], &Device::Cpu)?;
        for tensors in [
            HashMap::from([("x".repeat(129), tensor.clone())]),
            HashMap::from([("alpha".into(), tensor.to_dtype(DType::F64)?)]),
            HashMap::from([("alpha".into(), tensor.reshape(vec![1; 9])?)]),
        ] {
            assert!(save_new(&tensors, &path).is_err());
            assert!(!path.exists());
        }
        Ok(())
    }
}
