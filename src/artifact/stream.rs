//! Bounded I/O over already-open conversion sources and output writers.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Seek, SeekFrom, Write};

use candle_core::{DType, Device, Tensor};
use safetensors::Dtype as SafeDtype;
use sha2::{Digest, Sha256};

use super::{
    ArtifactLimits, Budget, Header, TensorInfo, allocate_bytes, artifact_error, validate_dtype,
    validate_hash, validate_info, validate_name,
};
use crate::Result;

const CHUNK: usize = 64 * 1024;

type Checkpoint<'a> = &'a mut dyn FnMut(u64, u64) -> Result<()>;

struct InputEntry {
    name: String,
    info: TensorInfo,
    digest: [u8; 32],
}

pub(crate) struct PreparedTensors {
    sha256: String,
    size: u64,
    prefix: [u8; 8],
    header: Vec<u8>,
    entries: Vec<InputEntry>,
    dtype: DType,
}

impl PreparedTensors {
    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }
}

// Callbacks run before each transfer, including the first, and after the last.
fn read_bounded<R: Read + ?Sized>(
    reader: &mut R,
    data: &mut [u8],
    position: &mut u64,
    total: u64,
    checkpoint: Checkpoint<'_>,
) -> Result<()> {
    for chunk in data.chunks_mut(CHUNK) {
        let mut pending = chunk;
        while !pending.is_empty() {
            checkpoint(*position, total)?;
            let count = match reader.read(pending) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => result?,
            };
            if count == 0 {
                return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof).into());
            }
            *position = position
                .checked_add(
                    u64::try_from(count).map_err(|_| artifact_error("read size overflows"))?,
                )
                .ok_or_else(|| artifact_error("read position overflows"))?;
            pending = pending
                .get_mut(count..)
                .ok_or_else(|| artifact_error("invalid read count"))?;
        }
    }
    Ok(())
}

fn total_budget(
    budget: &Budget,
    count: usize,
    bytes: u64,
    limits: &ArtifactLimits,
) -> Result<(usize, u64)> {
    let count = budget
        .tensors
        .checked_add(count)
        .ok_or_else(|| artifact_error("tensor count overflows"))?;
    let bytes = budget
        .tensor_bytes
        .checked_add(bytes)
        .ok_or_else(|| artifact_error("tensor budget overflows"))?;
    if count > limits.max_tensors || bytes > limits.max_total_tensor_bytes {
        return Err(artifact_error("aggregate tensor budget exceeded"));
    }
    Ok((count, bytes))
}

pub(crate) fn prepare<R: Read + Seek + ?Sized>(
    reader: &mut R,
    expected_sha256: &str,
    dtype: DType,
    limits: &ArtifactLimits,
    budget: &mut Budget,
    checkpoint: Checkpoint<'_>,
) -> Result<PreparedTensors> {
    validate_hash(expected_sha256)?;
    validate_dtype(dtype)?;
    let size = reader.seek(SeekFrom::End(0))?;
    if size < 8 || size > limits.max_file_bytes {
        return Err(artifact_error(
            "invalid or oversized safetensors file length",
        ));
    }
    reader.seek(SeekFrom::Start(0))?;
    let mut position = 0;
    let mut prefix = [0; 8];
    read_bounded(reader, &mut prefix, &mut position, size, checkpoint)?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len == 0
        || header_len > limits.max_header_bytes
        || header_len.checked_add(8).is_none_or(|n| n > size)
    {
        return Err(artifact_error("invalid or oversized safetensors header"));
    }
    let mut header = allocate_bytes(
        usize::try_from(header_len)
            .map_err(|_| artifact_error("header too large for this host"))?,
    )?;
    read_bounded(reader, &mut header, &mut position, size, checkpoint)?;
    let Header(tensors) = serde_json::from_slice(&header)
        .map_err(|e| artifact_error(format!("invalid safetensors header: {e}")))?;
    if tensors.is_empty() || tensors.len() > limits.max_tensors {
        return Err(artifact_error(
            "invalid or excessive safetensors tensor count",
        ));
    }
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(tensors.len())
        .map_err(|e| artifact_error(format!("cannot allocate tensor plan: {e}")))?;
    entries.extend(tensors.into_iter().map(|(name, info)| InputEntry {
        name,
        info,
        digest: [0; 32],
    }));
    entries.sort_by_key(|entry| entry.info.data_offsets.0);
    let mut offset = 0_u64;
    let mut decoded = 0_u64;
    for entry in &entries {
        if entry.info.data_offsets.0 != offset {
            return Err(artifact_error(
                "safetensors data offsets overlap or contain gaps",
            ));
        }
        decoded = decoded
            .checked_add(validate_info(&entry.name, &entry.info, limits, dtype)?)
            .ok_or_else(|| artifact_error("decoded tensor budget overflows"))?;
        offset = entry.info.data_offsets.1;
    }
    if offset
        .checked_add(header_len)
        .and_then(|n| n.checked_add(8))
        != Some(size)
    {
        return Err(artifact_error("safetensors offsets do not cover the file"));
    }
    let (count, bytes) = total_budget(budget, entries.len(), decoded, limits)?;

    // Hash the exact header bytes parsed above, then the consecutive tensor bytes.
    let mut hash = Sha256::new();
    hash.update(prefix);
    hash.update(&header);
    let mut scratch = [0_u8; CHUNK];
    for entry in &mut entries {
        let mut tensor_hash = Sha256::new();
        let mut remaining = entry.info.data_offsets.1 - entry.info.data_offsets.0;
        while remaining != 0 {
            let len = usize::try_from(remaining.min(CHUNK as u64))
                .map_err(|_| artifact_error("tensor chunk too large"))?;
            let chunk = scratch
                .get_mut(..len)
                .ok_or_else(|| artifact_error("invalid tensor chunk"))?;
            read_bounded(reader, chunk, &mut position, size, checkpoint)?;
            hash.update(&*chunk);
            tensor_hash.update(&*chunk);
            remaining -= len as u64;
        }
        entry.digest = tensor_hash.finalize().into();
    }
    checkpoint(position, size)?;
    if reader.seek(SeekFrom::End(0))? != size {
        return Err(artifact_error("safetensors size changed while verifying"));
    }
    let sha256 = format!("{:x}", hash.finalize());
    if !sha256.eq_ignore_ascii_case(expected_sha256) {
        return Err(artifact_error("safetensors SHA-256 mismatch"));
    }
    budget.tensors = count;
    budget.tensor_bytes = bytes;
    Ok(PreparedTensors {
        sha256,
        size,
        prefix,
        header,
        entries,
        dtype,
    })
}

pub(crate) fn decode<R: Read + Seek + ?Sized>(
    reader: &mut R,
    plan: &PreparedTensors,
    device: &Device,
    checkpoint: Checkpoint<'_>,
) -> Result<HashMap<String, Tensor>> {
    if reader.seek(SeekFrom::End(0))? != plan.size {
        return Err(artifact_error(
            "safetensors size changed after verification",
        ));
    }
    reader.seek(SeekFrom::Start(0))?;
    let mut position = 0;
    let mut scratch = [0_u8; CHUNK];
    for expected in [plan.prefix.as_slice(), plan.header.as_slice()] {
        for part in expected.chunks(CHUNK) {
            let target = scratch
                .get_mut(..part.len())
                .ok_or_else(|| artifact_error("invalid header chunk"))?;
            read_bounded(reader, target, &mut position, plan.size, checkpoint)?;
            if target != part {
                return Err(artifact_error(
                    "safetensors header changed after verification",
                ));
            }
        }
    }
    let mut result = HashMap::new();
    result
        .try_reserve(plan.entries.len())
        .map_err(|e| artifact_error(format!("cannot allocate tensor map: {e}")))?;
    let mut bytes = Vec::new();
    for entry in &plan.entries {
        let len = usize::try_from(entry.info.data_offsets.1 - entry.info.data_offsets.0)
            .map_err(|_| artifact_error("tensor too large for this host"))?;
        if bytes.len() < len {
            bytes
                .try_reserve_exact(len - bytes.len())
                .map_err(|e| artifact_error(format!("cannot grow tensor read buffer: {e}")))?;
            bytes.resize(len, 0);
        }
        let data = bytes
            .get_mut(..len)
            .ok_or_else(|| artifact_error("invalid tensor buffer"))?;
        read_bounded(reader, data, &mut position, plan.size, checkpoint)?;
        let digest: [u8; 32] = Sha256::digest(&*data).into();
        if digest != entry.digest {
            return Err(artifact_error(format!(
                "tensor {} changed after verification",
                entry.name
            )));
        }
        let source_dtype = DType::try_from(entry.info.dtype)
            .map_err(|e| artifact_error(format!("unsupported tensor dtype: {e}")))?;
        let tensor = Tensor::from_raw_buffer(data, source_dtype, &entry.info.shape, &Device::Cpu)?
            .to_dtype(plan.dtype)?
            .to_device(device)?;
        result.insert(entry.name.clone(), tensor);
    }
    checkpoint(position, plan.size)?;
    if reader.seek(SeekFrom::End(0))? != plan.size {
        return Err(artifact_error("safetensors size changed while decoding"));
    }
    Ok(result)
}

pub(crate) struct PreparedOutput {
    header: Vec<u8>,
    entries: BTreeMap<String, TensorInfo>,
    size: u64,
}

// Bound serialization itself, rather than checking only after an allocation.
struct HeaderWriter {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for HeaderWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let len = self
            .bytes
            .len()
            .checked_add(data.len())
            .ok_or_else(|| std::io::Error::other("output header size overflows"))?;
        if len > self.limit {
            return Err(std::io::Error::other("output header exceeds limit"));
        }
        self.bytes
            .try_reserve(data.len())
            .map_err(std::io::Error::other)?;
        self.bytes.extend_from_slice(data);
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) fn prepare_output(
    tensors: &HashMap<String, Tensor>,
    limits: &ArtifactLimits,
    budget: &mut Budget,
) -> Result<PreparedOutput> {
    validate_dtype(DType::F32)?;
    if tensors.is_empty() || tensors.len() > limits.max_tensors {
        return Err(artifact_error("invalid output tensor count"));
    }
    let sorted: BTreeMap<_, _> = tensors.iter().collect();
    let mut entries = BTreeMap::new();
    let mut offset = 0_u64;
    for (name, tensor) in sorted {
        validate_name(name, limits)?;
        if tensor.dims().is_empty() || tensor.dims().len() > limits.max_rank {
            return Err(artifact_error(format!(
                "{name} has an invalid or oversized shape"
            )));
        }
        let count = tensor
            .dims()
            .iter()
            .try_fold(1_u64, |count, &dim| {
                u64::try_from(dim)
                    .ok()
                    .and_then(|dim| count.checked_mul(dim))
            })
            .ok_or_else(|| artifact_error(format!("{name} shape overflows")))?;
        let end = count
            .checked_mul(4)
            .and_then(|n| offset.checked_add(n))
            .ok_or_else(|| artifact_error("output offsets overflow"))?;
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(tensor.dims().len())
            .map_err(|e| artifact_error(format!("cannot allocate output shape: {e}")))?;
        shape.extend_from_slice(tensor.dims());
        let info = TensorInfo {
            dtype: SafeDtype::F32,
            shape,
            data_offsets: (offset, end),
        };
        validate_info(name, &info, limits, DType::F32)?;
        entries.insert(name.clone(), info);
        offset = end;
    }
    let (count, bytes) = total_budget(budget, entries.len(), offset, limits)?;
    let limit = match usize::try_from(limits.max_header_bytes) {
        Ok(limit) => limit,
        Err(_) => usize::MAX,
    };
    let mut writer = HeaderWriter {
        bytes: Vec::new(),
        limit,
    };
    serde_json::to_writer(&mut writer, &entries)
        .map_err(|e| artifact_error(format!("cannot serialize bounded output header: {e}")))?;
    let padding = (8 - writer.bytes.len() % 8) % 8;
    writer.write_all(
        [b' '; 8]
            .get(..padding)
            .ok_or_else(|| artifact_error("invalid header padding"))?,
    )?;
    let header_len =
        u64::try_from(writer.bytes.len()).map_err(|_| artifact_error("output header too large"))?;
    let size = header_len
        .checked_add(8)
        .and_then(|n| n.checked_add(offset))
        .filter(|&n| n <= limits.max_file_bytes)
        .ok_or_else(|| artifact_error("output exceeds configured limits"))?;
    budget.tensors = count;
    budget.tensor_bytes = bytes;
    Ok(PreparedOutput {
        header: writer.bytes,
        entries,
        size,
    })
}

fn write_bounded<W: Write>(
    writer: &mut W,
    data: &[u8],
    hasher: &mut Sha256,
    position: &mut u64,
    total: u64,
    checkpoint: Checkpoint<'_>,
) -> Result<()> {
    for chunk in data.chunks(CHUNK) {
        let mut pending = chunk;
        while !pending.is_empty() {
            checkpoint(*position, total)?;
            let count = match writer.write(pending) {
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                result => result?,
            };
            if count == 0 {
                return Err(std::io::Error::from(std::io::ErrorKind::WriteZero).into());
            }
            hasher.update(
                pending
                    .get(..count)
                    .ok_or_else(|| artifact_error("invalid write count"))?,
            );
            *position = position
                .checked_add(
                    u64::try_from(count).map_err(|_| artifact_error("write size overflows"))?,
                )
                .ok_or_else(|| artifact_error("write position overflows"))?;
            pending = pending
                .get(count..)
                .ok_or_else(|| artifact_error("invalid write count"))?;
        }
    }
    Ok(())
}

pub(crate) fn write_output<W: Write>(
    writer: &mut W,
    tensors: &HashMap<String, Tensor>,
    plan: &PreparedOutput,
    checkpoint: Checkpoint<'_>,
) -> Result<String> {
    if tensors.len() != plan.entries.len()
        || plan.entries.iter().any(|(name, info)| {
            tensors
                .get(name)
                .is_none_or(|tensor| tensor.dims() != info.shape)
        })
    {
        return Err(artifact_error("output tensors changed after preflight"));
    }
    let mut hasher = Sha256::new();
    let mut position = 0;
    let header_len =
        u64::try_from(plan.header.len()).map_err(|_| artifact_error("output header too large"))?;
    write_bounded(
        writer,
        &header_len.to_le_bytes(),
        &mut hasher,
        &mut position,
        plan.size,
        checkpoint,
    )?;
    write_bounded(
        writer,
        &plan.header,
        &mut hasher,
        &mut position,
        plan.size,
        checkpoint,
    )?;
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(CHUNK)
        .map_err(|e| artifact_error(format!("cannot allocate output buffer: {e}")))?;
    for (name, info) in &plan.entries {
        let tensor = tensors
            .get(name)
            .ok_or_else(|| artifact_error("output tensor missing after preflight"))?
            .flatten_all()?;
        let count = usize::try_from((info.data_offsets.1 - info.data_offsets.0) / 4)
            .map_err(|_| artifact_error("output tensor too large for this host"))?;
        let mut start = 0;
        while start < count {
            let len = (count - start).min(CHUNK / 4);
            let values = tensor
                .narrow(0, start, len)?
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            buffer.clear();
            for value in values {
                buffer.extend_from_slice(&value.to_le_bytes());
            }
            write_bounded(
                writer,
                &buffer,
                &mut hasher,
                &mut position,
                plan.size,
                checkpoint,
            )?;
            start += len;
        }
    }
    checkpoint(position, plan.size)?;
    writer.flush()?;
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use std::io::{self, Cursor};

    fn source(header: &str, data: &[u8]) -> Vec<u8> {
        let mut header = header.as_bytes().to_vec();
        header.extend(std::iter::repeat_n(b' ', (8 - header.len() % 8) % 8));
        let mut result = (header.len() as u64).to_le_bytes().to_vec();
        result.extend(header);
        result.extend(data);
        result
    }
    fn digest(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
    fn sample() -> Vec<u8> {
        source(
            r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &[0, 0, 128, 63, 0, 0, 0, 192],
        )
    }
    fn plan(bytes: &[u8], limits: &ArtifactLimits, budget: &mut Budget) -> Result<PreparedTensors> {
        prepare(
            &mut Cursor::new(bytes),
            &digest(bytes),
            DType::F32,
            limits,
            budget,
            &mut |_, _| Ok(()),
        )
    }
    #[test]
    fn canonical_roundtrip_and_hash() -> Result<()> {
        let bytes = sample();
        let limits = ArtifactLimits::default();
        let plan = plan(&bytes, &limits, &mut Budget::default())?;
        assert_eq!(plan.sha256(), digest(&bytes));
        let tensors = decode(
            &mut Cursor::new(&bytes),
            &plan,
            &Device::Cpu,
            &mut |_, _| Ok(()),
        )?;
        assert_eq!(
            tensors
                .get("weight")
                .ok_or_else(|| artifact_error("missing weight"))?
                .to_vec1::<f32>()?,
            vec![1.0, -2.0]
        );
        let out = prepare_output(&tensors, &limits, &mut Budget::default())?;
        let mut result = Vec::new();
        let hash = write_output(&mut result, &tensors, &out, &mut |_, _| Ok(()))?;
        assert_eq!(result, bytes);
        assert_eq!(hash, digest(&bytes));
        Ok(())
    }
    #[test]
    fn bad_pins_and_changed_streams_are_rejected() -> Result<()> {
        let bytes = sample();
        let limits = ArtifactLimits::default();
        for hash in ["bad".to_string(), "0".repeat(64)] {
            let mut budget = Budget::default();
            assert!(
                prepare(
                    &mut Cursor::new(&bytes),
                    &hash,
                    DType::F32,
                    &limits,
                    &mut budget,
                    &mut |_, _| Ok(())
                )
                .is_err()
            );
            assert_eq!(budget.tensors, 0);
        }
        let prepared = plan(&bytes, &limits, &mut Budget::default())?;
        let upper = digest(&bytes).to_ascii_uppercase();
        prepare(
            &mut Cursor::new(&bytes),
            &upper,
            DType::F32,
            &limits,
            &mut Budget::default(),
            &mut |_, _| Ok(()),
        )?;
        let mut changed = bytes.clone();
        let last = changed.len() - 1;
        changed[last] ^= 1;
        assert!(
            decode(
                &mut Cursor::new(changed),
                &prepared,
                &Device::Cpu,
                &mut |_, _| Ok(())
            )
            .is_err()
        );
        let mut changed = bytes.clone();
        changed[10] ^= 1;
        assert!(
            decode(
                &mut Cursor::new(changed),
                &prepared,
                &Device::Cpu,
                &mut |_, _| Ok(())
            )
            .is_err()
        );
        let mut changed = bytes.clone();
        changed.push(0);
        assert!(
            decode(
                &mut Cursor::new(changed),
                &prepared,
                &Device::Cpu,
                &mut |_, _| Ok(())
            )
            .is_err()
        );
        assert!(
            decode(
                &mut Cursor::new(&bytes[..bytes.len() - 1]),
                &prepared,
                &Device::Cpu,
                &mut |_, _| Ok(())
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    fn rejects_invalid_layouts_and_aggregate_expansion() -> Result<()> {
        let limits = ArtifactLimits::default();
        for bytes in [
            source(
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[1,5]}}"#,
                &[0; 5],
            ),
            source(
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                &[0; 4],
            ),
            source(
                r#"{"__metadata__":{},"__metadata__":{},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                &[0; 4],
            ),
            source(
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                &[0; 5],
            ),
            source(
                r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
                &[0; 3],
            ),
        ] {
            assert!(plan(&bytes, &limits, &mut Budget::default()).is_err());
        }
        let f16 = source(
            r#"{"x":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        let limits = ArtifactLimits {
            max_tensor_bytes: 7,
            ..limits
        };
        assert!(plan(&f16, &limits, &mut Budget::default()).is_err());
        let limits = ArtifactLimits {
            max_tensor_bytes: 8,
            max_total_tensor_bytes: 12,
            ..limits
        };
        let mut budget = Budget::default();
        plan(&f16, &limits, &mut budget)?;
        assert!(plan(&f16, &limits, &mut budget).is_err());
        let count_limits = ArtifactLimits {
            max_tensors: 1,
            max_total_tensor_bytes: 16,
            ..limits.clone()
        };
        let mut count_budget = Budget::default();
        plan(&f16, &count_limits, &mut count_budget)?;
        assert!(plan(&f16, &count_limits, &mut count_budget).is_err());
        let tensors = HashMap::from([(
            "x".to_string(),
            Tensor::zeros((2,), DType::F32, &Device::Cpu)?,
        )]);
        let mut output_budget = Budget::default();
        prepare_output(&tensors, &limits, &mut output_budget)?;
        assert!(prepare_output(&tensors, &limits, &mut output_budget).is_err());
        Ok(())
    }
    struct FailingWriter;
    impl Write for FailingWriter {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("injected writer failure"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn cancellation_and_writer_failures_propagate() -> Result<()> {
        let bytes = sample();
        let limits = ArtifactLimits::default();
        let cancelled = &mut |_, _| Err(Error::Cancelled);
        assert!(matches!(
            prepare(
                &mut Cursor::new(&bytes),
                &digest(&bytes),
                DType::F32,
                &limits,
                &mut Budget::default(),
                cancelled
            ),
            Err(Error::Cancelled)
        ));
        let prepared = plan(&bytes, &limits, &mut Budget::default())?;
        assert!(matches!(
            decode(&mut Cursor::new(&bytes), &prepared, &Device::Cpu, cancelled),
            Err(Error::Cancelled)
        ));
        let tensors = decode(
            &mut Cursor::new(&bytes),
            &prepared,
            &Device::Cpu,
            &mut |_, _| Ok(()),
        )?;
        let out = prepare_output(&tensors, &limits, &mut Budget::default())?;
        assert!(matches!(
            write_output(&mut Vec::new(), &tensors, &out, cancelled),
            Err(Error::Cancelled)
        ));
        assert!(matches!(
            write_output(&mut FailingWriter, &tensors, &out, &mut |_, _| Ok(())),
            Err(Error::Io(_))
        ));
        Ok(())
    }

    #[test]
    fn cancellation_after_bounded_data_transfer() -> Result<()> {
        let data = vec![0; CHUNK + 4];
        let bytes = source(
            r#"{"x":{"dtype":"F32","shape":[16385],"data_offsets":[0,65540]}}"#,
            &data,
        );
        let limits = ArtifactLimits::default();
        let data_start = u64::try_from(bytes.len() - data.len())
            .map_err(|_| artifact_error("fixture too large"))?;
        let mut stop = |position, _| {
            if position > data_start {
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        };
        let mut budget = Budget::default();
        assert!(matches!(
            prepare(
                &mut Cursor::new(&bytes),
                &digest(&bytes),
                DType::F32,
                &limits,
                &mut budget,
                &mut stop
            ),
            Err(Error::Cancelled)
        ));
        assert_eq!(budget.tensors, 0);
        let prepared = plan(&bytes, &limits, &mut budget)?;
        assert!(matches!(
            decode(&mut Cursor::new(&bytes), &prepared, &Device::Cpu, &mut stop),
            Err(Error::Cancelled)
        ));
        let tensors = decode(
            &mut Cursor::new(&bytes),
            &prepared,
            &Device::Cpu,
            &mut |_, _| Ok(()),
        )?;
        let out = prepare_output(&tensors, &limits, &mut Budget::default())?;
        let mut result = Vec::new();
        assert!(matches!(
            write_output(&mut result, &tensors, &out, &mut stop),
            Err(Error::Cancelled)
        ));
        assert_eq!(
            result.len(),
            usize::try_from(data_start).map_err(|_| artifact_error("fixture too large"))? + CHUNK
        );
        Ok(())
    }
}
