//! Local, bounded safetensors bundles and their provenance.
//!
//! Bundle files have fixed names. The digest is calculated over the same bytes
//! used to construct each tensor, before the bundle is returned to the caller.

pub(crate) mod stream;

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use safetensors::Dtype as SafeDtype;
use serde::de::{IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

use crate::scheduler::{Scheduler, SchedulerConfig};
use crate::{Error, Result, UPSTREAM_REVISION};

/// Pipeline identifier recorded in version 1 manifests.
pub const PIPELINE: &str = "osor-sdxlinpainting-phase2";
/// Bundle file holding merged UNet weights.
pub const UNET_FILE: &str = "unet.safetensors";
/// Bundle file holding VAE weights.
pub const VAE_FILE: &str = "vae.safetensors";
/// Bundle file holding the fixed prompt embeddings.
pub const PROMPT_FILE: &str = "prompt.safetensors";
/// Bundle manifest file, published last.
pub const MANIFEST_FILE: &str = "manifest.json";

/// An immutable upstream repository revision and its license identifier or URL.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceProvenance {
    /// Source repository URL or identifier.
    pub repository: String,
    /// A full 40- or 64-character hexadecimal commit/object identifier.
    pub revision: String,
    /// License identifier or URL for this source.
    pub license: String,
}

/// SHA-256 values of the resulting bundle files, in hexadecimal.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BundleDigests {
    /// SHA-256 of `unet.safetensors`.
    pub unet: String,
    /// SHA-256 of `vae.safetensors`.
    pub vae: String,
    /// SHA-256 of `prompt.safetensors`.
    pub prompt: String,
}

/// SHA-256 values of the exact exported inputs used for conversion.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InputDigests {
    /// SHA-256 of the exported base UNet.
    pub base_unet: String,
    /// SHA-256 of the exported base VAE.
    pub base_vae: String,
    /// SHA-256 of the exported Phase II checkpoint.
    pub checkpoint: String,
    /// SHA-256 of the exported prompt embeddings.
    pub prompt: String,
}

/// Self-contained description of the pinned Phase II bundle.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Manifest schema; must be 1.
    pub schema_version: u32,
    /// Pinned upstream OSOR code revision.
    pub upstream_revision: String,
    /// Must equal [`PIPELINE`].
    pub pipeline: String,
    /// Base SDXL inpainting model provenance.
    pub base_model: SourceProvenance,
    /// VAE provenance.
    pub vae_model: SourceProvenance,
    /// Phase II checkpoint provenance.
    pub checkpoint: SourceProvenance,
    /// Fixed prompt embedding provenance.
    pub prompt: SourceProvenance,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional configuration provenance.
    pub config: Option<SourceProvenance>,
    /// Digests of the exact conversion inputs.
    pub input_sha256: InputDigests,
    /// Digests of the bundle tensor files.
    pub sha256: BundleDigests,
    /// Complete DDPM scheduler configuration.
    pub scheduler: SchedulerConfig,
    /// Pinned timestep; must be 400.
    pub timestep: usize,
    /// Pinned VAE scaling factor; must be 0.13025.
    pub vae_scaling_factor: f64,
    /// Merged LoRA rank; must be 256.
    pub adapter_rank: usize,
    /// Merged LoRA alpha; must be 256.
    pub adapter_alpha: f64,
}

/// Limits apply before tensor allocation. Applications may lower or raise them.
#[derive(Clone, Debug)]
pub struct ArtifactLimits {
    /// Largest manifest file (default 1 MiB).
    pub max_manifest_bytes: u64,
    /// Largest safetensors file (default 24 GiB).
    pub max_file_bytes: u64,
    /// Largest safetensors JSON header (default 16 MiB).
    pub max_header_bytes: u64,
    /// Most tensors across the files of one load (default 30,000).
    pub max_tensors: usize,
    /// Largest single tensor in source or decoded bytes (default 1 GiB).
    pub max_tensor_bytes: u64,
    /// Aggregate decoded tensor bytes across files (default 32 GiB); not a peak-memory bound.
    pub max_total_tensor_bytes: u64,
    /// Largest tensor rank (default 8).
    pub max_rank: usize,
    /// Largest single dimension (default 1,000,000).
    pub max_dimension: usize,
    /// Longest tensor name (default 512 bytes).
    pub max_name_bytes: usize,
}

impl Default for ArtifactLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: 1024 * 1024,
            max_file_bytes: 24 * 1024 * 1024 * 1024,
            max_header_bytes: 16 * 1024 * 1024,
            max_tensors: 30_000,
            max_tensor_bytes: 1024 * 1024 * 1024,
            max_total_tensor_bytes: 32 * 1024 * 1024 * 1024,
            max_rank: 8,
            max_dimension: 1_000_000,
            max_name_bytes: 512,
        }
    }
}

/// Loaded tensor maps use unprefixed diffusers names.
pub struct ModelBundle {
    /// Validated manifest.
    pub manifest: Manifest,
    /// Merged UNet tensors by diffusers name.
    pub unet: HashMap<String, Tensor>,
    /// VAE tensors by diffusers name.
    pub vae: HashMap<String, Tensor>,
    /// Fixed prompt embeddings `[1, 77, 2048]`.
    pub sequence: Tensor,
    /// Fixed pooled prompt embedding `[1, 1280]`.
    pub pooled: Tensor,
}

impl ModelBundle {
    /// Load a pinned bundle with caller-specified artifact limits.
    /// The aggregate tensor limit counts decoded bytes in `dtype` across all
    /// bundle files; it is not a bound on peak memory use.
    pub fn load_with_limits_and_manifest_sha256(
        path: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
        limits: &ArtifactLimits,
        expected_hash: &str,
    ) -> Result<Self> {
        Self::load_internal(path.as_ref(), device, dtype, limits, Some(expected_hash))
    }

    /// Load a bundle only when the exact manifest bytes match a caller-trusted SHA-256.
    pub fn load_with_manifest_sha256(
        path: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
        expected_hash: &str,
    ) -> Result<Self> {
        Self::load_with_limits_and_manifest_sha256(
            path,
            device,
            dtype,
            &ArtifactLimits::default(),
            expected_hash,
        )
    }

    /// Load and verify a bundle with default limits and no manifest pin.
    pub fn load(path: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        Self::load_with_limits(path, device, dtype, &ArtifactLimits::default())
    }

    /// Load and verify a bundle with caller limits and no manifest pin.
    pub fn load_with_limits(
        path: impl AsRef<Path>,
        device: &Device,
        dtype: DType,
        limits: &ArtifactLimits,
    ) -> Result<Self> {
        Self::load_internal(path.as_ref(), device, dtype, limits, None)
    }

    fn load_internal(
        path: &Path,
        device: &Device,
        dtype: DType,
        limits: &ArtifactLimits,
        expected_hash: Option<&str>,
    ) -> Result<Self> {
        if let Some(hash) = expected_hash {
            validate_hash(hash)?;
        }
        validate_dtype(dtype)?;
        let root = bundle_root(path)?;
        let manifest_path = bundle_file(&root, MANIFEST_FILE)?;
        let bytes = read_small_file(&manifest_path, limits.max_manifest_bytes)?;
        if let Some(hash) = expected_hash {
            let actual = format!("{:x}", Sha256::digest(&bytes));
            if !actual.eq_ignore_ascii_case(hash) {
                return Err(artifact_error("manifest SHA256 mismatch"));
            }
        }
        let manifest: Manifest = serde_json::from_slice(&bytes)
            .map_err(|e| artifact_error(format!("invalid manifest: {e}")))?;
        let document: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| artifact_error(format!("invalid manifest: {e}")))?;
        let scheduler_input = document
            .get("scheduler")
            .ok_or_else(|| artifact_error("missing scheduler configuration"))?;
        reject_unknown_scheduler_fields(
            scheduler_input,
            &serde_json::to_value(&manifest.scheduler)?,
            "scheduler",
        )?;
        manifest.validate()?;

        let mut budget = Budget::default();
        let unet_path = bundle_file(&root, UNET_FILE)?;
        let (unet, _) = read_tensors(
            &unet_path,
            device,
            dtype,
            limits,
            &mut budget,
            Some(&manifest.sha256.unet),
        )?;
        validate_unet_contract(&unet)?;
        let vae_path = bundle_file(&root, VAE_FILE)?;
        let (vae, _) = read_tensors(
            &vae_path,
            device,
            dtype,
            limits,
            &mut budget,
            Some(&manifest.sha256.vae),
        )?;
        validate_plain_names(&vae, "VAE")?;
        if vae.is_empty() {
            return Err(artifact_error("VAE has no tensors"));
        }
        let prompt_path = bundle_file(&root, PROMPT_FILE)?;
        let (mut prompt, _) = read_tensors(
            &prompt_path,
            device,
            dtype,
            limits,
            &mut budget,
            Some(&manifest.sha256.prompt),
        )?;
        validate_prompt(&prompt)?;
        let sequence = prompt
            .remove("prompt_embeds")
            .ok_or_else(|| artifact_error("missing prompt_embeds"))?;
        let pooled = prompt
            .remove("pooled_prompt_embeds")
            .ok_or_else(|| artifact_error("missing pooled_prompt_embeds"))?;
        Ok(Self {
            manifest,
            unet,
            vae,
            sequence,
            pooled,
        })
    }
}

fn reject_unknown_scheduler_fields(
    input: &serde_json::Value,
    known: &serde_json::Value,
    path: &str,
) -> Result<()> {
    match (input, known) {
        (serde_json::Value::Object(input), serde_json::Value::Object(known)) => {
            for (key, value) in input {
                let field = known.get(key).ok_or_else(|| {
                    artifact_error(format!("unknown manifest field {path}.{key}"))
                })?;
                reject_unknown_scheduler_fields(value, field, &format!("{path}.{key}"))?;
            }
        }
        (serde_json::Value::Array(input), serde_json::Value::Array(known)) => {
            for (i, (value, field)) in input.iter().zip(known.iter()).enumerate() {
                reject_unknown_scheduler_fields(value, field, &format!("{path}[{i}]"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

impl Manifest {
    /// Check the schema, pinned recipe constants, provenance and digests.
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || self.upstream_revision != UPSTREAM_REVISION
            || self.pipeline != PIPELINE
        {
            return Err(artifact_error(
                "unsupported schema, upstream revision, or pipeline",
            ));
        }
        self.base_model.validate("base model")?;
        self.vae_model.validate("VAE model")?;
        self.checkpoint.validate("checkpoint")?;
        self.prompt.validate("prompt")?;
        if let Some(config) = &self.config {
            config.validate("config")?;
        }
        for digest in [
            &self.sha256.unet,
            &self.sha256.vae,
            &self.sha256.prompt,
            &self.input_sha256.base_unet,
            &self.input_sha256.base_vae,
            &self.input_sha256.checkpoint,
            &self.input_sha256.prompt,
        ] {
            validate_hash(digest)?;
        }
        if self.timestep != 400
            || !self.vae_scaling_factor.is_finite()
            || self.vae_scaling_factor != 0.13025
            || self.adapter_rank != 256
            || !self.adapter_alpha.is_finite()
            || self.adapter_alpha != 256.0
        {
            return Err(artifact_error(
                "unsupported timestep, VAE scaling factor, or adapter contract",
            ));
        }
        // A round trip also rejects non-finite mandatory scheduler numbers: JSON
        // represents them as null, which cannot deserialize back into a number.
        let scheduler = serde_json::to_vec(&self.scheduler)?;
        let _: SchedulerConfig = serde_json::from_slice(&scheduler)
            .map_err(|e| artifact_error(format!("invalid scheduler configuration: {e}")))?;
        if self.scheduler.num_train_timesteps <= self.timestep {
            return Err(artifact_error("scheduler does not include timestep 400"));
        }
        Scheduler::new(self.scheduler.clone())
            .map_err(|e| artifact_error(format!("unsupported scheduler configuration: {e}")))?;
        Ok(())
    }
}

impl SourceProvenance {
    pub(crate) fn validate(&self, label: &str) -> Result<()> {
        if self.repository.trim().is_empty()
            || self.repository.len() > 2048
            || self.license.trim().is_empty()
            || self.license.len() > 2048
            || !matches!(self.revision.len(), 40 | 64)
            || !self.revision.bytes().all(|b| b.is_ascii_hexdigit())
        {
            return Err(artifact_error(format!(
                "{label} requires a repository, immutable full revision, and license"
            )));
        }
        Ok(())
    }
}

pub(crate) fn artifact_error(message: impl Into<String>) -> Error {
    Error::InvalidArtifact(message.into())
}

fn validate_hash(value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(artifact_error("SHA-256 must be 64 hexadecimal characters"));
    }
    Ok(())
}

pub(crate) fn validate_dtype(dtype: DType) -> Result<()> {
    if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32) {
        return Err(artifact_error(
            "only F16, BF16, and F32 model tensors are supported",
        ));
    }
    if cfg!(target_endian = "big") {
        return Err(artifact_error(
            "safetensors loading requires a little-endian host",
        ));
    }
    Ok(())
}

fn check_path(path: &Path) -> Result<()> {
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(artifact_error(
            "parent-directory path components are not allowed",
        ));
    }
    Ok(())
}

fn bundle_root(path: &Path) -> Result<PathBuf> {
    check_path(path)?;
    let root = fs::canonicalize(path)?;
    if !fs::metadata(&root)?.is_dir() {
        return Err(artifact_error("bundle path is not a directory"));
    }
    Ok(root)
}

fn bundle_file(root: &Path, name: &str) -> Result<PathBuf> {
    // All names are compile-time constants, never manifest-controlled paths.
    let path = root.join(name);
    let metadata = fs::symlink_metadata(&path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(artifact_error(format!(
            "{name} must be a regular file, not a symlink"
        )));
    }
    let resolved = fs::canonicalize(&path)?;
    if resolved.parent() != Some(root) {
        return Err(artifact_error(format!(
            "{name} escapes the bundle directory"
        )));
    }
    Ok(resolved)
}

pub(crate) fn checked_input_file(path: &Path) -> Result<PathBuf> {
    check_path(path)?;
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(artifact_error(format!(
            "{} must be a regular file, not a symlink",
            path.display()
        )));
    }
    Ok(fs::canonicalize(path)?)
}

fn allocate_bytes(size: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|e| artifact_error(format!("cannot allocate bounded tensor buffer: {e}")))?;
    bytes.resize(size, 0);
    Ok(bytes)
}

fn open_regular_file(path: &Path) -> Result<File> {
    let before = fs::symlink_metadata(path)?;
    if before.file_type().is_symlink() || !before.is_file() {
        return Err(artifact_error(format!(
            "{} must be a regular file, not a symlink",
            path.display()
        )));
    }
    let mut options = OpenOptions::new();
    options.read(true);
    // Open the reparse point itself on Windows so a last-component symlink
    // cannot be followed if it is swapped between the metadata check and open.
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options.open(path)?;
    let opened = file.metadata()?;
    if opened.file_type().is_symlink() || !opened.is_file() {
        return Err(artifact_error(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != opened.dev() || before.ino() != opened.ino() {
            return Err(artifact_error(format!(
                "{} changed while opening",
                path.display()
            )));
        }
    }
    Ok(file)
}

fn read_small_file(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = open_regular_file(path)?;
    let size = file.metadata()?.len();
    if size > limit {
        return Err(artifact_error("manifest exceeds the configured size limit"));
    }
    let cap =
        usize::try_from(size).map_err(|_| artifact_error("manifest is too large for this host"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(cap)
        .map_err(|e| artifact_error(format!("cannot allocate manifest: {e}")))?;
    file.take(limit.saturating_add(1)).read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).map_err(|_| artifact_error("manifest too large"))? > limit {
        return Err(artifact_error("manifest exceeds the configured size limit"));
    }
    Ok(bytes)
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TensorInfo {
    dtype: SafeDtype,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

struct Header(BTreeMap<String, TensorInfo>);

impl<'de> Deserialize<'de> for Header {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = Header;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a safetensors header object")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Header, M::Error> {
                let mut tensors = BTreeMap::new();
                let mut saw_metadata = false;
                while let Some(name) = map.next_key::<String>()? {
                    if name == "__metadata__" {
                        if saw_metadata {
                            return Err(serde::de::Error::custom("duplicate metadata"));
                        }
                        saw_metadata = true;
                        map.next_value::<IgnoredAny>()?;
                    } else {
                        if tensors.contains_key(&name) {
                            return Err(serde::de::Error::custom("duplicate tensor name"));
                        }
                        tensors.insert(name, map.next_value()?);
                    }
                }
                Ok(Header(tensors))
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
    }
}

#[derive(Default)]
pub(crate) struct Budget {
    tensors: usize,
    tensor_bytes: u64,
}

fn validate_name(name: &str, limits: &ArtifactLimits) -> Result<()> {
    if name.is_empty()
        || name.len() > limits.max_name_bytes
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        || name.split('.').any(str::is_empty)
    {
        return Err(artifact_error(format!("invalid tensor name {name:?}")));
    }
    Ok(())
}

fn validate_info(
    name: &str,
    info: &TensorInfo,
    limits: &ArtifactLimits,
    dtype: DType,
) -> Result<u64> {
    validate_name(name, limits)?;
    if !matches!(
        info.dtype,
        SafeDtype::F16 | SafeDtype::BF16 | SafeDtype::F32
    ) {
        return Err(artifact_error(format!("{name} has an unsupported dtype")));
    }
    if info.shape.is_empty()
        || info.shape.len() > limits.max_rank
        || info
            .shape
            .iter()
            .any(|&dim| dim == 0 || dim > limits.max_dimension)
    {
        return Err(artifact_error(format!(
            "{name} has an invalid or oversized shape"
        )));
    }
    let count = info
        .shape
        .iter()
        .try_fold(1_u64, |count, &dim| {
            u64::try_from(dim)
                .ok()
                .and_then(|dim| count.checked_mul(dim))
        })
        .ok_or_else(|| artifact_error(format!("{name} shape overflows")))?;
    let source_bytes = count
        .checked_mul(info.dtype.size() as u64)
        .ok_or_else(|| artifact_error(format!("{name} byte count overflows")))?;
    let decoded_bytes = count
        .checked_mul(dtype.size_in_bytes() as u64)
        .ok_or_else(|| artifact_error(format!("{name} decoded byte count overflows")))?;
    if source_bytes > limits.max_tensor_bytes
        || decoded_bytes > limits.max_tensor_bytes
        || info.data_offsets.1.checked_sub(info.data_offsets.0) != Some(source_bytes)
    {
        return Err(artifact_error(format!(
            "{name} has invalid offsets or exceeds the tensor limit"
        )));
    }
    Ok(decoded_bytes)
}

/// Tensors at least this large are hashed on a helper thread while the
/// tensor is constructed; smaller ones are not worth a thread.
const PARALLEL_HASH_BYTES: usize = 1024 * 1024;

fn tensor_from_bytes(
    data: &[u8],
    source_dtype: DType,
    shape: &[usize],
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    Ok(if dtype.size_in_bytes() < source_dtype.size_in_bytes() {
        // Narrow on CPU before upload: avoid transferring the larger source
        // format and keeping a temporary full-precision GPU allocation.
        Tensor::from_raw_buffer(data, source_dtype, shape, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?
    } else {
        Tensor::from_raw_buffer(data, source_dtype, shape, device)?.to_dtype(dtype)?
    })
}

/// Reads and hashes the exact bytes used for tensors without mmap or pickle.
pub(crate) fn read_tensors(
    path: &Path,
    device: &Device,
    dtype: DType,
    limits: &ArtifactLimits,
    budget: &mut Budget,
    expected_hash: Option<&str>,
) -> Result<(HashMap<String, Tensor>, String)> {
    validate_dtype(dtype)?;
    let file = open_regular_file(path)?;
    let size = file.metadata()?.len();
    if size < 8 || size > limits.max_file_bytes {
        return Err(artifact_error(format!(
            "{} has an invalid or oversized file length",
            path.display()
        )));
    }
    let mut reader = BufReader::new(file);
    let mut prefix = [0_u8; 8];
    reader.read_exact(&mut prefix)?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len == 0
        || header_len > limits.max_header_bytes
        || header_len.checked_add(8).is_none_or(|n| n > size)
    {
        return Err(artifact_error("invalid or oversized safetensors header"));
    }
    let header_size = usize::try_from(header_len)
        .map_err(|_| artifact_error("header too large for this host"))?;
    let mut header_bytes = allocate_bytes(header_size)?;
    reader.read_exact(&mut header_bytes)?;
    let Header(tensors) = serde_json::from_slice(&header_bytes)
        .map_err(|e| artifact_error(format!("invalid safetensors header: {e}")))?;
    if tensors.is_empty() || tensors.len() > limits.max_tensors {
        return Err(artifact_error(
            "invalid or excessive safetensors tensor count",
        ));
    }
    let mut entries: Vec<_> = tensors.into_iter().collect();
    entries.sort_by_key(|(_, info)| info.data_offsets.0);
    let mut offset = 0_u64;
    let mut decoded = 0_u64;
    for (name, info) in &entries {
        if info.data_offsets.0 != offset {
            return Err(artifact_error(
                "safetensors data offsets overlap or contain gaps",
            ));
        }
        decoded = decoded
            .checked_add(validate_info(name, info, limits, dtype)?)
            .ok_or_else(|| artifact_error("decoded tensor budget overflows"))?;
        offset = info.data_offsets.1;
    }
    if offset
        .checked_add(header_len)
        .and_then(|n| n.checked_add(8))
        != Some(size)
    {
        return Err(artifact_error("safetensors offsets do not cover the file"));
    }
    let total_count = budget
        .tensors
        .checked_add(entries.len())
        .ok_or_else(|| artifact_error("tensor count overflows"))?;
    let total_bytes = budget
        .tensor_bytes
        .checked_add(decoded)
        .ok_or_else(|| artifact_error("tensor budget overflows"))?;
    if total_count > limits.max_tensors || total_bytes > limits.max_total_tensor_bytes {
        return Err(artifact_error("aggregate tensor budget exceeded"));
    }
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(&header_bytes);
    let mut result = HashMap::new();
    result
        .try_reserve(entries.len())
        .map_err(|e| artifact_error(format!("cannot allocate tensor map: {e}")))?;
    // Reuse one bounded read buffer across tensors instead of allocating and
    // releasing a new byte vector for every model parameter.
    let mut bytes = Vec::new();
    for (name, info) in entries {
        let byte_count = info
            .data_offsets
            .1
            .checked_sub(info.data_offsets.0)
            .ok_or_else(|| artifact_error("invalid tensor offsets"))?;
        let count = usize::try_from(byte_count)
            .map_err(|_| artifact_error("tensor too large for this host"))?;
        if bytes.len() < count {
            bytes
                .try_reserve_exact(count - bytes.len())
                .map_err(|e| artifact_error(format!("cannot grow tensor read buffer: {e}")))?;
            bytes.resize(count, 0);
        }
        let data = bytes
            .get_mut(..count)
            .ok_or_else(|| artifact_error("tensor read buffer is shorter than validated size"))?;
        reader.read_exact(data)?;
        let data: &[u8] = data;
        let source_dtype = DType::try_from(info.dtype)
            .map_err(|e| artifact_error(format!("unsupported tensor dtype: {e}")))?;
        let tensor = if count >= PARALLEL_HASH_BYTES {
            // Hash and construct from the same immutable bytes concurrently;
            // the digest still covers exactly the bytes used for the tensor.
            std::thread::scope(|scope| {
                let hashing = scope.spawn(|| hasher.update(data));
                let tensor = tensor_from_bytes(data, source_dtype, &info.shape, device, dtype);
                hashing
                    .join()
                    .map_err(|_| artifact_error("tensor hashing thread failed"))?;
                tensor
            })?
        } else {
            hasher.update(data);
            tensor_from_bytes(data, source_dtype, &info.shape, device, dtype)?
        };
        result.insert(name, tensor);
    }
    if reader.get_ref().metadata()?.len() != size {
        return Err(artifact_error("safetensors file changed while reading"));
    }
    let hash = format!("{:x}", hasher.finalize());
    if expected_hash.is_some_and(|expected| !hash.eq_ignore_ascii_case(expected)) {
        return Err(artifact_error(format!(
            "SHA-256 mismatch for {}",
            path.display()
        )));
    }
    budget.tensors = total_count;
    budget.tensor_bytes = total_bytes;
    Ok((result, hash))
}

pub(crate) fn validate_plain_names(tensors: &HashMap<String, Tensor>, label: &str) -> Result<()> {
    if tensors.keys().any(|name| {
        name.starts_with("unet.")
            || name.starts_with("module.")
            || name.starts_with("_orig_mod.")
            || name.contains(".base_layer.")
            || name.contains(".lora_")
    }) {
        return Err(artifact_error(format!(
            "{label} tensor names must be plain diffusers names"
        )));
    }
    Ok(())
}

pub(crate) fn require_shape(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    shape: &[usize],
) -> Result<()> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| artifact_error(format!("missing critical tensor {name}")))?;
    if tensor.dims() != shape {
        return Err(artifact_error(format!(
            "{name} has shape {:?}, expected {shape:?}",
            tensor.dims()
        )));
    }
    Ok(())
}

pub(crate) fn validate_unet_contract(tensors: &HashMap<String, Tensor>) -> Result<()> {
    validate_plain_names(tensors, "UNet")?;
    require_shape(tensors, "conv_in.weight", &[320, 9, 3, 3])?;
    require_shape(tensors, "conv_out.weight", &[5, 320, 3, 3])?;
    require_shape(tensors, "conv_out.bias", &[5])
}

pub(crate) fn validate_prompt(tensors: &HashMap<String, Tensor>) -> Result<()> {
    if tensors.len() != 2 {
        return Err(artifact_error(
            "prompt file must contain exactly the two fixed embeddings",
        ));
    }
    require_shape(tensors, "prompt_embeds", &[1, 77, 2048])?;
    require_shape(tensors, "pooled_prompt_embeds", &[1, 1280])
}

/// Writes F32 safetensors without Candle's infallible `View::data` conversion.
/// The destination is created exclusively; its exact bytes are hashed as written.
pub(crate) fn write_tensors(
    path: &Path,
    tensors: &HashMap<String, Tensor>,
    limits: &ArtifactLimits,
) -> Result<String> {
    if tensors.is_empty() || tensors.len() > limits.max_tensors {
        return Err(artifact_error("invalid output tensor count"));
    }
    let sorted: BTreeMap<_, _> = tensors.iter().collect();
    let mut entries = BTreeMap::new();
    let mut offset = 0_u64;
    for (name, tensor) in &sorted {
        validate_name(name, limits)?;
        let count = tensor
            .dims()
            .iter()
            .try_fold(1_u64, |count, &dim| {
                u64::try_from(dim)
                    .ok()
                    .and_then(|dim| count.checked_mul(dim))
            })
            .ok_or_else(|| artifact_error(format!("{name} shape overflows")))?;
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| artifact_error("output byte count overflows"))?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| artifact_error("output offsets overflow"))?;
        let info = TensorInfo {
            dtype: SafeDtype::F32,
            shape: tensor.dims().to_vec(),
            data_offsets: (offset, end),
        };
        validate_info(name, &info, limits, DType::F32)?;
        entries.insert((*name).clone(), info);
        offset = end;
    }
    let mut header = serde_json::to_vec(&entries)?;
    let padding = (8 - header.len() % 8) % 8;
    header.extend(std::iter::repeat_n(b' ', padding));
    let header_len =
        u64::try_from(header.len()).map_err(|_| artifact_error("output header too large"))?;
    if header_len > limits.max_header_bytes
        || offset > limits.max_total_tensor_bytes
        || header_len
            .checked_add(8)
            .and_then(|n| n.checked_add(offset))
            .is_none_or(|n| n > limits.max_file_bytes)
    {
        return Err(artifact_error("output exceeds configured limits"));
    }
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    let mut hasher = Sha256::new();
    write_hashed(&mut writer, &mut hasher, &header_len.to_le_bytes())?;
    write_hashed(&mut writer, &mut hasher, &header)?;
    let mut buffer = Vec::with_capacity(64 * 1024);
    for (_, tensor) in sorted {
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
            write_hashed(&mut writer, &mut hasher, &buffer)?;
        }
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(format!("{:x}", hasher.finalize()))
}

fn write_hashed(writer: &mut impl Write, hasher: &mut Sha256, bytes: &[u8]) -> Result<()> {
    writer.write_all(bytes)?;
    hasher.update(bytes);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::{Seek, SeekFrom};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    static NEXT_DIR: AtomicUsize = AtomicUsize::new(0);

    struct ScratchDir(PathBuf);

    impl ScratchDir {
        fn new() -> Result<Self> {
            let root = match std::env::var_os("OSOR_SCRATCH_ROOT") {
                Some(path) => PathBuf::from(path),
                None => {
                    if !Path::new("/Volumes/Scratch").is_dir() {
                        return Err(artifact_error(
                            "Scratch volume is not mounted; set OSOR_SCRATCH_ROOT on other hosts",
                        ));
                    }
                    PathBuf::from("/Volumes/Scratch/osor-rs")
                }
            };
            let parent = root.join("tmp/artifact-tests");
            fs::create_dir_all(&parent)?;
            let path = parent.join(format!(
                "{}-{}",
                std::process::id(),
                NEXT_DIR.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path)?;
            Ok(Self(path))
        }
    }

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            // Only this test-created directory is removed.
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    // This synthetic bundle satisfies ModelBundle's lightweight artifact
    // contract, but is intentionally not a complete inference graph.
    fn synthetic_bundle(root: &Path) -> Result<Vec<u8>> {
        let limits = ArtifactLimits::default();
        let unet = HashMap::from([
            (
                "conv_in.weight".into(),
                Tensor::zeros((320, 9, 3, 3), DType::F32, &Device::Cpu)?,
            ),
            (
                "conv_out.weight".into(),
                Tensor::zeros((5, 320, 3, 3), DType::F32, &Device::Cpu)?,
            ),
            (
                "conv_out.bias".into(),
                Tensor::zeros((5,), DType::F32, &Device::Cpu)?,
            ),
        ]);
        let vae = HashMap::from([(
            "encoder.weight".into(),
            Tensor::zeros((1,), DType::F32, &Device::Cpu)?,
        )]);
        let prompt = HashMap::from([
            (
                "prompt_embeds".into(),
                Tensor::zeros((1, 77, 2048), DType::F32, &Device::Cpu)?,
            ),
            (
                "pooled_prompt_embeds".into(),
                Tensor::zeros((1, 1280), DType::F32, &Device::Cpu)?,
            ),
        ]);
        let digests = BundleDigests {
            unet: write_tensors(&root.join(UNET_FILE), &unet, &limits)?,
            vae: write_tensors(&root.join(VAE_FILE), &vae, &limits)?,
            prompt: write_tensors(&root.join(PROMPT_FILE), &prompt, &limits)?,
        };
        let source = SourceProvenance {
            repository: "https://example.invalid/synthetic".into(),
            revision: "a".repeat(40),
            license: "synthetic".into(),
        };
        let input_digest = "0".repeat(64);
        let manifest = Manifest {
            schema_version: 1,
            upstream_revision: UPSTREAM_REVISION.into(),
            pipeline: PIPELINE.into(),
            base_model: source.clone(),
            vae_model: source.clone(),
            checkpoint: source.clone(),
            prompt: source,
            config: None,
            input_sha256: InputDigests {
                base_unet: input_digest.clone(),
                base_vae: input_digest.clone(),
                checkpoint: input_digest.clone(),
                prompt: input_digest,
            },
            sha256: digests,
            scheduler: SchedulerConfig::default(),
            timestep: 400,
            vae_scaling_factor: 0.13025,
            adapter_rank: 256,
            adapter_alpha: 256.0,
        };
        let bytes = serde_json::to_vec(&manifest)?;
        fs::write(root.join(MANIFEST_FILE), &bytes)?;
        Ok(bytes)
    }

    fn synthetic_f16_bundle(root: &Path) -> Result<Vec<u8>> {
        use safetensors::tensor::{TensorView, serialize};

        let original = synthetic_bundle(root)?;
        let mut manifest: Manifest = serde_json::from_slice(&original)?;
        fn write_f16(path: &Path, shapes: &[(&str, Vec<usize>)]) -> Result<String> {
            let buffers: Vec<Vec<u8>> = shapes
                .iter()
                .map(|(_, shape)| vec![0; shape.iter().product::<usize>() * 2])
                .collect();
            let views = shapes
                .iter()
                .zip(&buffers)
                .map(|((name, shape), data)| {
                    TensorView::new(SafeDtype::F16, shape.clone(), data)
                        .map(|view| (*name, view))
                        .map_err(|e| artifact_error(format!("invalid test tensor: {e}")))
                })
                .collect::<Result<Vec<_>>>()?;
            let bytes = serialize(views, &None)
                .map_err(|e| artifact_error(format!("cannot serialize test tensors: {e}")))?;
            fs::write(path, &bytes)?;
            Ok(format!("{:x}", Sha256::digest(&bytes)))
        }
        manifest.sha256.unet = write_f16(
            &root.join(UNET_FILE),
            &[
                ("conv_in.weight", vec![320, 9, 3, 3]),
                ("conv_out.weight", vec![5, 320, 3, 3]),
                ("conv_out.bias", vec![5]),
            ],
        )?;
        manifest.sha256.vae = write_f16(&root.join(VAE_FILE), &[("encoder.weight", vec![1])])?;
        manifest.sha256.prompt = write_f16(
            &root.join(PROMPT_FILE),
            &[
                ("prompt_embeds", vec![1, 77, 2048]),
                ("pooled_prompt_embeds", vec![1, 1280]),
            ],
        )?;
        let bytes = serde_json::to_vec(&manifest)?;
        fs::write(root.join(MANIFEST_FILE), &bytes)?;
        Ok(bytes)
    }

    #[test]
    fn pinned_limits_count_decoded_bytes_across_bundle_files() -> Result<()> {
        let dir = ScratchDir::new()?;
        let bytes = synthetic_f16_bundle(&dir.0)?;
        let pin = format!("{:x}", Sha256::digest(&bytes));
        let unet_elements = 320 * 9 * 3 * 3 + 5 * 320 * 3 * 3 + 5;
        let total_elements = unet_elements + 1 + 77 * 2048 + 1280;
        let decoded_bytes = total_elements * 4;
        let mut limits = ArtifactLimits {
            // All F16 payloads fit, but their decoded F32 tensors do not.
            max_total_tensor_bytes: total_elements * 2,
            ..ArtifactLimits::default()
        };
        let load = |limits: &ArtifactLimits| {
            ModelBundle::load_with_limits_and_manifest_sha256(
                &dir.0,
                &Device::Cpu,
                DType::F32,
                limits,
                &pin,
            )
        };
        assert!(
            matches!(load(&limits), Err(Error::InvalidArtifact(message)) if message.contains("aggregate tensor budget exceeded"))
        );
        // Each file fits on its own, but their decoded bytes exceed the shared cap.
        let prompt_bytes = (77 * 2048 + 1280) * 4;
        limits.max_total_tensor_bytes = prompt_bytes + unet_elements * 2;
        assert!(limits.max_total_tensor_bytes > unet_elements * 4);
        assert!(limits.max_total_tensor_bytes > prompt_bytes);
        assert!(
            matches!(load(&limits), Err(Error::InvalidArtifact(message)) if message.contains("aggregate tensor budget exceeded"))
        );
        limits.max_total_tensor_bytes = decoded_bytes - 1;
        assert!(
            matches!(load(&limits), Err(Error::InvalidArtifact(message)) if message.contains("aggregate tensor budget exceeded"))
        );
        limits.max_total_tensor_bytes = decoded_bytes;
        assert_eq!(load(&limits)?.sequence.dims(), &[1, 77, 2048]);

        limits.max_manifest_bytes =
            u64::try_from(bytes.len()).map_err(|_| artifact_error("test manifest too large"))? - 1;
        assert!(load(&limits).is_err());
        limits.max_manifest_bytes = ArtifactLimits::default().max_manifest_bytes;
        fs::remove_file(dir.0.join(UNET_FILE))?;
        limits.max_total_tensor_bytes = 0;
        let wrong_pin = "0".repeat(64);
        assert!(
            matches!(ModelBundle::load_with_limits_and_manifest_sha256(&dir.0, &Device::Cpu, DType::F32, &limits, &wrong_pin), Err(Error::InvalidArtifact(message)) if message.contains("manifest SHA256 mismatch"))
        );
        Ok(())
    }

    #[test]
    fn pinned_runtime_limits_reject_before_graph_validation() -> Result<()> {
        let dir = ScratchDir::new()?;
        let bytes = synthetic_f16_bundle(&dir.0)?;
        let pin = format!("{:x}", Sha256::digest(&bytes));
        let limits = ArtifactLimits {
            max_total_tensor_bytes: 1,
            ..ArtifactLimits::default()
        };
        let result = crate::Osor::load_with_options_and_manifest_sha256_and_limits(
            &dir.0,
            crate::LoadOptions::default(),
            &pin,
            &limits,
        );
        assert!(
            matches!(result, Err(Error::InvalidArtifact(message)) if message.contains("aggregate tensor budget exceeded"))
        );
        let result = crate::Osor::load_with_options_and_manifest_sha256_and_limits(
            &dir.0,
            crate::LoadOptions::default(),
            &pin,
            &ArtifactLimits::default(),
        );
        assert!(
            matches!(result, Err(Error::InvalidArtifact(message)) if message.contains("native model tensor missing or incompatible"))
        );
        Ok(())
    }

    #[test]
    fn manifest_pin_loads_exact_synthetic_bytes_and_rejects_changed_bytes() -> Result<()> {
        let dir = ScratchDir::new()?;
        let bytes = synthetic_bundle(&dir.0)?;
        let pin = format!("{:x}", Sha256::digest(&bytes));
        let bundle = ModelBundle::load_with_manifest_sha256(
            &dir.0,
            &Device::Cpu,
            DType::F32,
            &pin.to_uppercase(),
        )?;
        assert_eq!(bundle.sequence.dims(), &[1, 77, 2048]);
        let mut changed = bytes;
        changed.push(b' ');
        fs::write(dir.0.join(MANIFEST_FILE), changed)?;
        // No tensor file exists now: the mismatch must win over file I/O.
        fs::remove_file(dir.0.join(UNET_FILE))?;
        let result = ModelBundle::load_with_manifest_sha256(&dir.0, &Device::Cpu, DType::F32, &pin);
        assert!(
            matches!(result, Err(Error::InvalidArtifact(message)) if message.contains("manifest SHA256 mismatch"))
        );
        Ok(())
    }

    #[test]
    fn manifest_pin_rejects_malformed_hash_before_filesystem_access() {
        for pin in [
            "bad",
            "z000000000000000000000000000000000000000000000000000000000000000",
        ] {
            let result = ModelBundle::load_with_manifest_sha256(
                "missing-model",
                &Device::Cpu,
                DType::F32,
                pin,
            );
            assert!(
                matches!(result, Err(Error::InvalidArtifact(message)) if message.contains("64 hexadecimal"))
            );
        }
    }

    #[test]
    fn manifest_pin_runtime_entrypoint_rejects_before_weight_io() -> Result<()> {
        let dir = ScratchDir::new()?;
        let bytes = synthetic_bundle(&dir.0)?;
        let pin = format!("{:x}", Sha256::digest(&bytes));
        let mut changed = bytes;
        changed.push(b'\n');
        fs::write(dir.0.join(MANIFEST_FILE), changed)?;
        fs::remove_file(dir.0.join(UNET_FILE))?;
        let result = crate::Osor::load_with_options_and_manifest_sha256(
            &dir.0,
            crate::LoadOptions::default(),
            &pin,
        );
        assert!(
            matches!(result, Err(Error::InvalidArtifact(message)) if message.contains("manifest SHA256 mismatch"))
        );
        Ok(())
    }

    #[test]
    fn mutated_safetensors_return_errors_without_panicking() -> Result<()> {
        let dir = ScratchDir::new()?;
        let path = dir.0.join("mutated.safetensors");
        let mut tensors = HashMap::new();
        tensors.insert(
            "block.weight".to_owned(),
            Tensor::from_slice(&[1_f32, -2., 3., 4., 5., 6.], (2, 3), &Device::Cpu)?,
        );
        tensors.insert(
            "block.bias".to_owned(),
            Tensor::from_slice(&[0.5_f32, -0.5], 2, &Device::Cpu)?,
        );
        let limits = ArtifactLimits::default();
        let digest = write_tensors(&path, &tensors, &limits)?;
        let seed = fs::read(&path)?;
        // Deterministic xorshift64* so failures reproduce.
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        };
        let index =
            |value: u64, len: usize| usize::try_from(value % len.max(1) as u64).unwrap_or(0);
        let mut accepted = 0;
        for _ in 0..2000 {
            let mut bytes = seed.clone();
            match next() % 5 {
                0 => {
                    for _ in 0..=(next() % 4) {
                        let at = index(next(), bytes.len());
                        if let Some(byte) = bytes.get_mut(at) {
                            *byte ^= 1 << (next() % 8);
                        }
                    }
                }
                1 => bytes.truncate(index(next(), bytes.len())),
                2 => {
                    let prefix = match next() % 3 {
                        0 => next(),
                        1 => next() % 4096,
                        _ => u64::MAX,
                    };
                    if let Some(head) = bytes.get_mut(..8) {
                        head.copy_from_slice(&prefix.to_le_bytes());
                    }
                }
                3 => {
                    let header_end = 8 + index(next(), 200);
                    let at = 8 + index(next(), header_end.saturating_sub(8));
                    let symbols = b"{}[]\":,0123456789-eE.ab_";
                    if let (Some(byte), Some(symbol)) =
                        (bytes.get_mut(at), symbols.get(index(next(), symbols.len())))
                    {
                        *byte = *symbol;
                    }
                }
                _ => bytes.extend((0..=(next() % 64)).map(|_| (next() & 0xff) as u8)),
            }
            fs::remove_file(&path)?;
            fs::write(&path, &bytes)?;
            // Any outcome but a panic is acceptable; a pinned digest must
            // only accept the unmodified bytes.
            if let Ok((_, actual)) = read_tensors(
                &path,
                &Device::Cpu,
                DType::F32,
                &limits,
                &mut Budget::default(),
                Some(&digest),
            ) {
                assert_eq!(actual, digest);
                assert_eq!(bytes, seed);
                accepted += 1;
            }
        }
        assert!(accepted < 2000);
        Ok(())
    }

    #[test]
    fn large_tensors_hash_in_order_with_concurrent_construction() -> Result<()> {
        let dir = ScratchDir::new()?;
        let path = dir.0.join("large.safetensors");
        // Two tensors above the concurrent-hash threshold around a small one,
        // so ordering across both hashing paths is covered.
        let elements = PARALLEL_HASH_BYTES / 4 + 3;
        let large = |phase: f32| -> Result<Tensor> {
            let values: Vec<f32> = (0..elements).map(|i| i as f32 * 0.5 + phase).collect();
            Ok(Tensor::from_vec(values, elements, &Device::Cpu)?)
        };
        let mut tensors = HashMap::new();
        tensors.insert("a.weight".to_owned(), large(1.0)?);
        tensors.insert(
            "b.bias".to_owned(),
            Tensor::new(&[7_f32, 8.], &Device::Cpu)?,
        );
        tensors.insert("c.weight".to_owned(), large(-3.0)?);
        let limits = ArtifactLimits::default();
        let digest = write_tensors(&path, &tensors, &limits)?;
        let mut file_hasher = Sha256::new();
        file_hasher.update(fs::read(&path)?);
        assert_eq!(digest, format!("{:x}", file_hasher.finalize()));
        let (loaded, actual) = read_tensors(
            &path,
            &Device::Cpu,
            DType::F32,
            &limits,
            &mut Budget::default(),
            Some(&digest),
        )?;
        assert_eq!(actual, digest);
        for (name, expected) in &tensors {
            let loaded = loaded
                .get(name)
                .ok_or_else(|| artifact_error("missing test tensor"))?;
            assert_eq!(loaded.to_vec1::<f32>()?, expected.to_vec1::<f32>()?);
        }
        // Change one byte inside the first large tensor's data.
        let mut bytes = fs::read(&path)?;
        let header = usize::try_from(u64::from_le_bytes(
            bytes
                .get(..8)
                .and_then(|prefix| prefix.try_into().ok())
                .ok_or_else(|| artifact_error("short test file"))?,
        ))
        .map_err(|_| artifact_error("test header too large"))?;
        let target = bytes
            .get_mut(8 + header + 100)
            .ok_or_else(|| artifact_error("short test file"))?;
        *target ^= 0x01;
        fs::remove_file(&path)?;
        fs::write(&path, &bytes)?;
        assert!(
            read_tensors(
                &path,
                &Device::Cpu,
                DType::F32,
                &limits,
                &mut Budget::default(),
                Some(&digest)
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn safetensors_round_trip_checks_hashes_and_never_overwrites() -> Result<()> {
        let dir = ScratchDir::new()?;
        let path = dir.0.join("weights.safetensors");
        let mut tensors = HashMap::new();
        tensors.insert(
            "block.weight".to_owned(),
            Tensor::from_slice(&[1_f32, -2., 3., 4.], (2, 2), &Device::Cpu)?,
        );
        let limits = ArtifactLimits::default();
        let digest = write_tensors(&path, &tensors, &limits)?;
        assert!(write_tensors(&path, &tensors, &limits).is_err());
        let (loaded, actual) = read_tensors(
            &path,
            &Device::Cpu,
            DType::F32,
            &limits,
            &mut Budget::default(),
            Some(&digest),
        )?;
        assert_eq!(actual, digest);
        assert_eq!(
            loaded
                .get("block.weight")
                .ok_or_else(|| artifact_error("missing test tensor"))?
                .to_vec2::<f32>()?,
            vec![vec![1., -2.], vec![3., 4.]]
        );
        let mut file = OpenOptions::new().write(true).open(&path)?;
        file.seek(SeekFrom::End(-1))?;
        file.write_all(&[0x7f])?;
        file.flush()?;
        assert!(
            read_tensors(
                &path,
                &Device::Cpu,
                DType::F32,
                &limits,
                &mut Budget::default(),
                Some(&digest)
            )
            .is_err()
        );
        let tight = ArtifactLimits {
            max_tensor_bytes: 4,
            ..limits
        };
        assert!(
            read_tensors(
                &path,
                &Device::Cpu,
                DType::F32,
                &tight,
                &mut Budget::default(),
                None
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_duplicate_names_and_bad_tensor_offsets() -> Result<()> {
        let dir = ScratchDir::new()?;
        let path = dir.0.join("bad.safetensors");
        let header = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(
            &u64::try_from(header.len())
                .map_err(|_| artifact_error("test header too large"))?
                .to_le_bytes(),
        )?;
        file.write_all(header)?;
        file.write_all(&[0_u8; 4])?;
        file.flush()?;
        assert!(
            read_tensors(
                &path,
                &Device::Cpu,
                DType::F32,
                &ArtifactLimits::default(),
                &mut Budget::default(),
                None
            )
            .is_err()
        );
        let invalid = TensorInfo {
            dtype: SafeDtype::F32,
            shape: vec![2, 2],
            data_offsets: (0, 4),
        };
        assert!(validate_info("x", &invalid, &ArtifactLimits::default(), DType::F32).is_err());
        let invalid = TensorInfo {
            dtype: SafeDtype::F32,
            shape: vec![usize::MAX, 2],
            data_offsets: (0, 4),
        };
        assert!(validate_info("x", &invalid, &ArtifactLimits::default(), DType::F32).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn rejects_bundle_symlink_escape_and_parent_traversal() -> Result<()> {
        let dir = ScratchDir::new()?;
        let root = dir.0.join("bundle");
        fs::create_dir(&root)?;
        let outside = dir.0.join("outside");
        fs::write(&outside, b"not a manifest")?;
        std::os::unix::fs::symlink(&outside, root.join(MANIFEST_FILE))?;
        assert!(bundle_file(&fs::canonicalize(&root)?, MANIFEST_FILE).is_err());
        assert!(bundle_root(&root.join("..")).is_err());
        assert!(checked_input_file(&root.join(MANIFEST_FILE)).is_err());
        Ok(())
    }

    #[test]
    fn provenance_and_nested_scheduler_fields_are_strict() -> Result<()> {
        let source = SourceProvenance {
            repository: "https://example.invalid/model".into(),
            revision: "a".repeat(40),
            license: "example-license".into(),
        };
        source.validate("test")?;
        let branch = SourceProvenance {
            revision: "main".into(),
            ..source
        };
        assert!(branch.validate("test").is_err());
        let input = serde_json::json!({"outer": {"known": 1, "unexpected": 2}});
        let known = serde_json::json!({"outer": {"known": 1}});
        assert!(reject_unknown_scheduler_fields(&input, &known, "scheduler").is_err());
        Ok(())
    }

    #[test]
    fn rejects_partial_models_before_bundle_publication() -> Result<()> {
        use crate::conversion::{ConversionRequest, convert};

        let dir = ScratchDir::new()?;
        let limits = ArtifactLimits::default();
        let mut base = HashMap::new();
        let mut adapter = HashMap::new();
        for target in [
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "conv",
            "conv1",
            "conv2",
            "conv_shortcut",
            "proj_in",
            "proj_out",
            "ff.net.2",
            "ff.net.0.proj",
        ] {
            let module = format!("block.{target}");
            base.insert(
                format!("{module}.weight"),
                Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?,
            );
            adapter.insert(
                format!("unet.{module}.lora_A.default.weight"),
                Tensor::ones((256, 2), DType::F32, &Device::Cpu)?,
            );
            adapter.insert(
                format!("unet.{module}.lora_B.default.weight"),
                Tensor::ones((2, 256), DType::F32, &Device::Cpu)?,
            );
        }
        base.insert(
            "conv_in.weight".into(),
            Tensor::zeros((320, 9, 3, 3), DType::F32, &Device::Cpu)?,
        );
        base.insert(
            "conv_out.weight".into(),
            Tensor::zeros((4, 320, 3, 3), DType::F32, &Device::Cpu)?,
        );
        base.insert(
            "conv_out.bias".into(),
            Tensor::zeros((4,), DType::F32, &Device::Cpu)?,
        );
        adapter.insert(
            "unet.conv_out.weight".into(),
            Tensor::ones((5, 320, 3, 3), DType::F32, &Device::Cpu)?,
        );
        adapter.insert(
            "unet.conv_out.bias".into(),
            Tensor::ones((5,), DType::F32, &Device::Cpu)?,
        );
        let vae = HashMap::from([(
            "encoder.weight".into(),
            Tensor::ones((2,), DType::F32, &Device::Cpu)?,
        )]);
        let prompt = HashMap::from([
            (
                "prompt_embeds".into(),
                Tensor::zeros((1, 77, 2048), DType::F32, &Device::Cpu)?,
            ),
            (
                "pooled_prompt_embeds".into(),
                Tensor::zeros((1, 1280), DType::F32, &Device::Cpu)?,
            ),
        ]);
        let base_path = dir.0.join("base.safetensors");
        let vae_path = dir.0.join("vae.safetensors");
        let adapter_path = dir.0.join("adapter.safetensors");
        let prompt_path = dir.0.join("prompt.safetensors");
        write_tensors(&base_path, &base, &limits)?;
        write_tensors(&vae_path, &vae, &limits)?;
        write_tensors(&adapter_path, &adapter, &limits)?;
        write_tensors(&prompt_path, &prompt, &limits)?;
        let source = SourceProvenance {
            repository: "https://example.invalid/fixture".into(),
            revision: "a".repeat(40),
            license: "fixture".into(),
        };
        let output = dir.0.join("bundle");
        let request = ConversionRequest {
            base_unet: base_path,
            base_vae: vae_path,
            checkpoint: adapter_path,
            prompt: prompt_path,
            output_dir: output.clone(),
            base_model_source: source.clone(),
            vae_source: source.clone(),
            checkpoint_source: source.clone(),
            prompt_source: source,
            config_source: None,
            scheduler: SchedulerConfig::default(),
            limits,
        };
        // Head shapes and a matching adapter inventory cannot establish a
        // complete model. Missing backbone/VAE tensors must prevent publication.
        assert!(matches!(convert(&request), Err(Error::InvalidArtifact(_))));
        assert!(!output.exists());

        let mut missing = adapter;
        missing.remove("unet.block.to_k.lora_A.default.weight");
        let missing_path = dir.0.join("missing.safetensors");
        write_tensors(&missing_path, &missing, &request.limits)?;
        let failed_output = dir.0.join("missing-bundle");
        let bad = ConversionRequest {
            checkpoint: missing_path,
            output_dir: failed_output.clone(),
            ..request.clone()
        };
        assert!(convert(&bad).is_err());
        assert!(!failed_output.exists());

        let partial_output = dir.0.join("partial-bundle");
        let bad_limits = ArtifactLimits {
            max_manifest_bytes: 1,
            ..request.limits.clone()
        };
        let bad = ConversionRequest {
            output_dir: partial_output.clone(),
            limits: bad_limits,
            ..request.clone()
        };
        assert!(convert(&bad).is_err());
        assert!(!partial_output.join(MANIFEST_FILE).exists());

        Ok(())
    }
}
