//! Offline conversion of already-exported local Phase II safetensors.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, PathBuf};

use candle_core::{DType, Device, Tensor};

use crate::artifact::{
    ArtifactLimits, Budget, BundleDigests, InputDigests, MANIFEST_FILE, Manifest, PIPELINE,
    PROMPT_FILE, SourceProvenance, UNET_FILE, VAE_FILE, artifact_error, checked_input_file,
    read_tensors, require_shape, validate_plain_names, validate_prompt, validate_unet_contract,
    write_tensors,
};
use crate::scheduler::SchedulerConfig;
use crate::{Result, UPSTREAM_REVISION};

/// Every path is an existing local safetensors export except `output_dir`,
/// which must not exist. Nothing is downloaded or deserialized as pickle.
#[derive(Clone, Debug)]
pub struct ConversionRequest {
    pub base_unet: PathBuf,
    pub base_vae: PathBuf,
    pub checkpoint: PathBuf,
    pub prompt: PathBuf,
    pub output_dir: PathBuf,
    pub base_model_source: SourceProvenance,
    pub vae_source: SourceProvenance,
    pub checkpoint_source: SourceProvenance,
    pub prompt_source: SourceProvenance,
    pub config_source: Option<SourceProvenance>,
    pub scheduler: SchedulerConfig,
    pub limits: ArtifactLimits,
}

const TARGETS: [&str; 11] = [
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
];
const FF_PROJ_TARGET: &str = "ff.net.0.proj";

/// Merge rank-256 adapters and write a new, manifest-last bundle.
///
/// Failed writes can leave a directory without a manifest; it is not a valid
/// bundle and can be removed by the caller. Existing paths are never replaced.
pub fn convert(request: &ConversionRequest) -> Result<Manifest> {
    request.base_model_source.validate("base model")?;
    request.vae_source.validate("VAE model")?;
    request.checkpoint_source.validate("checkpoint")?;
    request.prompt_source.validate("prompt")?;
    if let Some(config) = &request.config_source {
        config.validate("config")?;
    }
    if request
        .output_dir
        .components()
        .any(|part| matches!(part, Component::ParentDir))
    {
        return Err(artifact_error(
            "output directory cannot contain parent-directory components",
        ));
    }

    let zero_digest = "0".repeat(64);
    let mut manifest = Manifest {
        schema_version: 1,
        upstream_revision: UPSTREAM_REVISION.to_owned(),
        pipeline: PIPELINE.to_owned(),
        base_model: request.base_model_source.clone(),
        vae_model: request.vae_source.clone(),
        checkpoint: request.checkpoint_source.clone(),
        prompt: request.prompt_source.clone(),
        config: request.config_source.clone(),
        input_sha256: InputDigests {
            base_unet: zero_digest.clone(),
            base_vae: zero_digest.clone(),
            checkpoint: zero_digest.clone(),
            prompt: zero_digest.clone(),
        },
        sha256: BundleDigests {
            unet: zero_digest.clone(),
            vae: zero_digest.clone(),
            prompt: zero_digest,
        },
        scheduler: request.scheduler.clone(),
        timestep: 400,
        vae_scaling_factor: 0.13025,
        adapter_rank: 256,
        adapter_alpha: 256.0,
    };
    manifest.validate()?;

    let mut budget = Budget::default();
    let base_unet = checked_input_file(&request.base_unet)?;
    let (base, unet_hash) = read_tensors(
        &base_unet,
        &Device::Cpu,
        DType::F32,
        &request.limits,
        &mut budget,
        None,
    )?;
    let base = canonicalize_weights(base, "base UNet")?;
    require_shape(&base, "conv_in.weight", &[320, 9, 3, 3])?;
    require_shape(&base, "conv_out.weight", &[4, 320, 3, 3])?;
    require_shape(&base, "conv_out.bias", &[4])?;

    let checkpoint_path = checked_input_file(&request.checkpoint)?;
    let (checkpoint, checkpoint_hash) = read_tensors(
        &checkpoint_path,
        &Device::Cpu,
        DType::F32,
        &request.limits,
        &mut budget,
        None,
    )?;
    let unet = merge_checkpoint(base, checkpoint, 256, 256.0)?;
    validate_unet_contract(&unet)?;

    let base_vae = checked_input_file(&request.base_vae)?;
    let (vae, vae_hash) = read_tensors(
        &base_vae,
        &Device::Cpu,
        DType::F32,
        &request.limits,
        &mut budget,
        None,
    )?;
    let vae = canonicalize_weights(vae, "base VAE")?;
    validate_plain_names(&vae, "VAE")?;
    if vae.is_empty() {
        return Err(artifact_error("VAE has no tensors"));
    }

    let prompt_path = checked_input_file(&request.prompt)?;
    let (prompt, prompt_hash) = read_tensors(
        &prompt_path,
        &Device::Cpu,
        DType::F32,
        &request.limits,
        &mut budget,
        None,
    )?;
    validate_prompt(&prompt)?;
    // Check the complete fixed graph before publishing, including non-adapted
    // tensors and the full VAE. Tensor clones share their model storage.
    let validated =
        crate::native::NativeModel::new(unet.clone(), vae.clone(), DType::F32, &Device::Cpu)?;
    drop(validated);
    manifest.input_sha256 = InputDigests {
        base_unet: unet_hash,
        base_vae: vae_hash,
        checkpoint: checkpoint_hash,
        prompt: prompt_hash,
    };

    // create_dir has exclusive semantics, including for a pre-existing symlink.
    fs::create_dir(&request.output_dir)?;
    manifest.sha256 = BundleDigests {
        unet: write_tensors(&request.output_dir.join(UNET_FILE), &unet, &request.limits)?,
        vae: write_tensors(&request.output_dir.join(VAE_FILE), &vae, &request.limits)?,
        prompt: write_tensors(
            &request.output_dir.join(PROMPT_FILE),
            &prompt,
            &request.limits,
        )?,
    };
    manifest.validate()?;
    let encoded = serde_json::to_vec_pretty(&manifest)?;
    if u64::try_from(encoded.len()).map_err(|_| artifact_error("manifest is too large"))?
        > request.limits.max_manifest_bytes
    {
        return Err(artifact_error(
            "output manifest exceeds the configured size limit",
        ));
    }
    let temporary_manifest = request.output_dir.join(".manifest.json.pending");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_manifest)?;
    file.write_all(&encoded)?;
    file.sync_all()?;
    drop(file);
    // A hard link publishes the complete, synced manifest atomically and
    // fails rather than replacing any existing destination.
    fs::hard_link(&temporary_manifest, request.output_dir.join(MANIFEST_FILE))?;
    let _ = fs::remove_file(temporary_manifest);
    Ok(manifest)
}

fn clean_name(mut name: &str) -> String {
    loop {
        if let Some(rest) = name.strip_prefix("module.") {
            name = rest;
        } else if let Some(rest) = name.strip_prefix("_orig_mod.") {
            name = rest;
        } else if let Some(rest) = name.strip_prefix("unet.") {
            name = rest;
        } else {
            break;
        }
    }
    name.replace(".base_layer.", ".")
}

fn canonicalize_weights(
    tensors: HashMap<String, Tensor>,
    label: &str,
) -> Result<HashMap<String, Tensor>> {
    let mut result = HashMap::new();
    result
        .try_reserve(tensors.len())
        .map_err(|e| artifact_error(format!("cannot allocate {label} names: {e}")))?;
    for (name, tensor) in tensors {
        let canonical = clean_name(&name);
        if result.insert(canonical.clone(), tensor).is_some() {
            return Err(artifact_error(format!(
                "{label} name collision at {canonical}"
            )));
        }
    }
    validate_plain_names(&result, label)?;
    Ok(result)
}

fn target_kind(module: &str) -> Option<&'static str> {
    TARGETS
        .into_iter()
        .chain(std::iter::once(FF_PROJ_TARGET))
        .find(|target| {
            module == *target
                || module
                    .strip_suffix(*target)
                    .is_some_and(|prefix| prefix.ends_with('.'))
        })
}

#[derive(Default)]
struct Pair {
    a: Option<Tensor>,
    b: Option<Tensor>,
}

fn adapter_part(name: &str) -> Option<(&str, bool)> {
    for (suffix, is_a) in [
        (".lora_A.default.weight", true),
        (".lora_B.default.weight", false),
        (".lora_A.weight", true),
        (".lora_B.weight", false),
    ] {
        if let Some(module) = name.strip_suffix(suffix) {
            return Some((module, is_a));
        }
    }
    None
}

fn merge_checkpoint(
    mut base: HashMap<String, Tensor>,
    checkpoint: HashMap<String, Tensor>,
    rank: usize,
    alpha: f64,
) -> Result<HashMap<String, Tensor>> {
    let mut expected = BTreeSet::new();
    let mut kinds = BTreeSet::new();
    for name in base.keys() {
        if let Some(module) = name.strip_suffix(".weight")
            && let Some(kind) = target_kind(module)
        {
            expected.insert(module.to_owned());
            kinds.insert(kind);
        }
    }
    for required in TARGETS.into_iter().chain(std::iter::once(FF_PROJ_TARGET)) {
        if !kinds.contains(required) {
            return Err(artifact_error(format!(
                "base UNet is missing Phase II target class {required}"
            )));
        }
    }

    let mut pairs: BTreeMap<String, Pair> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut head_weight = None;
    let mut head_bias = None;
    for (name, tensor) in checkpoint {
        let canonical = clean_name(&name);
        if !seen.insert(canonical.clone()) {
            return Err(artifact_error(format!(
                "checkpoint name collision at {canonical}"
            )));
        }
        if let Some((module, is_a)) = adapter_part(&canonical) {
            if !expected.contains(module) {
                return Err(artifact_error(format!("unexpected adapter for {module}")));
            }
            let pair = pairs.entry(module.to_owned()).or_default();
            let slot = if is_a { &mut pair.a } else { &mut pair.b };
            if slot.replace(tensor).is_some() {
                return Err(artifact_error(format!("duplicate adapter for {module}")));
            }
        } else if canonical.contains(".lora_") {
            return Err(artifact_error(format!(
                "unsupported adapter tensor {canonical}"
            )));
        } else if canonical == "conv_out.weight" {
            head_weight = Some(tensor);
        } else if canonical == "conv_out.bias" {
            head_bias = Some(tensor);
        } else {
            // Full state-dict exports may include frozen copies. They must
            // correspond exactly to the explicitly supplied base model.
            let original = base.get(&canonical).ok_or_else(|| {
                artifact_error(format!("unexpected checkpoint tensor {canonical}"))
            })?;
            if !same_values(original, &tensor)? {
                return Err(artifact_error(format!(
                    "checkpoint base tensor differs from supplied base: {canonical}"
                )));
            }
        }
    }
    let weight = head_weight.ok_or_else(|| artifact_error("missing trained conv_out.weight"))?;
    let bias = head_bias.ok_or_else(|| artifact_error("missing trained conv_out.bias"))?;
    if weight.dims() != [5, 320, 3, 3] || bias.dims() != [5] {
        return Err(artifact_error(
            "trained conv_out must have weight [5, 320, 3, 3] and bias [5]",
        ));
    }
    for module in expected {
        let pair = pairs
            .remove(&module)
            .ok_or_else(|| artifact_error(format!("missing adapter A and B for {module}")))?;
        let a = pair
            .a
            .ok_or_else(|| artifact_error(format!("missing adapter A for {module}")))?;
        let b = pair
            .b
            .ok_or_else(|| artifact_error(format!("missing adapter B for {module}")))?;
        let key = format!("{module}.weight");
        let original = base
            .get(&key)
            .ok_or_else(|| artifact_error(format!("missing base tensor {key}")))?;
        let merged = merge_low_rank(original, &a, &b, rank, alpha)?;
        base.insert(key, merged);
    }
    if !pairs.is_empty() {
        return Err(artifact_error("unexpected unmerged adapters"));
    }
    base.insert("conv_out.weight".to_owned(), weight);
    base.insert("conv_out.bias".to_owned(), bias);
    Ok(base)
}

fn same_values(a: &Tensor, b: &Tensor) -> Result<bool> {
    if a.dims() != b.dims() {
        return Ok(false);
    }
    let a = a
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let b = b
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(a == b)
}

fn merge_low_rank(
    base: &Tensor,
    a: &Tensor,
    b: &Tensor,
    rank: usize,
    alpha: f64,
) -> Result<Tensor> {
    if rank == 0 || !alpha.is_finite() {
        return Err(artifact_error("invalid LoRA rank or alpha"));
    }
    let (output, inner) = match (base.dims(), a.dims(), b.dims()) {
        ([out, input], [ar, ai], [bo, br])
            if *ar == rank && *ai == *input && *bo == *out && *br == rank =>
        {
            (*out, *input)
        }
        ([out, input, kh, kw], [ar, ai, ah, aw], [bo, br, 1, 1])
            if *ar == rank
                && *ai == *input
                && *ah == *kh
                && *aw == *kw
                && *bo == *out
                && *br == rank =>
        {
            let inner = input
                .checked_mul(*kh)
                .and_then(|n| n.checked_mul(*kw))
                .ok_or_else(|| artifact_error("convolution adapter dimensions overflow"))?;
            (*out, inner)
        }
        _ => {
            return Err(artifact_error(format!(
                "LoRA shapes do not match base {:?}, A {:?}, B {:?}, rank {rank}",
                base.dims(),
                a.dims(),
                b.dims()
            )));
        }
    };
    let a = a.to_dtype(DType::F32)?.reshape((rank, inner))?;
    let b = b.to_dtype(DType::F32)?.reshape((output, rank))?;
    let delta = b
        .matmul(&a)?
        .affine(alpha / rank as f64, 0.0)?
        .reshape(base.dims())?;
    Ok(base.to_dtype(DType::F32)?.add(&delta)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_linear_and_spatial_convolution_updates() -> Result<()> {
        let device = &Device::Cpu;
        let base = Tensor::from_slice(&[1_f32, 2., 3., 4.], (2, 2), device)?;
        let a = Tensor::from_slice(&[1_f32, 2.], (1, 2), device)?;
        let b = Tensor::from_slice(&[2_f32, 3.], (2, 1), device)?;
        let merged = merge_low_rank(&base, &a, &b, 1, 1.0)?;
        assert_eq!(merged.to_vec2::<f32>()?, vec![vec![3., 6.], vec![6., 10.]]);

        let base = Tensor::from_slice(&[1_f32, 2., 3., 4.], (1, 1, 2, 2), device)?;
        let a = Tensor::from_slice(&[1_f32, 2., 3., 4.], (1, 1, 2, 2), device)?;
        let b = Tensor::from_slice(&[2_f32], (1, 1, 1, 1), device)?;
        let merged = merge_low_rank(&base, &a, &b, 1, 0.5)?;
        assert_eq!(
            merged.flatten_all()?.to_vec1::<f32>()?,
            vec![2., 4., 6., 8.]
        );
        let wrong_b = Tensor::zeros((1, 1, 2, 2), DType::F32, device)?;
        assert!(merge_low_rank(&base, &a, &wrong_b, 1, 1.0).is_err());
        assert!(merge_low_rank(&base, &a, &b, 2, 2.0).is_err());
        Ok(())
    }

    fn fixture() -> Result<(HashMap<String, Tensor>, HashMap<String, Tensor>)> {
        let mut base = HashMap::new();
        let mut checkpoint = HashMap::new();
        for target in TARGETS.into_iter().chain(std::iter::once(FF_PROJ_TARGET)) {
            let module = format!("block.{target}");
            base.insert(
                format!("{module}.weight"),
                Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?,
            );
            checkpoint.insert(
                format!("module._orig_mod.unet.{module}.lora_A.default.weight"),
                Tensor::ones((1, 2), DType::F32, &Device::Cpu)?,
            );
            checkpoint.insert(
                format!("unet.{module}.lora_B.default.weight"),
                Tensor::ones((2, 1), DType::F32, &Device::Cpu)?,
            );
        }
        base.insert(
            "block.to_q.bias".into(),
            Tensor::from_slice(&[7_f32, 8.], (2,), &Device::Cpu)?,
        );
        checkpoint.insert(
            "unet.block.to_q.base_layer.weight".into(),
            Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?,
        );
        checkpoint.insert(
            "module.unet.conv_out.weight".into(),
            Tensor::zeros((5, 320, 3, 3), DType::F32, &Device::Cpu)?,
        );
        checkpoint.insert(
            "unet.conv_out.bias".into(),
            Tensor::zeros((5,), DType::F32, &Device::Cpu)?,
        );
        Ok((base, checkpoint))
    }

    #[test]
    fn every_base_target_requires_both_adapters_and_trained_head() -> Result<()> {
        let (base, checkpoint) = fixture()?;
        let merged = merge_checkpoint(base, checkpoint, 1, 1.0)?;
        assert_eq!(
            merged
                .get("block.to_q.weight")
                .ok_or_else(|| artifact_error("missing test weight"))?
                .to_vec2::<f32>()?,
            vec![vec![1., 1.], vec![1., 1.]]
        );
        assert_eq!(
            merged
                .get("block.to_q.bias")
                .ok_or_else(|| artifact_error("missing test bias"))?
                .to_vec1::<f32>()?,
            vec![7., 8.]
        );
        assert_eq!(
            merged
                .get("conv_out.weight")
                .ok_or_else(|| artifact_error("missing test head"))?
                .dims(),
            &[5, 320, 3, 3]
        );

        let (base, mut checkpoint) = fixture()?;
        checkpoint.remove("unet.block.to_k.lora_B.default.weight");
        assert!(merge_checkpoint(base, checkpoint, 1, 1.0).is_err());
        let (base, mut checkpoint) = fixture()?;
        checkpoint.remove("unet.conv_out.bias");
        assert!(merge_checkpoint(base, checkpoint, 1, 1.0).is_err());
        let (base, mut checkpoint) = fixture()?;
        checkpoint.insert(
            "unet.block.extra.lora_A.default.weight".into(),
            Tensor::ones((1, 2), DType::F32, &Device::Cpu)?,
        );
        assert!(merge_checkpoint(base, checkpoint, 1, 1.0).is_err());
        Ok(())
    }

    #[test]
    fn rejects_collisions_and_checkpoint_base_mismatch() -> Result<()> {
        let (base, mut checkpoint) = fixture()?;
        checkpoint.insert(
            "block.to_q.base_layer.weight".into(),
            Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?,
        );
        assert!(merge_checkpoint(base, checkpoint, 1, 1.0).is_err());
        let (base, mut checkpoint) = fixture()?;
        checkpoint.insert(
            "unet.block.to_q.base_layer.weight".into(),
            Tensor::ones((2, 2), DType::F32, &Device::Cpu)?,
        );
        assert!(merge_checkpoint(base, checkpoint, 1, 1.0).is_err());
        let (base, mut checkpoint) = fixture()?;
        checkpoint.insert(
            "unet.conv_out.weight".into(),
            Tensor::zeros((4, 320, 3, 3), DType::F32, &Device::Cpu)?,
        );
        // This also collides after prefix cleaning, so it cannot hide the trained head.
        assert!(merge_checkpoint(base, checkpoint, 1, 1.0).is_err());
        Ok(())
    }
}
