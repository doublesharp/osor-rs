//! Explicitly opted-in acceptance tests; normal tests never load or download weights.

use std::{
    collections::{BTreeMap, HashMap},
    io::{Read, Write},
    path::PathBuf,
};

use osor_rs::{
    Backend, DType, Device, Error, InferenceInput, LoadOptions, Osor, Result,
    artifact::{ArtifactLimits, Manifest, ModelBundle},
    conversion::{
        ConversionOptions, ConversionOutput, ConversionReaders, ConversionSink, convert_with_io,
    },
    native::NativeModel,
    parity::{Tolerance, compare_reference_stages},
    tensor_io,
};
use sha2::{Digest, Sha256};

fn path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name).map(PathBuf::from).ok_or_else(|| {
        Error::InvalidInput(format!(
            "set {name} to an explicitly installed local artifact"
        ))
    })
}

fn reference(backend: Backend) -> Result<()> {
    let bundle_path = path("OSOR_TEST_BUNDLE")?;
    let input_path = path("OSOR_TEST_INPUT")?;
    let expected_path = path("OSOR_TEST_REFERENCE")?;
    let tolerance_path = path("OSOR_TEST_TOLERANCES")?;
    let mut tolerance_bytes = Vec::new();
    std::fs::File::open(&tolerance_path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut tolerance_bytes)?;
    if tolerance_bytes.len() > 1024 * 1024 {
        return Err(Error::InvalidInput("tolerances exceed 1 MiB".into()));
    }
    let tolerances: BTreeMap<String, Tolerance> = serde_json::from_slice(&tolerance_bytes)?;
    let expected_manifest_sha256 = std::env::var_os("OSOR_TEST_MANIFEST_SHA256")
        .map(|value| {
            value.into_string().map_err(|_| {
                Error::InvalidInput("OSOR_TEST_MANIFEST_SHA256 must be UTF-8 hexadecimal".into())
            })
        })
        .transpose()?;
    let options = LoadOptions {
        backend,
        ..LoadOptions::default()
    };
    let max_decoded_bytes = std::env::var_os("OSOR_TEST_MAX_DECODED_BYTES")
        .map(|value| {
            value
                .to_str()
                .and_then(|value| value.parse::<u64>().ok())
                .ok_or_else(|| {
                    Error::InvalidInput("OSOR_TEST_MAX_DECODED_BYTES must be a u64".into())
                })
        })
        .transpose()?;
    let model = match (expected_manifest_sha256.as_deref(), max_decoded_bytes) {
        (Some(hash), Some(max_total_tensor_bytes)) => {
            Osor::load_with_options_and_manifest_sha256_and_limits(
                &bundle_path,
                options,
                hash,
                &ArtifactLimits {
                    max_total_tensor_bytes,
                    ..ArtifactLimits::default()
                },
            )?
        }
        (Some(hash), None) => {
            Osor::load_with_options_and_manifest_sha256(&bundle_path, options, hash)?
        }
        (None, Some(_)) => {
            return Err(Error::InvalidInput(
                "set OSOR_TEST_MANIFEST_SHA256 when testing a decoded-byte cap".into(),
            ));
        }
        (None, None) => Osor::load_with_options(&bundle_path, options)?,
    };
    let inputs = tensor_io::load(&input_path, model.device())?;
    let get = |name: &str| {
        inputs
            .get(name)
            .ok_or_else(|| Error::InvalidInput(format!("missing input tensor {name}")))
    };
    let mut actual = HashMap::new();
    let result = model.infer_with_observer(
        InferenceInput {
            image: get("image")?,
            mask: get("mask")?,
            posterior_noise: get("posterior_noise")?,
            diffusion_noise: get("diffusion_noise")?,
        },
        &mut |stage, tensor| {
            actual.insert(stage.as_str().to_owned(), tensor.clone());
            Ok(())
        },
    )?;
    model.device().synchronize()?;
    assert_eq!(result.image.dtype(), DType::F32);
    assert_eq!(result.alpha.dtype(), DType::F32);
    let expected = tensor_io::load(&expected_path, &Device::Cpu)?;
    let reports = compare_reference_stages(&expected, &actual, &tolerances)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "platform":std::env::consts::OS, "arch":std::env::consts::ARCH,
            "backend":format!("{backend:?}"), "dtype":"F32",
        "compute_device":model.compute_device_description(),
            "bundle":bundle_path.to_string_lossy(), "input":input_path.to_string_lossy(), "reference":expected_path.to_string_lossy(),
            "tolerances":tolerance_path.to_string_lossy(), "model_sha256":model.manifest().sha256,
            "expected_manifest_sha256":expected_manifest_sha256,
            "max_decoded_bytes":max_decoded_bytes,
            "upstream_revision":osor_rs::UPSTREAM_REVISION, "comparisons":reports,
        }))?
    );
    if let Some(output) = std::env::var_os("OSOR_TEST_ACTUAL_CAPTURE") {
        tensor_io::save_new(&actual, output)?;
    }
    assert!(
        reports.iter().all(|report| report.passed),
        "reference stage comparison failed"
    );
    Ok(())
}

#[test]
#[ignore = "requires an installed bundle, captured reference/noise, and measured per-stage tolerances"]
fn real_model_cpu_reference_stages() -> Result<()> {
    reference(Backend::Cpu)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal device and explicit reference artifacts"]
fn real_model_metal_reference_stages() -> Result<()> {
    reference(Backend::Metal { ordinal: 0 })
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device and explicit reference artifacts"]
fn real_model_cuda_reference_stages() -> Result<()> {
    reference(Backend::Cuda { ordinal: 0 })
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "requires a real WGPU device and explicit reference artifacts"]
fn real_model_wgpu_reference_stages() -> Result<()> {
    reference(Backend::Wgpu { ordinal: 0 })
}

#[test]
#[ignore = "requires an installed full bundle; run acceptance tests with --test-threads=1"]
fn real_model_missing_critical_weights_fail() -> Result<()> {
    let bundle = ModelBundle::load(path("OSOR_TEST_BUNDLE")?, &Device::Cpu, DType::F32)?;
    let mut missing_unet = bundle.unet.clone();
    let removed = missing_unet.remove("time_embedding.linear_1.weight");
    assert!(removed.is_some());
    assert!(matches!(
        NativeModel::new(
            missing_unet,
            bundle.vae.clone(),
            &bundle.sequence,
            &bundle.pooled,
            DType::F32,
            &Device::Cpu
        ),
        Err(Error::InvalidArtifact(_))
    ));
    let mut missing_vae = bundle.vae;
    let removed = missing_vae.remove("post_quant_conv.weight");
    assert!(removed.is_some());
    assert!(matches!(
        NativeModel::new(
            bundle.unet,
            missing_vae,
            &bundle.sequence,
            &bundle.pooled,
            DType::F32,
            &Device::Cpu
        ),
        Err(Error::InvalidArtifact(_))
    ));
    Ok(())
}

#[derive(Default)]
struct DigestWriter {
    digest: Sha256,
    bytes: u64,
}

impl Write for DigestWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.digest.update(bytes);
        self.bytes += bytes.len() as u64;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[derive(Default)]
struct DigestSink {
    finished: Vec<(ConversionOutput, u64, String)>,
    manifest: Vec<u8>,
}

impl ConversionSink for DigestSink {
    type Writer = DigestWriter;
    fn create_new(&mut self, output: ConversionOutput) -> Result<Self::Writer> {
        let order = [
            ConversionOutput::Unet,
            ConversionOutput::Vae,
            ConversionOutput::Prompt,
        ];
        if order.get(self.finished.len()) != Some(&output) {
            return Err(Error::InvalidInput("unexpected output order".into()));
        }
        Ok(DigestWriter::default())
    }
    fn finish(&mut self, output: ConversionOutput, writer: Self::Writer) -> Result<()> {
        self.finished.push((
            output,
            writer.bytes,
            format!("{:x}", writer.digest.finalize()),
        ));
        Ok(())
    }
    fn publish_manifest(&mut self, bytes: &[u8]) -> Result<()> {
        if self.finished.len() != 3 || !self.manifest.is_empty() {
            return Err(Error::InvalidInput(
                "unexpected manifest publication".into(),
            ));
        }
        self.manifest.extend_from_slice(bytes);
        Ok(())
    }
}

#[test]
#[ignore = "requires four installed conversion inputs and a pinned qualified bundle; hashes outputs without saving weights"]
fn real_model_conversion_io_matches_qualified_bundle() -> Result<()> {
    let mut bytes = Vec::new();
    std::fs::File::open(path("OSOR_TEST_BUNDLE")?.join("manifest.json"))?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return Err(Error::InvalidInput("manifest exceeds 1 MiB".into()));
    }
    let expected_pin = std::env::var("OSOR_TEST_MANIFEST_SHA256")
        .map_err(|_| Error::InvalidInput("set OSOR_TEST_MANIFEST_SHA256".into()))?;
    let actual_pin = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(actual_pin, expected_pin.to_ascii_lowercase());
    let expected: Manifest = serde_json::from_slice(&bytes)?;
    expected.validate()?;
    let options = ConversionOptions {
        base_model_source: expected.base_model.clone(),
        vae_source: expected.vae_model.clone(),
        checkpoint_source: expected.checkpoint.clone(),
        prompt_source: expected.prompt.clone(),
        config_source: expected.config.clone(),
        expected_input_sha256: expected.input_sha256.clone(),
        scheduler: expected.scheduler.clone(),
        limits: ArtifactLimits {
            max_total_tensor_bytes: 14 * 1024 * 1024 * 1024,
            ..ArtifactLimits::default()
        },
    };
    let mut base_unet = std::fs::File::open(path("OSOR_TEST_BASE_UNET")?)?;
    let mut base_vae = std::fs::File::open(path("OSOR_TEST_BASE_VAE")?)?;
    let mut checkpoint = std::fs::File::open(path("OSOR_TEST_CHECKPOINT")?)?;
    let mut prompt = std::fs::File::open(path("OSOR_TEST_PROMPT")?)?;
    let mut sink = DigestSink::default();
    let start = std::time::Instant::now();
    let converted = convert_with_io(
        &options,
        ConversionReaders {
            base_unet: &mut base_unet,
            base_vae: &mut base_vae,
            checkpoint: &mut checkpoint,
            prompt: &mut prompt,
        },
        &mut sink,
        &mut |_| Ok(()),
    )?;
    assert_eq!(sink.finished.len(), 3);
    for (output, _, digest) in &sink.finished {
        let expected_digest = match output {
            ConversionOutput::Unet => &expected.sha256.unet,
            ConversionOutput::Vae => &expected.sha256.vae,
            ConversionOutput::Prompt => &expected.sha256.prompt,
        };
        assert_eq!(digest, expected_digest);
    }
    assert_eq!(
        serde_json::to_vec(&converted)?,
        serde_json::to_vec(&expected)?
    );
    assert_eq!(sink.manifest, serde_json::to_vec_pretty(&expected)?);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "test": "conversion_io_matches_qualified_bundle",
            "platform": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "trusted_manifest_sha256": actual_pin,
            "elapsed_seconds": start.elapsed().as_secs_f64(),
            "max_decoded_input_bytes": options.limits.max_total_tensor_bytes,
            "finished_outputs": sink.finished.iter().map(|(output, bytes, sha256)| {
                serde_json::json!({"output": output.file_name(), "bytes": bytes, "sha256": sha256})
            }).collect::<Vec<_>>(),
            "output_weight_files_written": false,
        }))?
    );
    Ok(())
}
