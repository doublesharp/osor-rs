//! Explicitly opted-in acceptance tests; normal tests never load or download weights.

use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
    path::PathBuf,
};

use osor_rs::{
    Backend, DType, Device, Error, InferenceInput, LoadOptions, Osor, Result,
    artifact::{ArtifactLimits, ModelBundle},
    native::NativeModel,
    parity::{Tolerance, compare_reference_stages},
    tensor_io,
};

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
            actual.insert(stage.to_owned(), tensor.clone());
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
        NativeModel::new(missing_unet, bundle.vae.clone(), DType::F32, &Device::Cpu),
        Err(Error::InvalidArtifact(_))
    ));
    let mut missing_vae = bundle.vae;
    let removed = missing_vae.remove("post_quant_conv.weight");
    assert!(removed.is_some());
    assert!(matches!(
        NativeModel::new(bundle.unet, missing_vae, DType::F32, &Device::Cpu),
        Err(Error::InvalidArtifact(_))
    ));
    Ok(())
}
