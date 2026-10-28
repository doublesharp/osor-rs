//! Same-input VAE diagnostics using explicitly supplied local F32 artifacts.

use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device, Tensor};

use super::{
    super::{Weights, acquire_device, guarded, image_shape, readback, upload},
    vae::Vae,
};
use crate::{
    Error, Result,
    artifact::{
        ArtifactLimits, Budget, MANIFEST_FILE, Manifest, VAE_FILE, checked_input_file,
        read_tensors, validate_plain_names,
    },
    parity::{StageComparison, Tolerance, compare_stages},
    tensor_io,
};

fn path(name: &str) -> Result<PathBuf> {
    std::env::var_os(name).map(PathBuf::from).ok_or_else(|| {
        Error::InvalidInput(format!(
            "set {name} to an explicitly installed local artifact"
        ))
    })
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path, limit: u64) -> Result<T> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(limit.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len()).map_err(|_| Error::InvalidInput("JSON is too large".into()))?
        > limit
    {
        return Err(Error::InvalidInput(format!(
            "JSON input {} exceeds {limit} bytes",
            path.display()
        )));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn input<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| Error::InvalidInput(format!("missing VAE input tensor {name}")))
}

#[derive(serde::Serialize)]
struct Report<'a> {
    platform: &'static str,
    arch: &'static str,
    backend: &'static str,
    adapter: &'a str,
    dtype: &'static str,
    bundle: &'a Path,
    vae_sha256: &'a str,
    upstream_revision: &'a str,
    input: &'a Path,
    reference: &'a Path,
    tolerances: &'a Path,
    comparisons: &'a [StageComparison],
}

#[test]
#[ignore = "requires a real WGPU adapter, local bundle and same-input F32 VAE reference and measured tolerances"]
fn vae_wgpu_same_input_reference() -> Result<()> {
    let bundle = path("OSOR_TEST_BUNDLE")?;
    let input_path = path("OSOR_VAE_TEST_INPUT")?;
    let reference_path = path("OSOR_VAE_TEST_REFERENCE")?;
    let tolerance_path = path("OSOR_VAE_TEST_TOLERANCES")?;
    let limits = ArtifactLimits::default();
    let manifest_path = checked_input_file(&bundle.join(MANIFEST_FILE))?;
    let manifest: Manifest = read_json(&manifest_path, limits.max_manifest_bytes)?;
    manifest.validate()?;
    let tolerances: BTreeMap<String, Tolerance> = read_json(&tolerance_path, 1024 * 1024)?;
    let inputs = tensor_io::load(&input_path, &Device::Cpu)?;
    if inputs.len() != 2 || !inputs.contains_key("image") || !inputs.contains_key("latent") {
        return Err(Error::InvalidInput(
            "VAE input must contain exactly image and latent".into(),
        ));
    }
    let image = input(&inputs, "image")?;
    let latent = input(&inputs, "latent")?;
    let [b, _, h, w] = image_shape(image, 3, false, "image")?;
    let [lb, _, lh, lw] = image_shape(latent, 4, true, "latent")?;
    let image_values = super::super::cpu_values(image, "image", &[b, 3, h, w], false)?;
    if image_values.iter().any(|&v| !(-1.0..=1.0).contains(&v)) {
        return Err(Error::InvalidInput(
            "normalized image values must be in [-1, 1]".into(),
        ));
    }
    let expected = tensor_io::load(&reference_path, &Device::Cpu)?;
    if expected.values().any(|tensor| tensor.dtype() != DType::F32) {
        return Err(Error::InvalidInput(
            "VAE reference tensors must be F32".into(),
        ));
    }

    let vae_path = checked_input_file(&bundle.join(VAE_FILE))?;
    let (weights, digest) = read_tensors(
        &vae_path,
        &Device::Cpu,
        DType::F32,
        &limits,
        &mut Budget::default(),
        Some(&manifest.sha256.vae),
    )?;
    validate_plain_names(&weights, "VAE")?;

    guarded("VAE same-input reference test", || {
        let (device, adapter) = acquire_device(0)?;
        let mut weights = Weights::new(weights, &device);
        let vae = Vae::new(&mut weights)?;
        // Image is normalized to [-1, 1]; latent is unscaled. They are independent inputs.
        let image = upload(image, "image", [b, 3, h, w], &device)?;
        let latent = upload(latent, "latent", [lb, 4, lh, lw], &device)?;
        let encoded = if let Some(output) = std::env::var_os("OSOR_VAE_TRACE_CAPTURE") {
            let mut trace = HashMap::new();
            let mut elements = 0_usize;
            let encoded = vae.encode_with_trace(image, |name, tensor| {
                let shape = tensor.dims();
                let count = shape.iter().try_fold(1_usize, |count, &dimension| {
                    count
                        .checked_mul(dimension)
                        .ok_or_else(|| Error::InvalidInput("VAE trace dimensions overflow".into()))
                })?;
                elements = elements
                    .checked_add(count)
                    .ok_or_else(|| Error::InvalidInput("VAE trace size overflows".into()))?;
                if elements > 64 * 1024 * 1024 {
                    return Err(Error::InvalidInput(
                        "VAE trace exceeds capture element limit".into(),
                    ));
                }
                trace.insert(name.to_owned(), readback(tensor, shape, name)?);
                Ok(())
            })?;
            tensor_io::save_new(&trace, output)?;
            encoded
        } else {
            vae.encode(image)
        };
        let moments = readback(encoded, [b, 8, h / 8, w / 8], "VAE moments")?;
        let decoded = readback(
            vae.decode(latent),
            [lb, 3, lh * 8, lw * 8],
            "VAE decoded image",
        )?;
        let actual = HashMap::from([
            ("posterior_moments".into(), moments),
            ("decoded_raw".into(), decoded),
        ]);
        let reports = compare_stages(&expected, &actual, &tolerances)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&Report {
                platform: std::env::consts::OS,
                arch: std::env::consts::ARCH,
                backend: "WGPU",
                adapter: &adapter,
                dtype: "F32",
                bundle: &bundle,
                vae_sha256: &digest,
                upstream_revision: &manifest.upstream_revision,
                input: &input_path,
                reference: &reference_path,
                tolerances: &tolerance_path,
                comparisons: &reports,
            })?
        );
        if let Some(output) = std::env::var_os("OSOR_VAE_TEST_CAPTURE") {
            tensor_io::save_new(&actual, output)?;
        }
        assert!(
            reports.iter().all(|report| report.passed),
            "VAE comparison failed"
        );
        Ok(())
    })
}
