//! Same-input VAE diagnostics using explicitly supplied local F32 artifacts.

use std::{
    collections::{BTreeMap, HashMap},
    io::Read,
    path::{Path, PathBuf},
};

use candle_nn::Module;

use super::{Architecture, VaeDecoder, VaeEncoder, VarBuilder};
use crate::{
    DType, Device, Error, Result, Tensor,
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

// This follows the bounded JSON input pattern used by the local CLI and real-model test.
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
    dtype: &'static str,
    bundle: &'a Path,
    vae_sha256: &'a str,
    upstream_revision: &'a str,
    input: &'a Path,
    reference: &'a Path,
    tolerances: &'a Path,
    comparisons: &'a [StageComparison],
}

fn validate_input(
    tensor: &Tensor,
    name: &str,
    channels: usize,
    min: usize,
    max: usize,
    multiple: usize,
) -> Result<()> {
    let (batch, actual_channels, height, width) = tensor
        .dims4()
        .map_err(|_| Error::InvalidInput(format!("{name} must be rank four")))?;
    if tensor.dtype() != DType::F32
        || batch == 0
        || batch > 4
        || actual_channels != channels
        || height < min
        || width < min
        || height > max
        || width > max
        || !height.is_multiple_of(multiple)
        || !width.is_multiple_of(multiple)
    {
        return Err(Error::InvalidInput(format!(
            "{name} must be F32 NCHW, batch 1..=4, channels {channels}, dimensions {min}..={max} divisible by {multiple}"
        )));
    }
    Ok(())
}

fn trace_encoder(encoder: &VaeEncoder, image: &Tensor, output: &Path) -> Result<Tensor> {
    let mut trace = HashMap::new();
    let mut elements = 0_usize;
    let mut capture = |name: String, tensor: &Tensor| -> Result<()> {
        elements = elements
            .checked_add(tensor.elem_count())
            .ok_or_else(|| Error::InvalidInput("VAE trace size overflows".into()))?;
        if elements > 64 * 1024 * 1024 {
            return Err(Error::InvalidInput(
                "VAE trace exceeds capture element limit".into(),
            ));
        }
        trace.insert(name, tensor.to_device(&Device::Cpu)?.to_dtype(DType::F32)?);
        Ok(())
    };

    let mut xs = encoder.conv_in.forward(image)?;
    capture("encoder.conv_in".into(), &xs)?;
    for (index, block) in encoder.down.iter().enumerate() {
        xs = block.forward(&xs)?;
        capture(format!("encoder.down_blocks.{index}"), &xs)?;
    }
    xs = encoder.mid.forward(&xs, None)?;
    capture("encoder.mid_block".into(), &xs)?;
    xs = encoder.norm_out.forward(&xs)?;
    capture("encoder.conv_norm_out".into(), &xs)?;
    xs = candle_nn::ops::silu(&xs)?;
    xs = encoder.conv_out.forward(&xs)?;
    capture("encoder.conv_out".into(), &xs)?;
    xs = encoder.quant_conv.forward(&xs)?;
    capture("quant_conv".into(), &xs)?;
    tensor_io::save_new(&trace, output)?;
    Ok(xs)
}

#[test]
#[ignore = "requires a local bundle and same-input F32 VAE reference and measured tolerances"]
fn vae_cpu_same_input_reference() -> Result<()> {
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
    validate_input(image, "image", 3, 32, 1024, 8)?;
    validate_input(latent, "latent", 4, 1, 128, 1)?;
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
    let arch = Architecture::sdxl();
    let vs = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let encoder = VaeEncoder::new(vs.clone(), &arch)?;
    let encoder_only = std::env::var_os("OSOR_VAE_ENCODER_ONLY").is_some();
    let vae = if encoder_only {
        None
    } else {
        // Keep this construction identical to NativeModel::build.
        Some(VaeDecoder::new(vs, &arch)?)
    };
    // The image is already normalized to [-1, 1]; the latent is already unscaled.
    let moments = if let Some(output) = std::env::var_os("OSOR_VAE_TRACE_CAPTURE") {
        trace_encoder(&encoder, image, Path::new(&output))?
    } else {
        encoder.forward(image)?
    };
    let mut actual = HashMap::from([("posterior_moments".into(), moments)]);
    if let Some(vae) = vae {
        actual.insert("decoded_raw".into(), vae.forward(latent)?);
    }
    let reports = if encoder_only {
        let moments_reference = input(&expected, "posterior_moments")?.clone();
        let moments_tolerance = *tolerances
            .get("posterior_moments")
            .ok_or_else(|| Error::InvalidInput("missing tolerance for posterior_moments".into()))?;
        compare_stages(
            &HashMap::from([("posterior_moments".into(), moments_reference)]),
            &actual,
            &BTreeMap::from([("posterior_moments".into(), moments_tolerance)]),
        )?
    } else {
        compare_stages(&expected, &actual, &tolerances)?
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&Report {
            platform: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            backend: "CPU",
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
}
