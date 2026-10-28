//! Local-only tensor inference and qualification utility.

use std::{
    collections::HashMap,
    ffi::OsString,
    io::Read,
    path::{Path, PathBuf},
    time::Instant,
};

use osor_rs::{
    Backend, DType, Error, InferenceInput, LoadOptions, Osor, Result, Tensor,
    artifact::{ArtifactLimits, SourceProvenance},
    conversion::{ConversionRequest, convert},
    parity::{Tolerance, compare_reference_stages},
    scheduler::SchedulerConfig,
    tensor_io,
};
use serde::Deserialize;

const USAGE: &str = "osor convert CONVERSION.json\nosor infer BUNDLE INPUT.safetensors OUTPUT.safetensors [cpu|metal|cuda|wgpu] [STAGES.safetensors]\nosor compare REFERENCE.safetensors ACTUAL.safetensors TOLERANCES.json\n\nAll inputs are local. Inference requires image, mask, posterior_noise, and diffusion_noise tensors.\nConversion accepts exported safetensors only; see docs/artifacts.md.\nCompare prints per-stage JSON and exits nonzero for a failed comparison.";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConversionPlan {
    base_unet: PathBuf,
    base_vae: PathBuf,
    checkpoint: PathBuf,
    prompt: PathBuf,
    output_dir: PathBuf,
    base_model_source: SourceProvenance,
    vae_source: SourceProvenance,
    checkpoint_source: SourceProvenance,
    prompt_source: SourceProvenance,
    config_source: Option<SourceProvenance>,
    scheduler: SchedulerConfig,
}

fn read_json<T: serde::de::DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return Err(Error::InvalidInput("JSON input exceeds 1 MiB".into()));
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| Error::InvalidInput(format!("input is missing tensor {name}")))
}

fn run(args: Vec<OsString>) -> Result<()> {
    let Some(command) = args.first().and_then(|argument| argument.to_str()) else {
        return Err(Error::InvalidInput(USAGE.into()));
    };
    match command {
        "--help" | "-h" => {
            println!("{USAGE}");
            Ok(())
        }
        "convert" if args.len() == 2 => {
            let path = args
                .get(1)
                .ok_or_else(|| Error::InvalidInput(USAGE.into()))?;
            let plan: ConversionPlan = read_json(path)?;
            let manifest = convert(&ConversionRequest {
                base_unet: plan.base_unet,
                base_vae: plan.base_vae,
                checkpoint: plan.checkpoint,
                prompt: plan.prompt,
                output_dir: plan.output_dir,
                base_model_source: plan.base_model_source,
                vae_source: plan.vae_source,
                checkpoint_source: plan.checkpoint_source,
                prompt_source: plan.prompt_source,
                config_source: plan.config_source,
                scheduler: plan.scheduler,
                limits: ArtifactLimits::default(),
            })?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
            Ok(())
        }
        "infer" if (4..=6).contains(&args.len()) => {
            let bundle = args
                .get(1)
                .ok_or_else(|| Error::InvalidInput(USAGE.into()))?;
            let input_path = args
                .get(2)
                .ok_or_else(|| Error::InvalidInput(USAGE.into()))?;
            let output_path = args
                .get(3)
                .ok_or_else(|| Error::InvalidInput(USAGE.into()))?;
            if Path::new(output_path).exists()
                || args.get(5).is_some_and(|path| Path::new(path).exists())
            {
                return Err(Error::InvalidInput(
                    "output and stage captures must be new files".into(),
                ));
            }
            let backend = match args.get(4).map(|value| value.to_str()) {
                None | Some(Some("cpu")) => Backend::Cpu,
                Some(Some("metal")) => Backend::Metal { ordinal: 0 },
                Some(Some("cuda")) => Backend::Cuda { ordinal: 0 },
                Some(Some("wgpu")) => Backend::Wgpu { ordinal: 0 },
                _ => {
                    return Err(Error::InvalidInput(
                        "backend must be cpu, metal, cuda, or wgpu".into(),
                    ));
                }
            };
            let start = Instant::now();
            let model = Osor::load_with_options(
                bundle,
                LoadOptions {
                    backend,
                    ..LoadOptions::default()
                },
            )?;
            model.device().synchronize()?;
            let load_ms = start.elapsed().as_secs_f64() * 1000.0;
            let inputs = tensor_io::load(input_path, model.device())?;
            let inputs = inputs
                .into_iter()
                .map(|(name, tensor)| Ok((name, tensor.to_dtype(DType::F32)?)))
                .collect::<Result<HashMap<_, _>>>()?;
            if inputs.len() != 4 {
                return Err(Error::InvalidInput(
                    "input must contain exactly image, mask, posterior_noise, diffusion_noise"
                        .into(),
                ));
            }
            let input = InferenceInput {
                image: tensor(&inputs, "image")?,
                mask: tensor(&inputs, "mask")?,
                posterior_noise: tensor(&inputs, "posterior_noise")?,
                diffusion_noise: tensor(&inputs, "diffusion_noise")?,
            };
            let mut stages = HashMap::new();
            let mut timings = Vec::new();
            let start = Instant::now();
            let mut previous = start;
            let result = model.infer_with_observer(input, &mut |stage, tensor| {
                model.device().synchronize()?;
                let now = Instant::now();
                timings.push(osor_rs::runtime::StageTiming {
                    stage: stage.into(),
                    elapsed_ms: now.duration_since(previous).as_secs_f64() * 1000.0,
                });
                previous = now;
                if args.len() == 6 {
                    stages.insert(stage.to_owned(), tensor.clone());
                }
                Ok(())
            })?;
            model.device().synchronize()?;
            let inference_ms = start.elapsed().as_secs_f64() * 1000.0;
            tensor_io::save_new(
                &HashMap::from([
                    ("image".into(), result.image),
                    ("alpha".into(), result.alpha),
                    ("latent_alpha".into(), result.latent_alpha),
                ]),
                output_path,
            )?;
            if let Some(path) = args.get(5) {
                tensor_io::save_new(&stages, path)?;
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "backend":format!("{backend:?}"), "dtype":"F32", "load_ms":load_ms,
                    "inference_ms":inference_ms, "stages":timings,
                    "upstream_revision":osor_rs::UPSTREAM_REVISION,
                    "model_sha256":model.manifest().sha256,
                    "compute_device":model.compute_device_description(),
                    "profiling_synchronizes_each_stage":true,
                }))?
            );
            Ok(())
        }
        "compare" if args.len() == 4 => {
            let expected = tensor_io::load(
                args.get(1)
                    .ok_or_else(|| Error::InvalidInput(USAGE.into()))?,
                &osor_rs::Device::Cpu,
            )?;
            let actual = tensor_io::load(
                args.get(2)
                    .ok_or_else(|| Error::InvalidInput(USAGE.into()))?,
                &osor_rs::Device::Cpu,
            )?;
            let tolerances = read_json::<std::collections::BTreeMap<String, Tolerance>>(
                args.get(3)
                    .ok_or_else(|| Error::InvalidInput(USAGE.into()))?,
            )?;
            let reports = compare_reference_stages(&expected, &actual, &tolerances)?;
            println!("{}", serde_json::to_string_pretty(&reports)?);
            if reports.iter().any(|report| !report.passed) {
                return Err(Error::InvalidInput(
                    "stage comparison failed; see per-stage report".into(),
                ));
            }
            Ok(())
        }
        _ => Err(Error::InvalidInput(USAGE.into())),
    }
}

fn main() -> std::process::ExitCode {
    match run(std::env::args_os().skip(1).collect()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
