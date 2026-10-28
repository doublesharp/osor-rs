//! Measure loading, first inference, warm inference, and synchronized stages.

use std::{collections::HashMap, ffi::OsString, time::Instant};

use osor_rs::{Backend, Error, InferenceInput, LoadOptions, Osor, Result, Tensor, tensor_io};

const USAGE: &str = "osor-profile BUNDLE INPUT.safetensors [cpu|metal|cuda|wgpu] [WARM_RUNS=3]\nWarm runs must be 1..=10. Uses F32 and caller-supplied noise; downloads nothing.\nThe final stage profile synchronizes after each stage; normal inference timings do not.";

fn input(tensors: &HashMap<String, Tensor>) -> Result<InferenceInput<'_>> {
    let get = |name: &str| {
        tensors
            .get(name)
            .ok_or_else(|| Error::InvalidInput(format!("missing input tensor {name}")))
    };
    Ok(InferenceInput {
        image: get("image")?,
        mask: get("mask")?,
        posterior_noise: get("posterior_noise")?,
        diffusion_noise: get("diffusion_noise")?,
    })
}

fn run(mut arguments: impl Iterator<Item = OsString>) -> Result<()> {
    let bundle = arguments
        .next()
        .ok_or_else(|| Error::InvalidInput(USAGE.into()))?;
    if bundle == "--help" || bundle == "-h" {
        println!("{USAGE}");
        return Ok(());
    }
    let input_path = arguments
        .next()
        .ok_or_else(|| Error::InvalidInput(USAGE.into()))?;
    let backend_argument = arguments.next();
    let backend = match backend_argument.as_ref().map(|argument| argument.to_str()) {
        None | Some(Some("cpu")) => Backend::Cpu,
        Some(Some("metal")) => Backend::Metal { ordinal: 0 },
        Some(Some("cuda")) => Backend::Cuda { ordinal: 0 },
        Some(Some("wgpu")) => Backend::Wgpu { ordinal: 0 },
        _ => return Err(Error::InvalidInput(USAGE.into())),
    };
    let warm_argument = arguments.next();
    let warm_runs = match warm_argument {
        Some(argument) => argument
            .to_str()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| (1..=10).contains(value))
            .ok_or_else(|| Error::InvalidInput(USAGE.into()))?,
        None => 3,
    };
    if arguments.next().is_some() {
        return Err(Error::InvalidInput(USAGE.into()));
    }
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
    let tensors = tensor_io::load(input_path, model.device())?;
    if tensors.len() != 4 {
        return Err(Error::InvalidInput(
            "exactly four input tensors are required".into(),
        ));
    }
    let shape = input(&tensors)?.image.dims().to_vec();
    let mut runs = Vec::with_capacity(warm_runs + 1);
    for _ in 0..=warm_runs {
        model.device().synchronize()?;
        let start = Instant::now();
        let result = model.infer(input(&tensors)?)?;
        model.device().synchronize()?;
        runs.push(start.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(result);
    }
    let first_ms = runs
        .first()
        .copied()
        .ok_or_else(|| Error::InvalidInput("empty benchmark".into()))?;
    let warm = runs
        .get(1..)
        .ok_or_else(|| Error::InvalidInput("missing warm runs".into()))?;
    let warm_mean_ms = warm.iter().sum::<f64>() / warm.len() as f64;
    let (_, profile) = model.infer_profiled(input(&tensors)?)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "backend":format!("{backend:?}"), "dtype":"F32", "shape":shape,
            "load_ms":load_ms, "first_inference_ms":first_ms,
            "warm_inference_ms":warm, "warm_mean_ms":warm_mean_ms,
            "stage_profile":profile, "version":env!("CARGO_PKG_VERSION"),
            "model_sha256":model.manifest().sha256,
            "compute_device":model.compute_device_description(),
            "upstream_revision":osor_rs::UPSTREAM_REVISION,
            "timing_scope":"synchronous inference including validation; excludes input file I/O and output serialization",
        }))?
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run(std::env::args_os().skip(1)) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            std::process::ExitCode::FAILURE
        }
    }
}
