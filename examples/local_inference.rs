//! Run CPU inference from a local bundle and an explicit safetensors capture.
//! See `docs/platforms.md` for the input contract and invocation.

use std::{collections::HashMap, env, path::PathBuf};

use osor::{Error, InferenceInput, Osor, Result, Tensor, tensor_io};

const USAGE: &str = "Usage: local_inference BUNDLE INPUT.safetensors OUTPUT.safetensors\n\
INPUT must contain image, mask, posterior_noise and diffusion_noise.\n\
All files are local; OUTPUT must not already exist.";

fn required_tensor<'a>(inputs: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    inputs
        .get(name)
        .ok_or_else(|| Error::InvalidInput(format!("input capture is missing tensor {name}")))
}

fn run() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let Some(first) = args.next() else {
        return Err(Error::InvalidInput(USAGE.into()));
    };
    if first == "--help" || first == "-h" {
        println!("{USAGE}");
        return Ok(());
    }
    let input_path = args
        .next()
        .ok_or_else(|| Error::InvalidInput(USAGE.into()))?;
    let output_path = args
        .next()
        .ok_or_else(|| Error::InvalidInput(USAGE.into()))?;
    if args.next().is_some() {
        return Err(Error::InvalidInput(USAGE.into()));
    }

    let model = Osor::load(PathBuf::from(first))?;
    let inputs = tensor_io::load(PathBuf::from(input_path), model.device())?;
    let result = model.infer(InferenceInput {
        image: required_tensor(&inputs, "image")?,
        mask: required_tensor(&inputs, "mask")?,
        posterior_noise: required_tensor(&inputs, "posterior_noise")?,
        diffusion_noise: required_tensor(&inputs, "diffusion_noise")?,
    })?;
    let outputs = HashMap::from([
        ("image".to_owned(), result.image),
        ("alpha".to_owned(), result.alpha),
    ]);
    tensor_io::save_new(&outputs, PathBuf::from(output_path))
}

fn main() -> Result<()> {
    run()
}
