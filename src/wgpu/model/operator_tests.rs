//! Small checkpoint-backed operator comparisons; none run without an explicit GPU request.
use std::collections::HashMap;

use burn::tensor::Device as GpuDevice;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::stable_diffusion::{
    attention::CrossAttention,
    embeddings::{TimestepEmbedding, Timesteps},
};

use super::super::{Weights, acquire_device, guarded, readback, upload};
use super::layers::{
    Attention, Conv, FixedCrossAttention, GroupNorm, LayerNorm, Linear, TimeEmbedding,
};
use crate::Result;

type Checkpoint = HashMap<String, Tensor>;

fn insert(map: &mut Checkpoint, name: &str, shape: &[usize], values: Vec<f32>) -> Result<()> {
    map.insert(
        name.to_owned(),
        Tensor::from_vec(values, shape, &Device::Cpu)?,
    );
    Ok(())
}

// Fixed, small tensors with mixed signs and non-symmetric rows exercise checkpoint orientation.
fn values(count: usize, phase: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 7 + phase) % 19) as f32 - 9.) / 13.)
        .collect()
}

fn compare(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance,
            "element {index}: GPU {a}, CPU {e}, absolute tolerance {tolerance}"
        );
    }
    Ok(())
}

fn cpu(values: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
    Ok(Tensor::from_vec(values, shape, &Device::Cpu)?)
}

fn weight_map(map: Checkpoint, device: &GpuDevice) -> Weights {
    Weights::new(map, device)
}

#[test]
#[ignore = "requires a real WGPU adapter"]
fn gpu_checkpoint_linear_and_conv_match_known_values() -> Result<()> {
    guarded("operator linear and convolution test", || {
        let (device, _) = acquire_device(0)?;
        let mut map = Checkpoint::new();
        insert(
            &mut map,
            "linear.weight",
            &[2, 3],
            vec![1., 2., -1., -2., 0.5, 3.],
        )?;
        insert(&mut map, "linear.bias", &[2], vec![0.5, -1.])?;
        insert(
            &mut map,
            "conv.weight",
            &[2, 1, 2, 2],
            vec![1., 0., 0., -1., 0., 1., 2., 0.],
        )?;
        insert(&mut map, "conv.bias", &[2], vec![0.5, -1.])?;
        let mut weights = weight_map(map, &device);
        let linear = Linear::new(&mut weights, "linear", 3, 2, true)?;
        let input = cpu(vec![1., 2., 3., -1., 0.5, 2.], &[2, 3])?;
        let result = readback(
            linear.forward(upload(&input, "linear input", [2, 3], &device)?),
            [2, 2],
            "linear",
        )?;
        compare(&result, &cpu(vec![2.5, 7., -1.5, 7.25], &[2, 2])?, 1e-6)?;

        let conv = Conv::new(&mut weights, "conv", 1, 2, 2, 1, 0, false)?;
        let input = cpu((1..=9).map(|i| i as f32).collect(), &[1, 1, 3, 3])?;
        let result = readback(
            conv.forward(upload(&input, "conv input", [1, 1, 3, 3], &device)?),
            [1, 2, 2, 2],
            "conv",
        )?;
        compare(
            &result,
            &cpu(
                vec![-3.5, -3.5, -3.5, -3.5, 9., 12., 18., 21.],
                &[1, 2, 2, 2],
            )?,
            1e-6,
        )
    })
}

#[test]
#[ignore = "requires a real WGPU adapter"]
fn gpu_checkpoint_normalization_matches_reference() -> Result<()> {
    guarded("operator normalization test", || {
        let (device, _) = acquire_device(0)?;
        let mut map = Checkpoint::new();
        insert(&mut map, "group.weight", &[32], values(32, 1))?;
        insert(&mut map, "group.bias", &[32], values(32, 5))?;
        insert(&mut map, "layer.weight", &[4], values(4, 3))?;
        insert(&mut map, "layer.bias", &[4], values(4, 8))?;
        let vb = VarBuilder::from_tensors(map.clone(), DType::F32, &Device::Cpu);
        // Close pairs make epsilon significant; each of 32 groups has two spatial values.
        let group_input = cpu(
            (0..32)
                .flat_map(|i| [i as f32 / 10., i as f32 / 10. + 0.01])
                .collect(),
            &[1, 32, 1, 2],
        )?;
        let expected =
            candle_nn::group_norm(32, 32, 1e-3, vb.pp("group"))?.forward(&group_input)?;
        let layer_input = cpu(vec![0.5, 0.501, 0.498, 0.502, -2., 1., 0.5, 4.], &[1, 2, 4])?;
        // Captured with PyTorch 2.7.1 F.layer_norm, epsilon=1e-5. Candle's
        // contiguous CPU fast path uses E[x²]-E[x]² and loses precision on
        // the nearly constant first row, so it is unsuitable as this oracle.
        let layer_expected = cpu(
            vec![
                -0.043_873_99,
                0.478_064_33,
                -0.858_153_8,
                -0.077_316_5,
                0.545_361_16,
                0.466_047_76,
                -0.569_761_8,
                -0.374_007_5,
            ],
            &[1, 2, 4],
        )?;
        let mut weights = weight_map(map, &device);
        let group = GroupNorm::new(&mut weights, "group", 32, 1e-3)?;
        let layer = LayerNorm::new(&mut weights, "layer", 4)?;
        let group_output = readback(
            group.forward(upload(&group_input, "group input", [1, 32, 1, 2], &device)?),
            [1, 32, 1, 2],
            "group norm",
        )?;
        let layer_output = readback(
            layer.forward(upload(&layer_input, "layer input", [1, 2, 4], &device)?),
            [1, 2, 4],
            "layer norm",
        )?;
        // These absolute F32 tolerances allow reduction rounding, including near-zero variance.
        compare(&group_output, &expected, 2e-5)?;
        compare(&layer_output, &layer_expected, 2e-5)
    })
}

#[test]
#[ignore = "requires a real GPU and the measured large-normalization fixtures"]
fn gpu_large_group_norm_matches_reference_fixture() -> Result<()> {
    guarded("large group normalization", || {
        let path = |name| {
            std::env::var_os(name)
                .ok_or_else(|| crate::Error::InvalidInput(format!("set {name} to a local fixture")))
        };
        let inputs = crate::tensor_io::load(path("OSOR_GN_TEST_INPUT")?, &Device::Cpu)?;
        let expected = crate::tensor_io::load(path("OSOR_GN_TEST_REFERENCE")?, &Device::Cpu)?;
        let get = |name: &str| {
            inputs.get(name).ok_or_else(|| {
                crate::Error::InvalidInput(format!("large normalization input lacks {name}"))
            })
        };
        let source = get("input")?;
        let reference = expected.get("output").ok_or_else(|| {
            crate::Error::InvalidInput("large normalization reference lacks output".into())
        })?;
        let shape = [1, 128, 368, 704];
        if source.dims() != shape || reference.dims() != shape {
            return Err(crate::Error::InvalidInput(
                "unexpected normalization fixture shape".into(),
            ));
        }
        let map = Checkpoint::from([
            ("group.weight".into(), get("weight")?.clone()),
            ("group.bias".into(), get("bias")?.clone()),
        ]);
        let (device, _) = acquire_device(0)?;
        let mut weights = weight_map(map, &device);
        let norm = GroupNorm::new(&mut weights, "group", 128, 1e-6)?;
        let output = readback(
            norm.forward(upload(source, "large group input", shape, &device)?),
            shape,
            "large group norm",
        )?;
        let report = crate::parity::compare_tensor(
            "group_norm",
            reference,
            &output,
            crate::parity::Tolerance {
                absolute: 1e-5,
                relative: 0.0,
                maximum_root_mean_square_error: None,
            },
        )?;
        println!("{}", serde_json::to_string_pretty(&report)?);
        assert!(
            report.passed,
            "large group normalization differs from reference"
        );
        Ok(())
    })
}

#[test]
#[ignore = "requires a real WGPU adapter"]
fn gpu_checkpoint_attention_matches_candle_across_query_slice() -> Result<()> {
    guarded("operator attention test", || {
        let (device, _) = acquire_device(0)?;
        let mut map = Checkpoint::new();
        // Width 4, context width 3, two heads; 65 queries cross the slice size 64.
        insert(&mut map, "attention.to_q.weight", &[4, 4], values(16, 1))?;
        insert(&mut map, "attention.to_k.weight", &[4, 3], values(12, 4))?;
        insert(&mut map, "attention.to_v.weight", &[4, 3], values(12, 8))?;
        insert(
            &mut map,
            "attention.to_out.0.weight",
            &[4, 4],
            values(16, 11),
        )?;
        insert(&mut map, "attention.to_out.0.bias", &[4], values(4, 15))?;
        let vb = VarBuilder::from_tensors(map.clone(), DType::F32, &Device::Cpu);
        let reference = CrossAttention::new(vb.pp("attention"), 4, Some(3), 2, 2, None, false)?;
        let query = cpu(values(65 * 4, 2), &[1, 65, 4])?;
        let context = cpu(values(5 * 3, 6), &[1, 5, 3])?;
        let expected = reference.forward(&query, Some(&context))?;
        let mut weights = weight_map(map, &device);
        // Force 64-query slices: one batch, two heads and five context tokens.
        let attention =
            Attention::new(&mut weights, "attention", 4, 3, 2, false)?.with_score_limit(2 * 5 * 64);
        let actual = attention.forward(
            upload(&query, "attention query", [1, 65, 4], &device)?,
            upload(&context, "attention context", [1, 5, 3], &device)?,
        );
        let actual = readback(actual, [1, 65, 4], "attention")?;
        // A short F32 matmul/softmax chain permits backend rounding, not structural errors.
        compare(&actual, &expected, 3e-5)
    })
}

#[test]
#[ignore = "requires a real WGPU adapter"]
fn gpu_fixed_cross_attention_matches_candle_on_the_fly_attention() -> Result<()> {
    guarded("operator fixed cross-attention test", || {
        let (device, _) = acquire_device(0)?;
        let mut map = Checkpoint::new();
        insert(&mut map, "attention.to_q.weight", &[4, 4], values(16, 1))?;
        insert(&mut map, "attention.to_k.weight", &[4, 3], values(12, 4))?;
        insert(&mut map, "attention.to_v.weight", &[4, 3], values(12, 8))?;
        insert(
            &mut map,
            "attention.to_out.0.weight",
            &[4, 4],
            values(16, 11),
        )?;
        insert(&mut map, "attention.to_out.0.bias", &[4], values(4, 15))?;
        let vb = VarBuilder::from_tensors(map.clone(), DType::F32, &Device::Cpu);
        let reference = CrossAttention::new(vb.pp("attention"), 4, Some(3), 2, 2, None, false)?;
        let query = cpu(values(65 * 4, 2), &[1, 65, 4])?;
        let context = cpu(values(5 * 3, 6), &[1, 5, 3])?;
        let expected = reference.forward(&query, Some(&context))?;
        let mut weights = weight_map(map, &device);
        let gpu_context = upload(&context, "attention context", [1, 5, 3], &device)?;
        let attention = FixedCrossAttention::new(&mut weights, "attention", 4, &gpu_context, 2)?;
        let actual = attention.forward(upload(&query, "attention query", [1, 65, 4], &device)?);
        let actual = readback(actual, [1, 65, 4], "fixed cross-attention")?;
        compare(&actual, &expected, 3e-5)
    })
}

#[test]
#[ignore = "requires a real WGPU adapter"]
fn gpu_checkpoint_time_embedding_matches_candle_cpu() -> Result<()> {
    guarded("operator time embedding test", || {
        let (device, _) = acquire_device(0)?;
        let mut map = Checkpoint::new();
        insert(&mut map, "time.linear_1.weight", &[6, 4], values(24, 2))?;
        insert(&mut map, "time.linear_1.bias", &[6], values(6, 4))?;
        insert(&mut map, "time.linear_2.weight", &[6, 6], values(36, 7))?;
        insert(&mut map, "time.linear_2.bias", &[6], values(6, 12))?;
        let vb = VarBuilder::from_tensors(map.clone(), DType::F32, &Device::Cpu);
        let projector = Timesteps::new(4, true, 0.);
        let reference = TimestepEmbedding::new(vb.pp("time"), 4, 6)?;
        let times = cpu(vec![0., 1., 400.], &[3])?;
        let projected = projector.forward(&times)?;
        let expected = reference.forward(&projected)?;
        let mut weights = weight_map(map, &device);
        let embedding = TimeEmbedding::new(&mut weights, "time", 4, 4, 6, &device)?;
        let gpu_projected = embedding.project(upload(&times, "timesteps", [3], &device)?);
        let result = embedding.forward(gpu_projected.clone());
        let gpu_projected = readback(gpu_projected, [3, 4], "time projection")?;
        let result = readback(result, [3, 6], "time embedding")?;
        // Trigonometry and two F32 linear operations can round differently across devices.
        compare(&gpu_projected, &projected, 3e-5)?;
        compare(&result, &expected, 3e-5)
    })
}
