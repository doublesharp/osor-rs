use candle_core::{D, Result, Tensor};
use candle_nn::{self as nn, Module, VarBuilder};

#[derive(Debug)]
pub(super) struct FeedForward {
    gate: nn::Linear,
    output: nn::Linear,
}

impl FeedForward {
    pub(super) fn new(vs: VarBuilder<'_>, dim: usize) -> Result<Self> {
        Ok(Self {
            gate: nn::linear(dim, dim * 8, vs.pp("net.0.proj"))?,
            output: nn::linear(dim * 4, dim, vs.pp("net.2"))?,
        })
    }

    pub(super) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let projected = self.gate.forward(xs)?;
        let width = projected.dim(D::Minus1)? / 2;
        let values = projected.narrow(D::Minus1, 0, width)?;
        let gates = projected.narrow(D::Minus1, width, width)?;
        self.output.forward(&(values * gates.gelu_erf()?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::collections::HashMap;

    #[test]
    fn exact_geglu_matches_pytorch_gate_two() -> Result<()> {
        // PyTorch 2.7 F.gelu(torch.tensor(2.0), approximate="none") = 1.9544997215270996.
        let device = &Device::Cpu;
        let mut weights = HashMap::new();
        weights.insert(
            "net.0.proj.weight".into(),
            Tensor::zeros((8, 1), DType::F32, device)?,
        );
        weights.insert(
            "net.0.proj.bias".into(),
            Tensor::new(&[1f32, 0., 0., 0., 2., 0., 0., 0.], device)?,
        );
        weights.insert(
            "net.2.weight".into(),
            Tensor::new(&[[1f32, 0., 0., 0.]], device)?,
        );
        weights.insert("net.2.bias".into(), Tensor::zeros(1, DType::F32, device)?);
        let ff = FeedForward::new(VarBuilder::from_tensors(weights, DType::F32, device), 1)?;
        let actual = ff
            .forward(&Tensor::zeros((1, 1, 1), DType::F32, device)?)?
            .sum_all()?
            .to_scalar::<f32>()?;
        assert!(
            (actual - 1.954_499_7).abs() < 1e-6,
            "GEGLU output {actual} differs from PyTorch"
        );
        Ok(())
    }
}
