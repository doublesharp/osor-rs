//! Query-row slicing shared by native UNet and VAE attention.

use candle_core::Tensor;

use crate::{Result, pipeline::Checkpoint};

/// Bound for each F32 score/probability tensor, excluding other activations.
pub(super) const MAX_SCORE_ELEMENTS: usize = 64 * 1024 * 1024;

pub(super) fn attend(
    query: &Tensor,
    keys_t: &Tensor,
    values: &Tensor,
    checkpoint: Checkpoint<'_>,
    softmax: impl Fn(&Tensor) -> candle_core::Result<Tensor>,
) -> Result<Tensor> {
    attend_with(
        query,
        keys_t,
        values,
        MAX_SCORE_ELEMENTS,
        checkpoint,
        &mut |q, k, v| Ok(softmax(&q.matmul(k)?)?.matmul(v)?),
    )
}

fn attend_with(
    query: &Tensor,
    keys_t: &Tensor,
    values: &Tensor,
    max_score_elements: usize,
    checkpoint: Checkpoint<'_>,
    evaluate: &mut dyn FnMut(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let rank = query.rank();
    if rank < 2 || keys_t.rank() != rank || values.rank() != rank {
        return Err(crate::Error::InvalidInput(
            "attention tensors must have matching ranks of at least two".into(),
        ));
    }
    let query_axis = rank - 2;
    let queries = query.dim(query_axis)?;
    let tokens = keys_t.dim(rank - 1)?;
    if queries == 0
        || tokens == 0
        || query.dims().get(..query_axis) != keys_t.dims().get(..query_axis)
        || query.dims().get(..query_axis) != values.dims().get(..query_axis)
        || query.dim(rank - 1)? != keys_t.dim(query_axis)?
        || tokens != values.dim(query_axis)?
    {
        return Err(crate::Error::InvalidInput(
            "attention tensor shapes are incompatible".into(),
        ));
    }
    let row_elements = query
        .dims()
        .iter()
        .take(query_axis)
        .try_fold(tokens, |count, &dim| count.checked_mul(dim))
        .filter(|&count| count > 0 && count <= max_score_elements)
        .ok_or_else(|| {
            crate::Error::InvalidInput("attention score budget cannot hold one query row".into())
        })?;
    let rows = (max_score_elements / row_elements).min(queries);
    if rows == queries {
        checkpoint()?;
        return evaluate(query, keys_t, values);
    }
    let mut pieces = Vec::new();
    pieces
        .try_reserve_exact(queries.div_ceil(rows))
        .map_err(|error| {
            crate::Error::Unavailable(format!("cannot allocate attention slices: {error}"))
        })?;
    for start in (0..queries).step_by(rows) {
        checkpoint()?;
        let query = query
            .narrow(query_axis, start, rows.min(queries - start))?
            .contiguous()?;
        pieces.push(evaluate(&query, keys_t, values)?);
    }
    checkpoint()?;
    Ok(Tensor::cat(&pieces, query_axis)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::cell::Cell;

    fn inputs(device: &Device, rank_four: bool) -> Result<(Tensor, Tensor, Tensor)> {
        let q = Tensor::arange(0f32, 6. * 8. * 11., device)?
            .affine(1. / 1024., -0.25)?
            .reshape((6, 8, 11))?
            .transpose(1, 2)?;
        let k = Tensor::arange(0f32, 6. * 8. * 13., device)?
            .affine(1. / 2048., -0.125)?
            .reshape((6, 8, 13))?;
        let v = Tensor::arange(0f32, 6. * 13. * 8., device)?
            .affine(1. / 512., -0.5)?
            .reshape((6, 13, 8))?;
        if rank_four {
            Ok((
                q.reshape((2, 3, 11, 8))?,
                k.reshape((2, 3, 8, 13))?,
                v.reshape((2, 3, 13, 8))?,
            ))
        } else {
            Ok((q, k, v))
        }
    }

    fn check_slicing(device: &Device) -> Result<()> {
        for rank_four in [false, true] {
            let (q, k, v) = inputs(device, rank_four)?;
            let expected = candle_nn::ops::softmax_last_dim(&q.matmul(&k)?)?.matmul(&v)?;
            for rows in [1, 3, 11, 20] {
                let budget = 6 * 13 * rows;
                let mut calls = 0;
                let actual = attend_with(&q, &k, &v, budget, &|| Ok(()), &mut |q, k, v| {
                    let scores = q.matmul(k)?;
                    assert!(
                        scores.elem_count() <= budget,
                        "attention score allocation exceeds budget"
                    );
                    calls += 1;
                    Ok(candle_nn::ops::softmax_last_dim(&scores)?.matmul(v)?)
                })?;
                assert_eq!(calls, 11usize.div_ceil(rows));
                assert_eq!(actual.dims(), expected.dims());
                let error = (actual - &expected)?
                    .abs()?
                    .flatten_all()?
                    .max(0)?
                    .to_device(&Device::Cpu)?
                    .to_scalar::<f32>()?;
                assert!(error <= 2e-6, "query slicing changed attention by {error}");
            }
        }
        Ok(())
    }

    #[test]
    fn native_attention_slices_queries_within_score_budget() -> Result<()> {
        check_slicing(&Device::Cpu)
    }

    #[test]
    fn native_attention_cancels_before_the_next_query_slice() -> Result<()> {
        let (q, k, v) = inputs(&Device::Cpu, false)?;
        let checks = Cell::new(0);
        let mut evaluations = 0;
        let result = attend_with(
            &q,
            &k,
            &v,
            6 * 13 * 3,
            &|| {
                checks.set(checks.get() + 1);
                if checks.get() == 2 {
                    Err(crate::Error::Cancelled)
                } else {
                    Ok(())
                }
            },
            &mut |q, k, v| {
                evaluations += 1;
                Ok(candle_nn::ops::softmax_last_dim(&q.matmul(k)?)?.matmul(v)?)
            },
        );
        assert!(matches!(result, Err(crate::Error::Cancelled)));
        assert_eq!(evaluations, 1, "a cancelled slice must not execute");
        Ok(())
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires a real Metal device"]
    fn metal_attention_slices_queries_within_score_budget() -> Result<()> {
        check_slicing(&Device::new_metal(0)?)
    }

    #[test]
    fn native_attention_rejects_a_budget_smaller_than_one_score_row() -> Result<()> {
        let (q, k, v) = inputs(&Device::Cpu, false)?;
        let evaluated = Cell::new(false);
        let result = attend_with(&q, &k, &v, 6 * 13 - 1, &|| Ok(()), &mut |_, _, _| {
            evaluated.set(true);
            Tensor::zeros((6, 11, 8), DType::F32, &Device::Cpu).map_err(Into::into)
        });
        assert!(matches!(result, Err(crate::Error::InvalidInput(_))));
        assert!(!evaluated.get());
        Ok(())
    }
}
