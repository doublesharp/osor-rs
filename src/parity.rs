//! Deterministic, stage-specific comparison of captured reference tensors.
//!
//! Tolerances must come from a measured reference comparison. This module has no
//! permissive default and never substitutes an RNG seed for captured noise.

use std::collections::{BTreeMap, HashMap};

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

const MAX_COMPARISON_ELEMENTS: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tolerance {
    pub absolute: f64,
    pub relative: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_root_mean_square_error: Option<f64>,
}

impl Tolerance {
    fn validate(self) -> Result<()> {
        if !self.absolute.is_finite()
            || !self.relative.is_finite()
            || self.absolute < 0.0
            || self.relative < 0.0
            || self
                .maximum_root_mean_square_error
                .is_some_and(|maximum| !maximum.is_finite() || maximum < 0.0)
        {
            return Err(Error::InvalidInput(
                "comparison tolerances must be finite and nonnegative".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StageComparison {
    pub stage: String,
    pub shape: Vec<usize>,
    pub reference_dtype: String,
    pub actual_dtype: String,
    pub elements: usize,
    pub max_absolute_error: f64,
    pub mean_absolute_error: f64,
    pub root_mean_square_error: f64,
    pub root_mean_square_error_passed: bool,
    pub max_relative_error: f64,
    pub violations: usize,
    pub tolerance: Tolerance,
    pub passed: bool,
}

/// Compare every element using `abs(actual-reference) <= absolute + relative*abs(reference)`.
/// An optional RMS cap also applies to the whole stage. Nonfinite values fail
/// validation even if they occur at the same position.
pub fn compare_tensor(
    stage: &str,
    reference: &Tensor,
    actual: &Tensor,
    tolerance: Tolerance,
) -> Result<StageComparison> {
    tolerance.validate()?;
    if reference.dims() != actual.dims() {
        return Err(Error::InvalidInput(format!(
            "stage {stage}: shape mismatch {:?} versus {:?}",
            reference.dims(),
            actual.dims()
        )));
    }
    let elements = reference.elem_count();
    if elements == 0 || elements > MAX_COMPARISON_ELEMENTS {
        return Err(Error::InvalidInput(format!(
            "stage {stage}: comparison requires 1..={MAX_COMPARISON_ELEMENTS} elements"
        )));
    }
    if !matches!(
        reference.dtype(),
        DType::F32 | DType::F16 | DType::BF16 | DType::F64
    ) || !matches!(
        actual.dtype(),
        DType::F32 | DType::F16 | DType::BF16 | DType::F64
    ) {
        return Err(Error::InvalidInput(format!(
            "stage {stage}: expected floating point tensors"
        )));
    }
    let reference_values = reference
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?;
    let actual_values = actual
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?;
    let mut max_absolute_error = 0.0_f64;
    let mut max_relative_error = 0.0_f64;
    let mut absolute_sum = 0.0_f64;
    let mut squared_sum = 0.0_f64;
    let mut violations = 0;
    for (expected, observed) in reference_values.iter().zip(&actual_values) {
        if !expected.is_finite() || !observed.is_finite() {
            return Err(Error::InvalidInput(format!(
                "stage {stage}: captured tensors contain nonfinite values"
            )));
        }
        let error = (observed - expected).abs();
        let allowed = tolerance.absolute + tolerance.relative * expected.abs();
        let relative = error / expected.abs().max(f64::EPSILON);
        if !error.is_finite() || !allowed.is_finite() || !relative.is_finite() {
            return Err(Error::InvalidInput(format!(
                "stage {stage}: error or tolerance arithmetic exceeds the finite comparison range"
            )));
        }
        max_absolute_error = max_absolute_error.max(error);
        // A zero reference has no useful relative error; report its absolute
        // error above and use a finite denominator for the diagnostic metric.
        max_relative_error = max_relative_error.max(relative);
        absolute_sum += error;
        squared_sum += error * error;
        if !absolute_sum.is_finite() || !squared_sum.is_finite() {
            return Err(Error::InvalidInput(format!(
                "stage {stage}: aggregate error metrics exceed the finite comparison range"
            )));
        }
        if error > allowed {
            violations += 1;
        }
    }
    let root_mean_square_error = (squared_sum / elements as f64).sqrt();
    let root_mean_square_error_passed = tolerance
        .maximum_root_mean_square_error
        .is_none_or(|maximum| root_mean_square_error <= maximum);
    Ok(StageComparison {
        stage: stage.into(),
        shape: reference.dims().to_vec(),
        reference_dtype: format!("{:?}", reference.dtype()),
        actual_dtype: format!("{:?}", actual.dtype()),
        elements,
        max_absolute_error,
        mean_absolute_error: absolute_sum / elements as f64,
        root_mean_square_error,
        root_mean_square_error_passed,
        max_relative_error,
        violations,
        tolerance,
        passed: violations == 0 && root_mean_square_error_passed,
    })
}

/// Exact stage-set comparison: missing stages and missing tolerances are errors.
pub fn compare_stages(
    reference: &HashMap<String, Tensor>,
    actual: &HashMap<String, Tensor>,
    tolerances: &BTreeMap<String, Tolerance>,
) -> Result<Vec<StageComparison>> {
    if reference.is_empty()
        || reference.len() != actual.len()
        || reference.len() != tolerances.len()
        || reference
            .keys()
            .any(|key| !actual.contains_key(key) || !tolerances.contains_key(key))
    {
        return Err(Error::InvalidInput(
            "reference, actual, and tolerances must contain exactly the same nonempty stage set"
                .into(),
        ));
    }
    tolerances
        .iter()
        .map(|(stage, tolerance)| {
            let expected = reference.get(stage).ok_or_else(|| {
                Error::InvalidInput(format!("reference capture is missing stage {stage}"))
            })?;
            let observed = actual.get(stage).ok_or_else(|| {
                Error::InvalidInput(format!("actual capture is missing stage {stage}"))
            })?;
            compare_tensor(stage, expected, observed, *tolerance)
        })
        .collect()
}

/// Compare the complete pinned reference stage set. Partial diagnostic captures
/// must use `compare_stages` and cannot produce a full-reference verdict.
pub fn compare_reference_stages(
    reference: &HashMap<String, Tensor>,
    actual: &HashMap<String, Tensor>,
    tolerances: &BTreeMap<String, Tolerance>,
) -> Result<Vec<StageComparison>> {
    if reference.len() != crate::pipeline::STAGES.len()
        || crate::pipeline::STAGES
            .iter()
            .any(|stage| !reference.contains_key(*stage))
    {
        return Err(Error::InvalidInput(
            "full reference comparison requires every pinned pipeline stage".into(),
        ));
    }
    compare_stages(reference, actual, tolerances)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_counts_elementwise_failures_without_hiding_mean_error() -> Result<()> {
        let reference = Tensor::new(&[1_f32, 2., 3., 4.], &Device::Cpu)?;
        let actual = Tensor::new(&[1_f32, 2., 3., 5.], &Device::Cpu)?;
        let report = compare_tensor(
            "decoded",
            &reference,
            &actual,
            Tolerance {
                absolute: 0.1,
                relative: 0.0,
                maximum_root_mean_square_error: None,
            },
        )?;
        assert!(!report.passed);
        assert_eq!(report.violations, 1);
        assert_eq!(report.max_absolute_error, 1.0);
        assert_eq!(report.mean_absolute_error, 0.25);
        assert_eq!(report.root_mean_square_error, 0.5);
        Ok(())
    }

    #[test]
    fn rms_cap_is_required_in_addition_to_elementwise_limits() -> Result<()> {
        let reference = Tensor::new(&[1_f32, 2., 3., 4.], &Device::Cpu)?;
        let actual = Tensor::new(&[1_f32, 2., 3., 5.], &Device::Cpu)?;
        let tolerance = Tolerance {
            absolute: 1.0,
            relative: 0.0,
            maximum_root_mean_square_error: Some(0.49),
        };
        let report = compare_tensor("decoded", &reference, &actual, tolerance)?;
        assert_eq!(report.violations, 0);
        assert_eq!(report.root_mean_square_error, 0.5);
        assert!(!report.root_mean_square_error_passed);
        assert!(!report.passed);
        assert_eq!(
            serde_json::to_value(&report)?["root_mean_square_error_passed"],
            false
        );

        let boundary = compare_tensor(
            "decoded",
            &reference,
            &actual,
            Tolerance {
                maximum_root_mean_square_error: Some(0.5),
                ..tolerance
            },
        )?;
        assert_eq!(boundary.violations, 0);
        assert!(boundary.root_mean_square_error_passed);
        assert!(boundary.passed);

        let element_failure = compare_tensor(
            "decoded",
            &reference,
            &actual,
            Tolerance {
                absolute: 0.5,
                maximum_root_mean_square_error: Some(0.5),
                ..tolerance
            },
        )?;
        assert_eq!(element_failure.violations, 1);
        assert!(element_failure.root_mean_square_error_passed);
        assert!(!element_failure.passed);
        Ok(())
    }

    #[test]
    fn invalid_rms_caps_are_rejected_before_tensor_validation() -> Result<()> {
        let reference = Tensor::new(&[1_f32], &Device::Cpu)?;
        let wrong_shape = reference.reshape((1, 1))?;
        for maximum in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result = compare_tensor(
                "alpha",
                &reference,
                &wrong_shape,
                Tolerance {
                    absolute: 0.0,
                    relative: 0.0,
                    maximum_root_mean_square_error: Some(maximum),
                },
            );
            assert!(matches!(
                result,
                Err(Error::InvalidInput(message))
                    if message == "comparison tolerances must be finite and nonnegative"
            ));
        }
        Ok(())
    }

    #[test]
    fn omitted_rms_json_preserves_elementwise_verdict_and_serialization() -> Result<()> {
        let tolerance: Tolerance = serde_json::from_str(r#"{"absolute":1.0,"relative":0.0}"#)?;
        assert!(tolerance.maximum_root_mean_square_error.is_none());
        assert_eq!(
            serde_json::to_value(tolerance)?,
            serde_json::json!({"absolute": 1.0, "relative": 0.0})
        );
        let capped: Tolerance = serde_json::from_str(
            r#"{"absolute":1.0,"relative":0.0,"maximum_root_mean_square_error":0.5}"#,
        )?;
        assert_eq!(capped.maximum_root_mean_square_error, Some(0.5));
        assert_eq!(
            serde_json::to_value(capped)?["maximum_root_mean_square_error"],
            0.5
        );
        let reference = Tensor::new(&[1_f32, 2.], &Device::Cpu)?;
        let within = Tensor::new(&[1_f32, 3.], &Device::Cpu)?;
        let outside = Tensor::new(&[1_f32, 4.], &Device::Cpu)?;
        let passing = compare_tensor("decoded", &reference, &within, tolerance)?;
        assert!(passing.root_mean_square_error_passed);
        assert_eq!(passing.violations, 0);
        assert!(passing.passed);
        let failing = compare_tensor("decoded", &reference, &outside, tolerance)?;
        assert!(failing.root_mean_square_error_passed);
        assert_eq!(failing.violations, 1);
        assert!(!failing.passed);
        Ok(())
    }

    #[test]
    fn nonfinite_capture_or_tolerance_and_shape_mismatch_fail_closed() -> Result<()> {
        let reference = Tensor::new(&[1_f32], &Device::Cpu)?;
        let invalid = Tensor::new(&[f32::NAN], &Device::Cpu)?;
        let exact = Tolerance {
            absolute: 0.0,
            relative: 0.0,
            maximum_root_mean_square_error: None,
        };
        assert!(compare_tensor("alpha", &reference, &invalid, exact).is_err());
        assert!(compare_tensor("alpha", &reference, &reference.reshape((1, 1))?, exact).is_err());
        assert!(
            compare_tensor(
                "alpha",
                &reference,
                &reference,
                Tolerance {
                    absolute: f64::INFINITY,
                    relative: 0.0,
                    maximum_root_mean_square_error: None,
                }
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn missing_stage_cannot_pass_a_partial_capture() -> Result<()> {
        let reference = HashMap::from([("alpha".into(), Tensor::new(&[0.5_f32], &Device::Cpu)?)]);
        let reports = compare_stages(
            &reference,
            &reference,
            &BTreeMap::from([(
                "alpha".into(),
                Tolerance {
                    absolute: 0.0,
                    relative: 0.0,
                    maximum_root_mean_square_error: None,
                },
            )]),
        )?;
        assert!(reports.iter().all(|stage| stage.passed));
        assert!(
            compare_reference_stages(
                &reference,
                &reference,
                &BTreeMap::from([(
                    "alpha".into(),
                    Tolerance {
                        absolute: 0.0,
                        relative: 0.0,
                        maximum_root_mean_square_error: None,
                    }
                ),])
            )
            .is_err()
        );
        assert!(compare_stages(&reference, &HashMap::new(), &BTreeMap::new()).is_err());
        Ok(())
    }

    #[test]
    fn finite_extremes_cannot_pass_through_overflow() -> Result<()> {
        let reference = Tensor::new(&[f64::MAX], &Device::Cpu)?;
        let actual = Tensor::new(&[-f64::MAX], &Device::Cpu)?;
        assert!(
            compare_tensor(
                "extreme",
                &reference,
                &actual,
                Tolerance {
                    absolute: 0.0,
                    relative: 2.0,
                    maximum_root_mean_square_error: None,
                }
            )
            .is_err()
        );
        let reference = Tensor::new(&[0_f64], &Device::Cpu)?;
        let actual = Tensor::new(&[1e200_f64], &Device::Cpu)?;
        assert!(
            compare_tensor(
                "overflow",
                &reference,
                &actual,
                Tolerance {
                    absolute: f64::MAX,
                    relative: 0.0,
                    maximum_root_mean_square_error: None,
                }
            )
            .is_err()
        );
        Ok(())
    }
}
