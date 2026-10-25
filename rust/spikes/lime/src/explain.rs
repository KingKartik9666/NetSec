use crate::{neighborhood, reference::ReferenceData, regression, Error, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    pub feature: String,
    pub feature_index: usize,
    pub condition: String,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Explanation {
    pub method: String,
    pub class_label: String,
    pub target_class: usize,
    pub seed: u64,
    pub rng: String,
    pub perturbations: usize,
    pub reference_rows: usize,
    pub observed_features: usize,
    pub missing_features: usize,
    pub perturbation_constraints: Option<String>,
    pub model_score: f64,
    pub surrogate_prediction: f64,
    pub local_fidelity_r2: f64,
    pub intercept: f64,
    pub conditions: Vec<Condition>,
    pub neighborhood_target_min: f64,
    pub neighborhood_target_max: f64,
    pub constant_target: bool,
    pub feature_selection: String,
    pub limitations: String,
}

pub fn explain<F>(
    reference: &ReferenceData,
    row: &[Option<f32>],
    classes: &[String],
    packet: bool,
    seed: u64,
    samples: usize,
    mut predict: F,
) -> Result<Explanation>
where
    F: FnMut(&[Option<f32>]) -> std::result::Result<Vec<f32>, String>,
{
    if classes.is_empty() {
        return Err(Error::Invalid("LIME requires class labels".into()));
    }
    let neighborhood = neighborhood::generate(reference, row, packet, seed, samples)?;
    let checked = |scores: Vec<f32>| -> Result<Vec<f32>> {
        if scores.len() != classes.len()
            || scores
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0 || *v > 1.0)
        {
            return Err(Error::Invalid(
                "Model port returned invalid class probabilities".into(),
            ));
        }
        Ok(scores)
    };
    let anchor_scores = checked(
        predict(&neighborhood.model_rows[0])
            .map_err(|error| Error::Invalid(format!("Model prediction failed: {error}")))?,
    )?;
    let mut target = 0;
    for (i, score) in anchor_scores.iter().enumerate().skip(1) {
        if *score > anchor_scores[target] {
            target = i;
        }
    }
    let mut labels = vec![anchor_scores[target] as f64];
    for neighbor in neighborhood.model_rows.iter().skip(1) {
        let scores = checked(
            predict(neighbor)
                .map_err(|error| Error::Invalid(format!("Model prediction failed: {error}")))?,
        )?;
        labels.push(scores[target] as f64);
    }
    let maximum = neighborhood.observed.len().min(12);
    let selected = regression::select_features(
        &neighborhood.binary,
        &labels,
        &neighborhood.weights,
        maximum,
    )?;
    let data: Vec<Vec<f64>> = neighborhood
        .binary
        .iter()
        .map(|r| selected.iter().map(|i| r[*i]).collect())
        .collect();
    let fit = regression::ridge(&data, &labels, &neighborhood.weights, 1.0)?;
    let surrogate_prediction = fit.intercept
        + fit
            .coefficients
            .iter()
            .zip(&data[0])
            .map(|(b, x)| b * x)
            .sum::<f64>();
    let mut conditions: Vec<Condition> = selected
        .iter()
        .zip(&fit.coefficients)
        .map(|(local, weight)| {
            let original = neighborhood.observed[*local];
            Condition {
                feature: reference.metadata.feature_names[original].clone(),
                feature_index: original,
                condition: neighborhood.conditions[*local].clone(),
                weight: *weight,
            }
        })
        .collect();
    conditions.sort_by(|a, b| b.weight.abs().total_cmp(&a.weight.abs()));
    let min = labels.iter().copied().fold(f64::INFINITY, f64::min);
    let max = labels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(Explanation {
        method: "Seeded Rust LIME local linear surrogate (Phase 1 spike)".into(),
        class_label: classes[target].clone(), target_class: target, seed, rng: "ChaCha8Rng 0.3.1".into(),
        perturbations: samples, reference_rows: reference.training.len(), observed_features: neighborhood.observed.len(),
        missing_features: row.len() - neighborhood.observed.len(),
        perturbation_constraints: neighborhood.constrained.then(|| "integer packet and TCP-flag counts".into()),
        model_score: anchor_scores[target] as f64, surrogate_prediction, local_fidelity_r2: fit.r2, intercept: fit.intercept, conditions,
        neighborhood_target_min: min, neighborhood_target_max: max, constant_target: min == max,
        feature_selection: if maximum <= 6 { "auto/forward_selection (alpha=0)" } else { "auto/highest_weights (alpha=0.01)" }.into(),
        limitations: "Approximate local surrogate on synthetic perturbations; not observed traffic, causal evidence, detector accuracy or the Python lime package. Missing numeric and categorical inputs remain absent from all model calls. RNG and float arithmetic differ from NumPy/SciPy/sklearn. Exactly constant neighborhoods use zero weights and R2=1; flat attribution ordering is not informative. Inspect R2 and anchor-versus-surrogate disagreement.".into(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::reference::ReferenceMetadata;

    fn fixture() -> ReferenceData {
        ReferenceData {
            metadata: ReferenceMetadata {
                feature_names: vec!["binary_evidence".into(), "missing_numeric".into()],
                categorical_indices: vec![0],
                rows: 8,
                sha256: String::new(),
                kind: "synthetic test only".into(),
            },
            training: (0..8)
                .map(|i| vec![Some((i % 2) as f32), Some(i as f32)])
                .collect(),
            medians: vec![0.0, 3.5],
        }
    }

    // A substitute prediction or imputed unobserved value would produce false evidence.
    #[test]
    fn injected_predictor_failure_returns_error_and_does_not_substitute() {
        let error = explain(
            &fixture(),
            &[Some(1.0), None],
            &["a".into(), "b".into()],
            false,
            42,
            128,
            |_| Err("model unavailable".into()),
        )
        .unwrap_err();
        assert!(error.to_string().contains("model unavailable"));
    }

    #[test]
    fn real_surrogate_recovers_binary_class_effect_and_keeps_missing_absent() {
        let result = explain(
            &fixture(),
            &[Some(1.0), None],
            &["a".into(), "b".into()],
            false,
            42,
            512,
            |row| {
                if row[1].is_some() {
                    return Err("missing input invented".into());
                }
                let score = 0.1 + 0.8 * row[0].unwrap();
                Ok(vec![1.0 - score, score])
            },
        )
        .unwrap();
        assert_eq!(result.class_label, "b");
        assert_eq!(result.observed_features, 1);
        assert_eq!(result.conditions.len(), 1);
        assert_eq!(result.conditions[0].condition, "binary_evidence=1");
        assert!(result.conditions[0].weight > 0.75);
        assert!(result.local_fidelity_r2 > 0.995);
        assert!((result.surrogate_prediction - 0.9).abs() < 0.02);
    }
}
