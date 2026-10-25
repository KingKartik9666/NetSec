use crate::{reference::ReferenceData, Error, Result};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::HashSet;

pub const PACKET_NAMES: [&str; 7] = [
    "network_packets_all_count",
    "network_tcp-flags-ack_count",
    "network_tcp-flags-fin_count",
    "network_tcp-flags-psh_count",
    "network_tcp-flags-rst_count",
    "network_tcp-flags-syn_count",
    "network_tcp-flags-urg_count",
];

#[derive(Debug, PartialEq)]
pub struct Neighborhood {
    pub binary: Vec<Vec<f64>>,
    pub model_rows: Vec<Vec<Option<f32>>>,
    pub weights: Vec<f64>,
    pub conditions: Vec<String>,
    pub observed: Vec<usize>,
    pub constrained: bool,
}

pub fn generate(
    reference: &ReferenceData,
    row: &[Option<f32>],
    packet: bool,
    seed: u64,
    samples: usize,
) -> Result<Neighborhood> {
    let names = &reference.metadata.feature_names;
    if row.len() != names.len() || row.iter().flatten().any(|v| v.is_infinite()) {
        return Err(Error::Invalid(
            "LIME requires one ordered row of finite or missing inputs".into(),
        ));
    }
    if !(2..=8192).contains(&samples)
        || names.is_empty()
        || names.len() > 2048
        || reference.training.is_empty()
        || reference.training.len() > 8192
        || reference.training.iter().any(|r| r.len() != names.len())
        || reference.medians.len() != names.len()
        || reference.medians.iter().any(|v| !v.is_finite())
        || reference
            .metadata
            .categorical_indices
            .iter()
            .any(|i| *i >= names.len())
        || names.iter().collect::<HashSet<_>>().len() != names.len()
    {
        return Err(Error::Invalid(
            "Invalid LIME reference or sample count".into(),
        ));
    }
    let row: Vec<Option<f32>> = row.iter().map(|v| v.filter(|x| x.is_finite())).collect();
    let observed: Vec<usize> = row
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|_| i))
        .collect();
    if observed.is_empty() {
        return Err(Error::Invalid(
            "LIME requires at least one observed input".into(),
        ));
    }
    let count_indices: Vec<usize> = PACKET_NAMES
        .iter()
        .filter_map(|name| names.iter().position(|n| n == name))
        .collect();
    let canonical = observed.len() == 7
        && count_indices.len() == 7
        && count_indices.iter().all(|i| row[*i].is_some());
    let counts: Vec<f32> = if canonical {
        count_indices.iter().filter_map(|i| row[*i]).collect()
    } else {
        Vec::new()
    };
    let constrained = canonical
        && counts
            .iter()
            .all(|v| *v >= 0.0 && *v <= 16_777_216.0 && v.floor() == *v)
        && counts[0] >= 1.0
        && counts[1..].iter().all(|v| *v <= counts[0]);
    if packet && !constrained {
        return Err(Error::Invalid(
            "Packet LIME requires seven valid observed packet-count inputs".into(),
        ));
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0)
        .map_err(|error| Error::Invalid(format!("Normal distribution error: {error}")))?;
    let mut binary = vec![vec![1.0; observed.len()]; samples];
    let mut model_rows = vec![vec![None; names.len()]; samples];
    let mut conditions = Vec::with_capacity(observed.len());
    let mut columns = Vec::with_capacity(observed.len());
    for (local, original) in observed.iter().copied().enumerate() {
        let anchor = row[original]
            .ok_or_else(|| Error::Invalid("Observed index lost its value".into()))?
            as f64;
        let column: Vec<f64> = reference
            .training
            .iter()
            .map(|r| {
                r[original]
                    .filter(|v| v.is_finite())
                    .unwrap_or(reference.medians[original]) as f64
            })
            .collect();
        let categorical = reference.metadata.categorical_indices.contains(&original);
        let stats = Column::new(&column, categorical);
        let anchor_code = stats.code(anchor);
        conditions.push(stats.condition(&names[original], anchor));
        for sample in 0..samples {
            let code = stats.sample_code(&mut rng);
            if sample == 0 {
                model_rows[sample][original] = Some(anchor as f32);
            } else {
                binary[sample][local] = f64::from(code == anchor_code);
                model_rows[sample][original] = Some(code as f32);
            }
        }
        columns.push(stats);
    }
    // The package first samples all discrete codes, then undiscretizes each
    // continuous column. Keep that order rather than resampling rows.
    for (local, original) in observed.iter().copied().enumerate() {
        for sample in model_rows.iter_mut().skip(1) {
            let code = sample[original]
                .ok_or_else(|| Error::Invalid("Sampled code missing".into()))?
                as f64;
            sample[original] = Some(columns[local].inverse(code, &normal, &mut rng) as f32);
        }
    }
    if constrained {
        for sample in &mut model_rows {
            let total_index = count_indices[0];
            let total = sample[total_index]
                .ok_or_else(|| Error::Invalid("Packet total missing".into()))?
                .round_ties_even()
                .clamp(1.0, 16_777_216.0);
            sample[total_index] = Some(total);
            for index in count_indices.iter().skip(1) {
                let count = sample[*index]
                    .ok_or_else(|| Error::Invalid("Packet flag missing".into()))?
                    .round_ties_even()
                    .clamp(0.0, total);
                sample[*index] = Some(count);
            }
        }
    }
    let width_squared = observed.len() as f64 * 0.75_f64.powi(2);
    let weights = binary
        .iter()
        .map(|r| {
            let distance_squared = r.iter().map(|v| (v - 1.0).powi(2)).sum::<f64>();
            (-distance_squared / (2.0 * width_squared)).exp()
        })
        .collect();
    Ok(Neighborhood {
        binary,
        model_rows,
        weights,
        conditions,
        observed,
        constrained,
    })
}

pub fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let position = fraction.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

struct Bin {
    mean: f64,
    std: f64,
    minimum: f64,
    maximum: f64,
}
struct Column {
    categorical: bool,
    borders: Vec<f64>,
    codes: Vec<f64>,
    frequencies: Vec<usize>,
    bins: Vec<Bin>,
}
impl Column {
    fn new(values: &[f64], categorical: bool) -> Self {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let mut borders = if categorical {
            Vec::new()
        } else {
            vec![
                percentile(&sorted, 0.25),
                percentile(&sorted, 0.5),
                percentile(&sorted, 0.75),
            ]
        };
        borders.dedup();
        let mut codes = if categorical {
            sorted.clone()
        } else {
            (0..=borders.len()).map(|i| i as f64).collect()
        };
        codes.dedup();
        let mut frequencies = vec![0; codes.len()];
        let mut bins = Vec::new();
        for code in &codes {
            let selected: Vec<f64> = values
                .iter()
                .copied()
                .filter(|v| {
                    if categorical {
                        *v == *code
                    } else {
                        borders.partition_point(|b| *b < *v) as f64 == *code
                    }
                })
                .collect();
            let index = bins.len();
            frequencies[index] = selected.len();
            let mean = if selected.is_empty() {
                0.0
            } else {
                selected.iter().sum::<f64>() / selected.len() as f64
            };
            let std = if selected.is_empty() {
                0.0
            } else {
                (selected.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / selected.len() as f64)
                    .sqrt()
            } + 1e-11;
            let minimum = if categorical || index == 0 {
                sorted[0]
            } else {
                borders[index - 1]
            };
            let maximum = if categorical || index == borders.len() {
                sorted[sorted.len() - 1]
            } else {
                borders[index]
            };
            bins.push(Bin {
                mean,
                std,
                minimum,
                maximum,
            });
        }
        Self {
            categorical,
            borders,
            codes,
            frequencies,
            bins,
        }
    }
    fn code(&self, value: f64) -> f64 {
        if self.categorical {
            value
        } else {
            self.borders.partition_point(|border| *border < value) as f64
        }
    }
    fn sample_code(&self, rng: &mut ChaCha8Rng) -> f64 {
        let sample = rng.gen::<f64>() * self.frequencies.iter().sum::<usize>() as f64;
        let mut cumulative = 0.0;
        for (code, frequency) in self.codes.iter().zip(&self.frequencies) {
            cumulative += *frequency as f64;
            if sample < cumulative {
                return *code;
            }
        }
        self.codes[self.codes.len() - 1]
    }
    fn inverse(&self, code: f64, normal: &Normal, rng: &mut ChaCha8Rng) -> f64 {
        if self.categorical {
            return code;
        }
        let bin = &self.bins[code as usize];
        let low = (bin.minimum - bin.mean) / bin.std;
        let high = (bin.maximum - bin.mean) / bin.std;
        // Preserve lime 0.2.0.1's degenerate-bin behavior: it returns standardized
        // minz when minz==maxz, not the original constant numeric value.
        if low == high {
            return low;
        }
        let lower_cdf = normal.cdf(low);
        let upper_cdf = normal.cdf(high);
        let probability = (lower_cdf + rng.gen::<f64>() * (upper_cdf - lower_cdf))
            .clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
        (bin.mean + bin.std * normal.inverse_cdf(probability)).clamp(bin.minimum, bin.maximum)
    }
    fn condition(&self, name: &str, value: f64) -> String {
        if self.categorical {
            return format!("{name}={}", value.trunc() as i64);
        }
        let code = self.code(value) as usize;
        if code == 0 {
            format!("{name} <= {:.2}", self.borders[0])
        } else if code == self.borders.len() {
            format!("{name} > {:.2}", self.borders[code - 1])
        } else {
            format!(
                "{:.2} < {name} <= {:.2}",
                self.borders[code - 1],
                self.borders[code]
            )
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::reference::ReferenceMetadata;

    pub(crate) fn reference(names: Vec<String>, categorical: Vec<usize>) -> ReferenceData {
        let columns = names.len();
        ReferenceData {
            metadata: ReferenceMetadata {
                feature_names: names,
                categorical_indices: categorical,
                rows: 8,
                sha256: String::new(),
                kind: "synthetic test only".into(),
            },
            training: (0..8)
                .map(|i| {
                    (0..columns)
                        .map(|j| {
                            Some(if j == 0 {
                                (i * 10 + 1) as f32
                            } else {
                                (i % 2) as f32
                            })
                        })
                        .collect()
                })
                .collect(),
            medians: vec![0.0; columns],
        }
    }

    // Wrong percentile interpolation changes bin membership at exact boundaries.
    #[test]
    fn quartiles_use_linear_interpolation() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.25), 1.75);
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.50), 2.5);
    }

    // A categorical imputation or all-column distance would invent measured evidence.
    #[test]
    fn missing_categorical_column_never_enters_neighborhood_or_distances() {
        let r = reference(vec!["x".into(), "protocol".into()], vec![1]);
        let result = generate(&r, &[Some(20.0), None], false, 42, 256).unwrap();
        assert_eq!(result.observed, vec![0]);
        assert_eq!(result.binary[0], vec![1.0]);
        assert_eq!(result.model_rows[0], vec![Some(20.0), None]);
        assert!(result.model_rows.iter().all(|row| row[1].is_none()));
        assert_eq!(result.weights[0], 1.0);
        let mismatch = result.binary.iter().position(|row| row[0] == 0.0).unwrap();
        assert!(
            (result.weights[mismatch] - (-1.0f64 / (2.0 * 0.75f64.powi(2))).exp()).abs() < 1e-12
        );
    }

    // Constraints must be mapped by canonical names, never fixed feature positions.
    #[test]
    fn packet_perturbations_are_integer_bounded_and_flags_do_not_exceed_total() {
        let mut names: Vec<String> = PACKET_NAMES.iter().rev().map(|s| s.to_string()).collect();
        names.push("unobserved".into());
        let r = reference(names, vec![]);
        let result = generate(
            &r,
            &[
                Some(0.0),
                Some(10.0),
                Some(1.0),
                Some(2.0),
                Some(1.0),
                Some(30.0),
                Some(100.0),
                None,
            ],
            true,
            42,
            256,
        )
        .unwrap();
        assert!(result.constrained);
        assert!(result.model_rows.iter().all(|row| {
            let total = row[6].unwrap();
            (1.0..=16_777_216.0).contains(&total)
                && row[..6]
                    .iter()
                    .all(|value| value.is_some_and(|v| v >= 0.0 && v <= total && v == v.floor()))
                && row[7].is_none()
        }));
    }

    #[test]
    fn invalid_explicit_packet_profile_fails_and_empty_observation_fails() {
        let r = reference(PACKET_NAMES.iter().map(|s| s.to_string()).collect(), vec![]);
        let bad = generate(&r, &[Some(0.0); 7], true, 42, 128).unwrap_err();
        assert!(bad.to_string().contains("valid observed packet-count"));
        let missing = generate(&r, &[None; 7], false, 42, 128).unwrap_err();
        assert!(missing.to_string().contains("observed input"));
    }

    #[test]
    fn same_seed_has_identical_neighborhood_and_different_seed_changes_sampling() {
        let r = reference(vec!["x".into(), "protocol".into()], vec![1]);
        let row = [Some(20.0), Some(0.0)];
        let first = generate(&r, &row, false, 42, 128).unwrap();
        assert_eq!(first, generate(&r, &row, false, 42, 128).unwrap());
        assert_ne!(
            first.model_rows,
            generate(&r, &row, false, 43, 128).unwrap().model_rows
        );
    }

    // searchsorted-left assigns an exact quartile edge to the lower bin.
    #[test]
    fn exact_quartile_edge_has_lower_bin_condition() {
        let r = reference(vec!["x".into()], vec![]);
        let result = generate(&r, &[Some(18.5)], false, 42, 128).unwrap();
        assert_eq!(result.conditions, vec!["x <= 18.50"]);
    }

    // Preserve the installed lime package's documented degenerate-bin quirk;
    // silently returning the original constant would change callback scores.
    #[test]
    fn constant_numeric_bin_preserves_package_standardized_bound_behavior() {
        let mut r = reference(vec!["constant".into()], vec![]);
        r.training = vec![vec![Some(7.0)]; 8];
        r.medians = vec![7.0];
        let result = generate(&r, &[Some(7.0)], false, 42, 128).unwrap();
        assert_eq!(result.model_rows[0], vec![Some(7.0)]);
        assert!(result.model_rows[1..].iter().all(|row| row == &[Some(0.0)]));
        assert!(result.binary.iter().all(|row| row == &[1.0]));
    }
}
