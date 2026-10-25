use crate::{Error, Result};

#[derive(Debug, serde::Serialize)]
pub struct Fit {
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    pub r2: f64,
}

pub fn select_features(
    data: &[Vec<f64>],
    labels: &[f64],
    weights: &[f64],
    maximum: usize,
) -> Result<Vec<usize>> {
    let columns = data.first().map_or(0, Vec::len);
    let maximum = maximum.min(columns);
    if maximum == 0 {
        return Err(Error::Invalid("LIME requires selected features".into()));
    }
    if maximum > 6 {
        let fit = ridge(data, labels, weights, 0.01)?;
        let mut ranking: Vec<(usize, f64)> = fit
            .coefficients
            .iter()
            .enumerate()
            .map(|(i, coef)| (i, coef * data[0][i]))
            .collect();
        ranking.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        return Ok(ranking.iter().take(maximum).map(|entry| entry.0).collect());
    }
    let mut used = Vec::new();
    for _ in 0..maximum {
        let mut best = None;
        let mut best_score = f64::NEG_INFINITY;
        for candidate in 0..columns {
            if used.contains(&candidate) {
                continue;
            }
            let mut selected = used.clone();
            selected.push(candidate);
            let subset: Vec<Vec<f64>> = data
                .iter()
                .map(|row| selected.iter().map(|i| row[*i]).collect())
                .collect();
            let fit = ridge(&subset, labels, weights, 0.0)?;
            if fit.r2 > best_score {
                best = Some(candidate);
                best_score = fit.r2;
            }
        }
        used.push(best.ok_or_else(|| Error::Invalid("Forward selection found no feature".into()))?);
    }
    Ok(used)
}

pub fn ridge(data: &[Vec<f64>], labels: &[f64], weights: &[f64], alpha: f64) -> Result<Fit> {
    let columns = data.first().map_or(0, Vec::len);
    if data.is_empty()
        || labels.len() != data.len()
        || weights.len() != data.len()
        || data
            .iter()
            .any(|row| row.len() != columns || row.iter().any(|x| !x.is_finite()))
        || labels.iter().any(|x| !x.is_finite())
        || weights.iter().any(|w| !w.is_finite() || *w <= 0.0)
        || !alpha.is_finite()
        || alpha < 0.0
    {
        return Err(Error::Invalid("Invalid weighted Ridge data".into()));
    }
    // Exact constant targets merit exact zero coefficients. The Python solver's
    // floating centering noise may instead yield 1e-31 weights and R2=0 here.
    if labels.iter().all(|label| *label == labels[0]) {
        return Ok(Fit {
            intercept: labels[0],
            coefficients: vec![0.0; columns],
            r2: 1.0,
        });
    }
    let total: f64 = weights.iter().sum();
    let mean_y = labels.iter().zip(weights).map(|(y, w)| y * w).sum::<f64>() / total;
    let mut means = vec![0.0; columns];
    for (row, weight) in data.iter().zip(weights) {
        for (mean, value) in means.iter_mut().zip(row) {
            *mean += weight * value / total;
        }
    }
    let mut covariance = vec![vec![0.0; columns]; columns];
    let mut rhs = vec![0.0; columns];
    for ((row, y), weight) in data.iter().zip(labels).zip(weights) {
        for i in 0..columns {
            let x = row[i] - means[i];
            rhs[i] += weight * x * (y - mean_y);
            for j in 0..=i {
                covariance[i][j] += weight * x * (row[j] - means[j]);
            }
        }
    }
    for i in 0..columns {
        let (before, current) = covariance.split_at_mut(i);
        for (j, row) in before.iter_mut().enumerate() {
            row[i] = current[0][j];
        }
        current[0][i] += alpha;
    }
    let coefficients = if alpha == 0.0 {
        pseudoinverse(&covariance, &rhs)
    } else {
        cholesky(&covariance, &rhs)?
    };
    let intercept = mean_y
        - coefficients
            .iter()
            .zip(&means)
            .map(|(b, x)| b * x)
            .sum::<f64>();
    let residual = data
        .iter()
        .zip(labels)
        .zip(weights)
        .map(|((row, y), w)| {
            let pred = intercept
                + row
                    .iter()
                    .zip(&coefficients)
                    .map(|(x, b)| x * b)
                    .sum::<f64>();
            w * (y - pred).powi(2)
        })
        .sum::<f64>();
    let variance = labels
        .iter()
        .zip(weights)
        .map(|(y, w)| w * (y - mean_y).powi(2))
        .sum::<f64>();
    let r2 = if variance == 0.0 {
        f64::from(residual == 0.0)
    } else {
        1.0 - residual / variance
    };
    Ok(Fit {
        intercept,
        coefficients,
        r2,
    })
}

fn cholesky(matrix: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>> {
    let n = rhs.len();
    let mut lower = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let product = (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
            let value = matrix[i][j] - product;
            if i == j {
                if value <= 0.0 {
                    return Err(Error::Invalid(
                        "Ridge normal matrix is not positive definite".into(),
                    ));
                }
                lower[i][j] = value.sqrt();
            } else {
                lower[i][j] = value / lower[j][j];
            }
        }
    }
    let mut solution = vec![0.0; n];
    for i in 0..n {
        solution[i] =
            (rhs[i] - (0..i).map(|j| lower[i][j] * solution[j]).sum::<f64>()) / lower[i][i];
    }
    for i in (0..n).rev() {
        solution[i] = (solution[i] - (i + 1..n).map(|j| lower[j][i] * solution[j]).sum::<f64>())
            / lower[i][i];
    }
    Ok(solution)
}

// Jacobi spectral pseudo-inverse handles constant/duplicate indicator columns
// in alpha=0 forward selection; only <=6 columns reach this branch.
fn pseudoinverse(matrix: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let mut diagonalized = matrix.to_vec();
    let mut vectors = vec![vec![0.0; n]; n];
    for (i, row) in vectors.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let scale = (0..n).map(|i| matrix[i][i].abs()).fold(0.0, f64::max);
    for _ in 0..100 * n * n {
        let mut pair = None;
        let mut largest = 0.0;
        for (i, row) in diagonalized.iter().enumerate() {
            for (j, value) in row.iter().enumerate().skip(i + 1) {
                if value.abs() > largest {
                    largest = value.abs();
                    pair = Some((i, j));
                }
            }
        }
        if largest <= 1e-14 * scale {
            break;
        }
        let Some((p, q)) = pair else {
            break;
        };
        let angle = 0.5 * (2.0 * diagonalized[p][q]).atan2(diagonalized[q][q] - diagonalized[p][p]);
        let (s, c) = angle.sin_cos();
        let (pp, qq, pq) = (diagonalized[p][p], diagonalized[q][q], diagonalized[p][q]);
        for k in 0..n {
            if k != p && k != q {
                let (kp, kq) = (diagonalized[k][p], diagonalized[k][q]);
                diagonalized[k][p] = c * kp - s * kq;
                diagonalized[p][k] = diagonalized[k][p];
                diagonalized[k][q] = s * kp + c * kq;
                diagonalized[q][k] = diagonalized[k][q];
            }
            let (vp, vq) = (vectors[k][p], vectors[k][q]);
            vectors[k][p] = c * vp - s * vq;
            vectors[k][q] = s * vp + c * vq;
        }
        diagonalized[p][p] = c * c * pp - 2.0 * s * c * pq + s * s * qq;
        diagonalized[q][q] = s * s * pp + 2.0 * s * c * pq + c * c * qq;
        diagonalized[p][q] = 0.0;
        diagonalized[q][p] = 0.0;
    }
    let mut solution = vec![0.0; n];
    for k in 0..n {
        if diagonalized[k][k] <= scale * 1e-12 {
            continue;
        }
        let projected = vectors
            .iter()
            .zip(rhs)
            .map(|(row, value)| row[k] * value)
            .sum::<f64>()
            / diagonalized[k][k];
        for i in 0..n {
            solution[i] += vectors[i][k] * projected;
        }
    }
    solution
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // Removing weighted centering or penalizing the intercept breaks this result.
    #[test]
    fn weighted_ridge_uses_unpenalized_intercept() {
        let fit = ridge(
            &[vec![0.0], vec![1.0], vec![2.0]],
            &[1.0, 3.0, 5.0],
            &[1.0, 2.0, 1.0],
            1.0,
        )
        .unwrap();
        assert!((fit.coefficients[0] - 4.0 / 3.0).abs() < 1e-12);
        assert!((fit.intercept - 5.0 / 3.0).abs() < 1e-12);
        assert!((fit.r2 - 8.0 / 9.0).abs() < 1e-12);
    }

    // Degenerate categorical indicator columns cannot be allowed to crash selection.
    #[test]
    fn constant_columns_and_labels_have_zero_weight_and_perfect_fidelity() {
        let fit = ridge(&[vec![1.0], vec![1.0]], &[0.75, 0.75], &[1.0, 0.5], 0.0).unwrap();
        assert_eq!(fit.coefficients, vec![0.0]);
        assert_eq!(fit.intercept, 0.75);
        assert_eq!(fit.r2, 1.0);
    }

    // Selecting low-index features without fitting would miss the predictive column.
    #[test]
    fn auto_forward_selection_finds_predictive_feature_first() {
        let data: Vec<Vec<f64>> = (0..8)
            .map(|i| vec![(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64])
            .collect();
        let labels: Vec<f64> = data.iter().map(|row| 0.2 + 0.6 * row[2]).collect();
        assert_eq!(
            select_features(&data, &labels, &[1.0; 8], 1).unwrap(),
            vec![2]
        );
    }

    #[test]
    fn unregularized_ridge_splits_duplicate_column_weight_equally() {
        let fit = ridge(
            &[vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]],
            &[0.0, 2.0, 4.0],
            &[1.0; 3],
            0.0,
        )
        .unwrap();
        assert!((fit.coefficients[0] - 1.0).abs() < 1e-12);
        assert!((fit.coefficients[1] - 1.0).abs() < 1e-12);
    }
}
