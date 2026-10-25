//! Isolated pure Rust XGBoost 3.1.3 numeric gbtree / multi:softprob spike.
//! TreeSHAP is adapted from XGBoost v3.1.3 src/predictor/treeshap.cc (Apache 2.0),
//! Copyright 2017-2025 XGBoost Contributors. See NOTICE and LICENSE.
#![forbid(unsafe_code)]
mod treeshap;
mod ubjson;

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("Checkpoint I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("Checkpoint JSON: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug)]
struct Node {
    left: Option<usize>,
    right: Option<usize>,
    feature: usize,
    value: f32,
    cover: f32,
    default_left: bool,
}
#[derive(Debug)]
struct Tree {
    nodes: Vec<Node>,
    group: usize,
    mean: f32,
}

#[derive(Debug)]
pub struct Booster {
    trees: Vec<Tree>,
    base: Vec<f32>,
    features: usize,
}

#[derive(Debug, Serialize)]
pub struct Evaluation {
    pub probabilities: Vec<f32>,
    pub margins: Vec<f32>,
    pub contributions: Vec<Vec<f32>>,
    pub argmax: usize,
    pub top_three: Vec<usize>,
    pub max_reconstruction_error: f64,
}

fn field<'a>(value: &'a Value, name: &str) -> Result<&'a Value, Error> {
    value
        .get(name)
        .ok_or_else(|| Error::Invalid(format!("Missing checkpoint field {name}")))
}
fn text<'a>(value: &'a Value, name: &str) -> Result<&'a str, Error> {
    field(value, name)?
        .as_str()
        .ok_or_else(|| Error::Invalid(format!("{name} must be text")))
}
fn integer(value: &Value, name: &str) -> Result<usize, Error> {
    text(value, name)?
        .parse()
        .map_err(|_| Error::Invalid(format!("{name} must be a nonnegative integer")))
}
fn array<'a>(value: &'a Value, name: &str) -> Result<&'a Vec<Value>, Error> {
    field(value, name)?
        .as_array()
        .ok_or_else(|| Error::Invalid(format!("{name} must be an array")))
}
fn number(value: &Value) -> Result<f32, Error> {
    let result = value
        .as_f64()
        .ok_or_else(|| Error::Invalid("Expected numeric checkpoint value".into()))?
        as f32;
    if !result.is_finite() {
        return Err(Error::Invalid("Nonfinite checkpoint value".into()));
    }
    Ok(result)
}
fn child(value: &Value, nodes: usize) -> Result<Option<usize>, Error> {
    let index = value
        .as_i64()
        .ok_or_else(|| Error::Invalid("Child index must be integer".into()))?;
    if index == -1 {
        return Ok(None);
    }
    let index = usize::try_from(index).map_err(|_| Error::Invalid("Invalid child index".into()))?;
    if index >= nodes {
        return Err(Error::Invalid("Child index outside tree".into()));
    }
    Ok(Some(index))
}

impl Booster {
    pub fn load_checkpoint(path: &Path, expected_sha256: Option<&str>) -> Result<Self, Error> {
        if fs::metadata(path)?.len() > 256 * 1024 * 1024 {
            return Err(Error::Invalid(
                "Checkpoint exceeds 256 MiB spike bound".into(),
            ));
        }
        let bytes = fs::read(path)?;
        if let Some(expected) = expected_sha256 {
            if format!("{:x}", Sha256::digest(&bytes)) != expected {
                return Err(Error::Invalid("Checkpoint checksum mismatch".into()));
            }
        }
        let document = match path.extension().and_then(|s| s.to_str()) {
            Some("ubj") => ubjson::decode(&bytes)?,
            Some("json") => serde_json::from_slice(&bytes)?,
            _ => {
                return Err(Error::Invalid(
                    "Only native UBJ/JSON checkpoint extensions are supported".into(),
                ))
            }
        };
        Self::from_document(&document)
    }

    fn from_document(document: &Value) -> Result<Self, Error> {
        let learner = field(document, "learner")?;
        if text(field(learner, "objective")?, "name")? != "multi:softprob" {
            return Err(Error::Invalid("Spike supports only multi:softprob".into()));
        }
        let booster = field(learner, "gradient_booster")?;
        if text(booster, "name")? != "gbtree" {
            return Err(Error::Invalid(
                "Spike supports only gbtree, not DART".into(),
            ));
        }
        let parameters = field(learner, "learner_model_param")?;
        let features = integer(parameters, "num_feature")?;
        let classes = integer(parameters, "num_class")?;
        if features == 0
            || features > 4096
            || !(2..=512).contains(&classes)
            || integer(parameters, "num_target")? != 1
        {
            return Err(Error::Invalid(
                "Unsupported feature/class/target dimensions".into(),
            ));
        }
        let base_string = text(parameters, "base_score")?;
        let base = if base_string.starts_with('[') {
            let values: Vec<f32> = serde_json::from_str(base_string)?;
            if values.len() != classes {
                return Err(Error::Invalid(
                    "Vector base score differs from class count".into(),
                ));
            }
            values
        } else {
            vec![
                base_string
                    .parse::<f32>()
                    .map_err(|_| Error::Invalid("Invalid scalar base score".into()))?;
                classes
            ]
        };
        if base.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("Nonfinite base score".into()));
        }
        let model = field(booster, "model")?;
        let raw_trees = array(model, "trees")?;
        let groups = array(model, "tree_info")?;
        if raw_trees.len() != groups.len() || raw_trees.len() > 100_000 {
            return Err(Error::Invalid("Invalid tree/class mapping".into()));
        }
        let mut trees = Vec::with_capacity(raw_trees.len());
        for (raw, group) in raw_trees.iter().zip(groups) {
            let group = usize::try_from(
                group
                    .as_u64()
                    .ok_or_else(|| Error::Invalid("Invalid tree class".into()))?,
            )
            .map_err(|_| Error::Invalid("Tree class overflow".into()))?;
            if group >= classes {
                return Err(Error::Invalid("Tree class outside model".into()));
            }
            let params = field(raw, "tree_param")?;
            let n = integer(params, "num_nodes")?;
            if n == 0
                || n > 1_000_000
                || integer(params, "num_deleted")? != 0
                || integer(params, "size_leaf_vector")? != 1
            {
                return Err(Error::Invalid(
                    "Unsupported deleted/vector-leaf tree".into(),
                ));
            }
            let columns = [
                "left_children",
                "right_children",
                "split_indices",
                "split_conditions",
                "sum_hessian",
                "default_left",
                "split_type",
            ];
            let arrays: Vec<_> = columns
                .iter()
                .map(|name| array(raw, name))
                .collect::<Result<_, _>>()?;
            if arrays.iter().any(|a| a.len() != n) {
                return Err(Error::Invalid("Tree arrays differ from node count".into()));
            }
            let mut nodes = Vec::with_capacity(n);
            for (i, raw_left) in arrays[0].iter().enumerate() {
                let left = child(raw_left, n)?;
                let right = child(&arrays[1][i], n)?;
                let feature = usize::try_from(
                    arrays[2][i]
                        .as_u64()
                        .ok_or_else(|| Error::Invalid("Invalid split index".into()))?,
                )
                .map_err(|_| Error::Invalid("Split index overflow".into()))?;
                let cover = number(&arrays[4][i])?;
                let default_left = match arrays[5][i].as_u64() {
                    Some(0) => false,
                    Some(1) => true,
                    _ => return Err(Error::Invalid("Invalid default branch".into())),
                };
                if arrays[6][i].as_u64() != Some(0) {
                    return Err(Error::Invalid(
                        "Categorical split unsupported by numeric spike".into(),
                    ));
                }
                if left.is_some() != right.is_some()
                    || (left.is_some() && (feature >= features || cover <= 0.0))
                    || cover < 0.0
                {
                    return Err(Error::Invalid(
                        "Invalid split children/feature/cover".into(),
                    ));
                }
                nodes.push(Node {
                    left,
                    right,
                    feature,
                    value: number(&arrays[3][i])?,
                    cover,
                    default_left,
                });
            }
            let mut visiting = vec![0u8; n];
            let mean = mean_value(&nodes, 0, 0, &mut visiting)?;
            if visiting.iter().any(|&v| v != 2) {
                return Err(Error::Invalid("Unreachable tree nodes".into()));
            }
            trees.push(Tree { nodes, group, mean });
        }
        Ok(Self {
            trees,
            base,
            features,
        })
    }

    pub fn num_features(&self) -> usize {
        self.features
    }
    pub fn num_classes(&self) -> usize {
        self.base.len()
    }
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }
    fn validate(&self, row: &[Option<f32>]) -> Result<(), Error> {
        if row.len() != self.features || row.iter().flatten().any(|v| !v.is_finite()) {
            return Err(Error::Invalid(
                "Expected ordered finite-or-null feature row".into(),
            ));
        }
        Ok(())
    }
    pub fn margins(&self, row: &[Option<f32>]) -> Result<Vec<f32>, Error> {
        self.validate(row)?;
        let mut values = self.base.clone();
        for tree in &self.trees {
            let mut index = 0;
            loop {
                let node = &tree.nodes[index];
                match (node.left, node.right) {
                    (Some(left), Some(right)) => {
                        index = if row[node.feature].map_or(node.default_left, |v| v < node.value) {
                            left
                        } else {
                            right
                        }
                    }
                    _ => {
                        values[tree.group] += node.value;
                        break;
                    }
                }
            }
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("Model margin overflow".into()));
        }
        Ok(values)
    }
    pub fn predict(&self, row: &[Option<f32>]) -> Result<Vec<f32>, Error> {
        let margins = self.margins(row)?;
        let maximum = margins.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut scores: Vec<_> = margins.iter().map(|v| (v - maximum).exp()).collect();
        // XGBoost accumulates the softmax normalizer as double precision.
        let total = scores.iter().map(|&v| v as f64).sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return Err(Error::Invalid("Invalid softmax normalization".into()));
        }
        for score in &mut scores {
            *score = (*score as f64 / total) as f32;
        }
        Ok(scores)
    }
    pub fn contributions(&self, row: &[Option<f32>]) -> Result<Vec<Vec<f32>>, Error> {
        self.validate(row)?;
        let mut result = vec![vec![0.0; self.features + 1]; self.num_classes()];
        for tree in &self.trees {
            let values = treeshap::explain(tree, row, self.features)?;
            for (target, value) in result[tree.group].iter_mut().zip(values) {
                *target += value;
            }
        }
        for (values, base) in result.iter_mut().zip(&self.base) {
            values[self.features] += base;
        }
        Ok(result)
    }
    pub fn evaluate(&self, row: &[Option<f32>]) -> Result<Evaluation, Error> {
        let margins = self.margins(row)?;
        let probabilities = self.predict(row)?;
        let contributions = self.contributions(row)?;
        let max_reconstruction_error = contributions
            .iter()
            .zip(&margins)
            .map(|(values, &margin)| {
                (values.iter().map(|&v| v as f64).sum::<f64>() - margin as f64).abs()
            })
            .fold(0.0, f64::max);
        if max_reconstruction_error
            > 1e-4 + 1e-5 * margins.iter().map(|v| v.abs() as f64).fold(0.0, f64::max)
        {
            return Err(Error::Invalid(
                "SHAP failed to reconstruct margins; refusing explanation".into(),
            ));
        }
        let mut order: Vec<usize> = (0..probabilities.len()).collect();
        order.sort_by(|&a, &b| {
            probabilities[b]
                .total_cmp(&probabilities[a])
                .then(a.cmp(&b))
        });
        let argmax = *order
            .first()
            .ok_or_else(|| Error::Invalid("No class probabilities".into()))?;
        order.truncate(3);
        Ok(Evaluation {
            probabilities,
            margins,
            contributions,
            argmax,
            top_three: order,
            max_reconstruction_error,
        })
    }
}

fn mean_value(
    nodes: &[Node],
    index: usize,
    depth: usize,
    visiting: &mut [u8],
) -> Result<f32, Error> {
    if depth > 64 || visiting[index] != 0 {
        return Err(Error::Invalid(
            "Tree cyclic, repeated node or depth exceeds 64".into(),
        ));
    }
    visiting[index] = 1;
    let node = &nodes[index];
    let result = match (node.left, node.right) {
        (Some(left), Some(right)) => {
            let l = mean_value(nodes, left, depth + 1, visiting)? * nodes[left].cover;
            let r = mean_value(nodes, right, depth + 1, visiting)? * nodes[right].cover;
            (l + r) / node.cover
        }
        _ => node.value,
    };
    visiting[index] = 2;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn corpus() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../evaluation/fixtures/rust_migration/v1/models")
    }
    #[test]
    fn native_ubj_matches_python_all_class_goldens() {
        let directory = corpus();
        let booster = Booster::load_checkpoint(&directory.join("model.ubj"), None).unwrap();
        let golden: Value =
            serde_json::from_slice(&fs::read(directory.join("predictions.json")).unwrap()).unwrap();
        for case in golden["cases"].as_array().unwrap() {
            let row: Vec<Option<f32>> = serde_json::from_value(case["input"].clone()).unwrap();
            let result = booster.evaluate(&row).unwrap();
            for (&value, expected) in result
                .probabilities
                .iter()
                .zip(case["probabilities"].as_array().unwrap())
            {
                assert!((value as f64 - expected.as_f64().unwrap()).abs() < 1e-5);
            }
            for (&value, expected) in result
                .margins
                .iter()
                .zip(case["margins"].as_array().unwrap())
            {
                assert!((value as f64 - expected.as_f64().unwrap()).abs() < 1e-5);
            }
            for (values, expected) in result
                .contributions
                .iter()
                .zip(case["contributions"].as_array().unwrap())
            {
                for (&value, expected) in values.iter().zip(expected.as_array().unwrap()) {
                    assert!(
                        (value as f64 - expected.as_f64().unwrap()).abs() < 1e-4,
                        "{} SHAP difference",
                        case["id"]
                    );
                }
            }
        }
    }
    #[test]
    fn checksum_shape_and_stable_ties() {
        let directory = corpus();
        assert!(Booster::load_checkpoint(&directory.join("model.ubj"), Some("00")).is_err());
        let booster = Booster::load_checkpoint(&directory.join("ties.ubj"), None).unwrap();
        assert!(booster.predict(&[]).is_err());
        assert!(booster
            .predict(&vec![Some(f32::INFINITY); booster.num_features()])
            .is_err());
        let result = booster
            .evaluate(&vec![Some(0.0); booster.num_features()])
            .unwrap();
        assert_eq!(result.argmax, 0);
        assert_eq!(result.top_three, vec![0, 1, 2]);
    }

    #[test]
    fn corrupt_tree_and_unsupported_split_are_explicit_errors() {
        let bytes = fs::read(corpus().join("model.ubj")).unwrap();
        let mut cyclic = ubjson::decode(&bytes).unwrap();
        cyclic["learner"]["gradient_booster"]["model"]["trees"][0]["left_children"][0] =
            serde_json::json!(0);
        assert!(Booster::from_document(&cyclic).is_err());
        let mut categorical = ubjson::decode(&bytes).unwrap();
        categorical["learner"]["gradient_booster"]["model"]["trees"][0]["split_type"][0] =
            serde_json::json!(1);
        assert!(Booster::from_document(&categorical).is_err());
        let overflowing = Booster {
            features: 1,
            base: vec![f32::MAX, 0.0],
            trees: vec![Tree {
                group: 0,
                mean: f32::MAX,
                nodes: vec![Node {
                    left: None,
                    right: None,
                    feature: 0,
                    value: f32::MAX,
                    cover: 1.0,
                    default_left: false,
                }],
            }],
        };
        assert!(overflowing.predict(&[Some(0.0)]).is_err());
    }
}
