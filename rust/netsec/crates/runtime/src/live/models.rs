//! The two legacy live-compatible checkpoints stay distinct from 655-feature CSV models.
use super::packet::{Window, COUNT_NAMES};
use netsec_xgboost_spike::Booster;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

const PARTIAL_BUNDLE_SHA: &str = "24a6486f15e6413a8f7eb9e43620414563aa56b1bc64038f012211aa41169e47";
const BINARY_BUNDLE_SHA: &str = "41381e439ee32fe0d065bbbf99a47270633bdbaf004c426c7dcf0de656eaaf38";

struct Model {
    booster: Booster,
    names: Vec<String>,
    classes: Vec<String>,
    sha: String,
    bundle_sha: String,
    version: String,
    binary: bool,
}

fn load(root: &Path, binary: bool) -> Result<Model, String> {
    let folder = if binary {
        "packet_live"
    } else {
        "xgboost_partial"
    };
    let bundle = root
        .join("framework/models")
        .join(folder)
        .join("bundle.json");
    if fs::metadata(&bundle)
        .map_err(|e| format!("Live {folder} bundle unavailable: {e}"))?
        .len()
        > 1024 * 1024
    {
        return Err("Live bundle exceeds its size bound".into());
    }
    let bytes = fs::read(&bundle).map_err(|e| e.to_string())?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    let expected = if binary {
        BINARY_BUNDLE_SHA
    } else {
        PARTIAL_BUNDLE_SHA
    };
    if hash != expected {
        return Err(format!(
            "Live {folder} bundle SHA256 differs from the explicitly approved version"
        ));
    }
    let manifest: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let names: Vec<String> =
        serde_json::from_value(manifest["feature_names"].clone()).map_err(|e| e.to_string())?;
    let classes: Vec<String> =
        serde_json::from_value(manifest["classes"].clone()).map_err(|e| e.to_string())?;
    if manifest["test_evaluated"] != false || manifest["live_validation"] != false {
        return Err("Unexpected live-model evaluation provenance".into());
    }
    if binary {
        if names != COUNT_NAMES
            || classes != ["normal", "suspicious"]
            || manifest["window_seconds"] != 10
        {
            return Err("Binary live count schema differs".into());
        }
    } else if names.len() != 170
        || classes.len() != 84
        || manifest["partial_input_supported"] != true
        || manifest["benign_index"] != 0
    {
        return Err("Partial live model schema differs".into());
    }
    let relative = manifest["checkpoint"]
        .as_str()
        .ok_or("Live checkpoint missing")?;
    let checkpoint = bundle
        .parent()
        .ok_or("Live bundle folder missing")?
        .join(relative)
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let canonical_root = root.canonicalize().map_err(|e| e.to_string())?;
    if !checkpoint.starts_with(&canonical_root) {
        return Err("Live checkpoint must stay in the packaged application root".into());
    }
    let sha = manifest["checkpoint_sha256"]
        .as_str()
        .ok_or("Live checkpoint SHA256 missing")?
        .to_owned();
    let booster = Booster::load_checkpoint(&checkpoint, Some(&sha)).map_err(|e| e.to_string())?;
    if booster.num_features() != names.len() || booster.num_classes() != classes.len() {
        return Err("Live checkpoint dimensions differ".into());
    }
    Ok(Model {
        booster,
        names,
        classes,
        sha,
        bundle_sha: hash,
        version: manifest["model"]["version"]
            .as_str()
            .or_else(|| manifest["run_id"].as_str())
            .unwrap_or(folder)
            .to_owned(),
        binary,
    })
}

impl Model {
    fn predict(&self, counts: &[f32; 7], observation_id: &str) -> Result<Value, String> {
        let mut row = vec![None; self.names.len()];
        for (name, count) in COUNT_NAMES.iter().zip(counts) {
            let index = self
                .names
                .iter()
                .position(|v| v == name)
                .ok_or("Measured live count absent from checkpoint schema")?;
            row[index] = Some(*count);
        }
        let probabilities = self.booster.predict(&row).map_err(|e| e.to_string())?;
        let mut order = (0..probabilities.len()).collect::<Vec<_>>();
        order.sort_by(|a, b| {
            probabilities[*b]
                .total_cmp(&probabilities[*a])
                .then(a.cmp(b))
        });
        let target = *order
            .first()
            .ok_or("Live model returned no probabilities")?;
        let mut result = json!({"available":true,"model_executed":true,"model_id":if self.binary{"live_binary_xgboost"}else{"live_partial_xgboost"},
            "label":self.classes[target],"is_attack":target!=0,"score":probabilities[target],"score_kind":"uncalibrated_probability",
            "top_classes":order.iter().take(3).map(|i|json!({"label":self.classes[*i],"score":probabilities[*i]})).collect::<Vec<_>>(),
            "observation_id":observation_id,"checkpoint_sha256":self.sha,"bundle_sha256":self.bundle_sha,"model_version":self.version,
            "feature_coverage":{"observed":7,"missing":self.names.len()-7,"total":self.names.len(),"profile":"seven_measured_packet_header_counts"},
            "features":row,"feature_names":self.names,"novelty":"not evaluated","live_validation":false,"test_evaluated":false,
            "limitations":["Experimental DataSense model; not labelled Windows-PC accuracy or incident confirmation",
                "Unknown attacks and encrypted application contents are not evaluated","Uncalibrated scores are not proof of safety or an attack"]});
        if self.binary {
            result["attack_score"] = json!(probabilities[1]);
        }
        match (self.booster.margins(&row), self.booster.contributions(&row)) {
            (Ok(margins), Ok(all)) => {
                let errors = all
                    .iter()
                    .zip(&margins)
                    .map(|(values, margin)| {
                        (values.iter().map(|v| *v as f64).sum::<f64>() - *margin as f64).abs()
                    })
                    .collect::<Vec<_>>();
                let max_error = errors.iter().copied().fold(0.0, f64::max);
                if max_error <= 1e-4 {
                    let contributions = &all[target];
                    let bias = *contributions.last().ok_or("Empty live SHAP output")?;
                    let mut features=contributions[..self.names.len()].iter().enumerate().map(|(i,v)|json!({"feature":self.names[i],"value":row[i],"contribution":v,"is_missing":row[i].is_none()})).collect::<Vec<_>>();
                    features.sort_by(|a, b| {
                        b["contribution"]
                            .as_f64()
                            .unwrap_or(0.)
                            .abs()
                            .total_cmp(&a["contribution"].as_f64().unwrap_or(0.).abs())
                    });
                    result["explanation"] = json!({"method":"exact_tree_shap","scope":"local","target_label":self.classes[target],"units":"raw class margin",
                        "bias":bias,"margin":margins[target],"features":features,"quality":{"all_class_max_reconstruction_error":max_error},
                        "warnings":["Missing-input routing is model behavior, not measured traffic","TreeSHAP is not causal evidence or probability percentage points"]});
                } else {
                    result["explanation_error"] = json!(format!(
                        "TreeSHAP reconstruction error {max_error} exceeded 1e-4"
                    ));
                }
            }
            (Err(error), _) | (_, Err(error)) => {
                result["explanation_error"] = json!(error.to_string())
            }
        }
        Ok(result)
    }
}

pub struct Models {
    partial: Result<Model, String>,
    binary: Result<Model, String>,
}
impl Models {
    pub fn load(root: &Path) -> Self {
        Self {
            partial: load(root, false),
            binary: load(root, true),
        }
    }
    pub fn predict(&self, window: &Window, seconds: f64, id: &str) -> (Value, Value) {
        let counts = window.counts(seconds);
        let predict = |model: &Result<Model, String>| match (&counts, model) {
            (Ok(counts), Ok(model)) => model.predict(counts, id).unwrap_or_else(unavailable),
            (Err(error), _) | (_, Err(error)) => unavailable(error.clone()),
        };
        (predict(&self.partial), predict(&self.binary))
    }
}
fn unavailable(error: String) -> Value {
    json!({"available":false,"model_executed":false,"error":error,"limitations":"This model capability failed or this window was incompatible. Observation and independent rules continue; no substitute prediction was returned."})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_models_are_scoped_and_never_replaced() {
        let models = Models::load(Path::new("no-such-native-live-runtime"));
        let mut window = Window::default();
        window.packets = 4;
        let (partial, binary) = models.predict(&window, 10., "fixture");
        assert_eq!(partial["available"], false);
        assert_eq!(binary["model_executed"], false);
    }
    #[test]
    #[ignore = "Requires preserved approved legacy checkpoint files; offline model replay, no capture"]
    fn approved_live_checkpoints_match_python_cpu_oracles() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(4)
            .unwrap();
        let partial = load(root, false).unwrap();
        let binary = load(root, true).unwrap();
        for (counts, label, score, binary_score) in [
            (
                [108., 14., 2., 0., 0., 20., 0.],
                "web_command-injection",
                0.5094696879386902,
                0.988793671131134,
            ),
            (
                [100., 100., 0., 0., 0., 100., 0.],
                "dos_http-flood-port-6668",
                0.22711171209812164,
                0.9880850911140442,
            ),
            (
                [1., 0., 0., 0., 0., 1., 0.],
                "web_sql-injection",
                0.5954315662384033,
                0.5969201326370239,
            ),
        ] {
            let p = partial
                .predict(&counts, "offline-synthetic-count-oracle")
                .unwrap();
            let b = binary
                .predict(&counts, "offline-synthetic-count-oracle")
                .unwrap();
            assert_eq!(p["label"], label);
            assert!((p["score"].as_f64().unwrap() - score).abs() < 1e-6);
            assert_eq!(b["label"], "suspicious");
            assert!((b["score"].as_f64().unwrap() - binary_score).abs() < 1e-6);
            assert!(
                p["explanation"]["quality"]["all_class_max_reconstruction_error"]
                    .as_f64()
                    .unwrap()
                    < 1e-4
            );
            assert_eq!(
                p["features"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|v| !v.is_null())
                    .count(),
                7
            );
        }
    }
}
