//! The live77 packet-window model: one SHA-pinned 252-feature, 77-class XGBoost bundle, distinct from the 655/445-feature CSV
//! catalog. Output order (bundle `output.order`): raw softprob -> temperature calibration -> class merge recon_ping-sweep into
//! recon_scan -> argmax of the merged vector (lowest index on ties). No binary attack probability exists for this model.
use super::packet::Window;
use netsec_models::{argmax, decision, ClassMerge};
use netsec_xgboost_spike::Booster;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

pub const LIVE77_MODEL_ID: &str = "live77_xgboost";
const LIVE77_BUNDLE: &str = "artifacts/application/live_models_v1/live77_xgboost/bundle.json";
const LIVE77_BUNDLE_SHA: &str = "537d2ad4f4d61d01af8e2c8decd9865c53015283acea268aca932ddab5e9fca2";

const LIVE77_CHECKPOINT_SHA: &str = "2ebc7f2da863d79d65b03ed785752b6b48d1bd273f0f446fc9235d059ac272e3";
/// sha256 of the names joined by "\n" (live77_contract.json feature_names_sha256 / model_schema.json order).
const LIVE77_NAMES_SHA: &str = "d8c7b629b08393ef79db5bb9ea42cae1cff155f3cc81b8a5c211e41fc3c67bfd";
/// sha256 of the classes joined by "\n" (class_merge_20261101/spec.json classes == model_schema.json classes).
const LIVE77_CLASSES_SHA: &str = "fcd44d8e0d723f2011fc9b960b46580da4e4c919e2d43e4030378b38098a9bab";
const CLASS_MERGE_SPEC_SHA: &str = "102373ce586953a44b08228ddc094e23a1ff883bcfe1625292788bedd9ff9514";
/// Owner decision 3 Nov 2026: 10 s tumbling windows (overrides the plan's 1 s).
const WINDOW_SECONDS: u64 = 10;
const SHAP_TOLERANCE: f64 = 1e-4;
const MERGE_NOTE: &str = "recon_ping-sweep probability was merged into recon_scan after calibration; exact TreeSHAP is shown for both classes side by side";

struct Model {
    booster: Booster,
    names: Vec<String>,
    classes: Vec<String>,
    merge: ClassMerge,
    calibration: Value,
    sha: String,
    bundle_sha: String,
    version: String,
    label: String,
    note: String,
}

fn joined_sha(values: &[String]) -> String {
    format!("{:x}", Sha256::digest(values.join("\n").as_bytes()))
}

fn load(root: &Path) -> Result<Model, String> {
    load_bundle(root, &root.join(LIVE77_BUNDLE), LIVE77_BUNDLE_SHA)
}

/// Every guard runs before the checkpoint is read; any failure leaves the model unavailable with no substitute.
fn load_bundle(root: &Path, bundle: &Path, expected: &str) -> Result<Model, String> {
    if fs::metadata(bundle).map_err(|e| format!("Live77 bundle unavailable: {e}"))?.len() > 1024 * 1024 {
        return Err("Live bundle exceeds its size bound".into());
    }
    let bytes = fs::read(bundle).map_err(|e| e.to_string())?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    if hash != expected {
        return Err("Live77 bundle SHA256 differs from the explicitly approved version".into());
    }
    let m: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let strings = |key: &str| serde_json::from_value::<Vec<String>>(m[key].clone()).map_err(|e| format!("Live77 {key}: {e}"));
    let (names, classes) = (strings("feature_names")?, strings("classes")?);
    if m["model_id"] != LIVE77_MODEL_ID || m["kind"] != "xgboost" || m["objective"] != "multi:softprob" {
        return Err("Live77 model id, kind or objective differs".into());
    }
    if names.len() != 252 || m["n_features"] != 252 || joined_sha(&names) != LIVE77_NAMES_SHA {
        return Err("Live77 feature names or their order differ from the 252-feature schema".into());
    }
    if classes.len() != 77 || m["n_classes"] != 77 || joined_sha(&classes) != LIVE77_CLASSES_SHA {
        return Err("Live77 class order differs from the 77-class spec".into());
    }
    if m["benign_index"] != 0 || classes[0] != "benign" {
        return Err("Live77 benign index differs".into());
    }
    if m["window_seconds"] != WINDOW_SECONDS || m["window"]["length_seconds"] != WINDOW_SECONDS {
        return Err(format!("Live77 window differs from the owner's {WINDOW_SECONDS} s window"));
    }
    if m["test_evaluated"] != false || m["live_validation"] != false {
        return Err("Unexpected live-model evaluation provenance".into());
    }
    let merge = &m["class_merge"];
    let parsed = ClassMerge::parse(merge, &classes).map_err(|e| format!("Live77 class merge: {e}"))?;
    if parsed != Some(ClassMerge { source: 71, target: 70 })
        || merge["name"] != "ping_sweep_into_recon_scan"
        || merge["version"] != 1
        || merge["source_index"] != 71
        || merge["target_index"] != 70
        || merge["spec_sha256"] != CLASS_MERGE_SPEC_SHA
    {
        return Err("Live77 class merge differs from spec ping_sweep_into_recon_scan v1".into());
    }
    let calibration = m["calibration"].clone();
    if calibration["kind"] != "temperature" || !calibration["temperature"].as_f64().is_some_and(|t| t.is_finite() && t > 0.) || !m["logit_adjustment"].is_null() {
        return Err("Live77 calibration differs from the frozen temperature rule".into());
    }
    if !m["output"]["binary_attack_probability"].is_null() {
        return Err("Live77 has no binary calibration; a binary attack probability is refused".into());
    }
    let text = |key: &str| m[key].as_str().filter(|v| !v.trim().is_empty()).map(str::to_owned).ok_or(format!("Live77 {key} missing"));
    let (label, note) = (text("label")?, text("provenance_note")?);
    let sha = m["checkpoint_sha256"].as_str().ok_or("Live checkpoint SHA256 missing")?.to_owned();
    if sha != LIVE77_CHECKPOINT_SHA {
        return Err("Live77 checkpoint SHA256 differs from the approved model".into());
    }
    let relative = m["checkpoint"].as_str().ok_or("Live checkpoint missing")?;
    let checkpoint = bundle.parent().ok_or("Live bundle folder missing")?.join(relative).canonicalize().map_err(|e| e.to_string())?;
    if !checkpoint.starts_with(root.canonicalize().map_err(|e| e.to_string())?) {
        return Err("Live checkpoint must stay in the packaged application root".into());
    }
    let booster = Booster::load_checkpoint(&checkpoint, Some(&sha)).map_err(|e| e.to_string())?;
    if booster.num_features() != names.len() || booster.num_classes() != classes.len() {
        return Err("Live checkpoint dimensions differ".into());
    }
    let version = m["source"]["job"].as_str().unwrap_or(LIVE77_MODEL_ID).to_owned();
    Ok(Model { booster, names, classes, merge: parsed.unwrap(), calibration, sha, bundle_sha: hash, version, label, note })
}

impl Model {
    fn explanation(&self, row: &[Option<f32>], contributions: &[f32], margin: f32, target: usize, max_error: f64) -> Result<Value, String> {
        let bias = *contributions.last().ok_or("Empty live SHAP output")?;
        let mut features=contributions[..self.names.len()].iter().enumerate().map(|(i,v)|json!({"feature":self.names[i],"value":row[i],"contribution":v,"is_missing":row[i].is_none()})).collect::<Vec<_>>();
        features.sort_by(|a, b| b["contribution"].as_f64().unwrap_or(0.).abs().total_cmp(&a["contribution"].as_f64().unwrap_or(0.).abs()));
        Ok(json!({"method":"exact_tree_shap","scope":"local","target_label":self.classes[target],"units":"raw class margin",
            "bias":bias,"margin":margin,"features":features,"quality":{"all_class_max_reconstruction_error":max_error},
            "warnings":["Missing-input routing is model behavior, not measured traffic","TreeSHAP is not causal evidence or probability percentage points"]}))
    }
    fn predict(&self, row: &[Option<f32>], observation_id: &str) -> Result<Value, String> {
        if row.len() != self.names.len() {
            return Err(format!("Live77 row has {} inputs; the model needs {}", row.len(), self.names.len()));
        }
        let row = row.iter().map(|v| v.filter(|v| !v.is_nan())).collect::<Vec<_>>();
        let raw = self.booster.predict(&row).map_err(|e| e.to_string())?;
        let (_, calibrated, _) = decision(&raw.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(), &self.calibration).map_err(|e| e.to_string())?;
        let mut merged = calibrated.clone();
        self.merge.apply(&mut merged);
        let target = argmax(&merged);
        let raw_target = argmax(&raw.iter().map(|v| f64::from(*v)).collect::<Vec<_>>());
        let mut order = (0..merged.len()).collect::<Vec<_>>();
        order.sort_by(|a, b| merged[*b].total_cmp(&merged[*a]).then(a.cmp(b)));
        let observed = row.iter().filter(|v| v.is_some()).count();
        let mut result = json!({"available":true,"model_executed":true,"model_id":LIVE77_MODEL_ID,"model_label":self.label,"provenance_note":self.note,
            "label":self.classes[target],"is_attack":target!=0,"score":merged[target],"score_kind":"calibrated_probability_after_class_merge",
            "top_classes":order.iter().take(3).map(|i|json!({"label":self.classes[*i],"score":merged[*i]})).collect::<Vec<_>>(),
            "probabilities":merged,"calibrated_probabilities":calibrated,"raw_probabilities":raw,"raw_label":self.classes[raw_target],
            "class_merge":{"source_class":self.classes[self.merge.source],"target_class":self.classes[self.merge.target],"applied_after":"temperature calibration"},
            "class_count":self.classes.len(),"classes":self.classes,"window_seconds":WINDOW_SECONDS,
            "observation_id":observation_id,"checkpoint_sha256":self.sha,"bundle_sha256":self.bundle_sha,"model_version":self.version,
            "feature_coverage":{"observed":observed,"missing":self.names.len()-observed,"total":self.names.len(),"profile":"live77_model_row"},
            "features":row,"feature_names":self.names,"novelty":"not evaluated","live_validation":false,"test_evaluated":false,
            "limitations":["DataSense-trained model; its accuracy on real traffic from this PC has not been measured","Unknown attacks and encrypted application contents are not evaluated",
                "Calibrated probabilities are model estimates, not proof of safety or an attack","No binary attack probability is reported for this model"]});
        match (self.booster.margins(&row), self.booster.contributions(&row)) {
            (Ok(margins), Ok(all)) => {
                let max_error = all.iter().zip(&margins).map(|(values, margin)| (values.iter().map(|v| f64::from(*v)).sum::<f64>() - f64::from(*margin)).abs()).fold(0.0, f64::max);
                if max_error <= SHAP_TOLERANCE {
                    let mut explanation = self.explanation(&row, &all[target], margins[target], target, max_error)?;
                    if target == self.merge.target {
                        let source = self.merge.source;
                        explanation["merge_note"] = json!(MERGE_NOTE);
                        let mut side = self.explanation(&row, &all[source], margins[source], source, max_error)?;
                        side["merge_note"] = json!(MERGE_NOTE);
                        result["merged_source_explanation"] = side;
                        result["merge_margins"] = json!({"units":"raw class margin, not probability",(self.classes[target].as_str()):margins[target],(self.classes[source].as_str()):margins[source]});
                    }
                    result["explanation"] = explanation;
                } else {
                    result["explanation_error"] = json!(format!("TreeSHAP reconstruction error {max_error} exceeded {SHAP_TOLERANCE:e}"));
                }
            }
            (Err(error), _) | (_, Err(error)) => result["explanation_error"] = json!(error.to_string()),
        }
        Ok(result)
    }
}

/// One live model. `predict` takes the ordered 252-input model row and never reads `Window::counts`.
pub struct Models {
    model: Result<Model, String>,
}
impl Models {
    pub fn load(root: &Path) -> Self {
        Self { model: load(root) }
    }
    /// Bundle feature names, in model order, for the extractor's by-name row projection (plan step 5).
    #[allow(dead_code)]
    pub fn feature_names(&self) -> Option<&[String]> {
        self.model.as_ref().ok().map(|m| m.names.as_slice())
    }
    /// Scores a bundle-ordered row (NaN is read as missing).
    #[allow(dead_code)]
    pub fn predict(&self, row: &[Option<f32>], id: &str) -> Value {
        match &self.model {
            Ok(model) => model.predict(row, id).unwrap_or_else(unavailable),
            Err(error) => unavailable(error.clone()),
        }
    }
    /// Scores the window's live77 row, projected by the bundle's feature names (features.rs Live77Features). A
    /// capture-loss window is all missing and is not scored; without the device MAC there is no row. The binary slot
    /// is null because no live77 binary calibration exists. The 10 s duration gate stays with the caller (live.rs).
    pub fn predict_window(&self, window: &Window, id: &str) -> (Value, Value) {
        let multiclass = match &self.model {
            Ok(_) if window.capture_drops > 0 => unavailable("Capture loss: this window's live77 inputs are missing, so it is not scored".into()),
            Ok(model) => match window.live77_row(&model.names) {
                Ok(row) => model.predict(&row, id).unwrap_or_else(unavailable),
                Err(error) => unavailable(error),
            },
            Err(error) => unavailable(error.clone()),
        };
        (multiclass, Value::Null)
    }
}
fn unavailable(error: String) -> Value {
    json!({"available":false,"model_executed":false,"error":error,"limitations":"This model capability failed or this window was incompatible. Observation and independent rules continue; no substitute prediction was returned."})
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    fn root() -> &'static Path {
        Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(4).unwrap()
    }
    fn golden() -> Value {
        let path = root().join(LIVE77_BUNDLE).with_file_name("golden_live77.json");
        serde_json::from_slice(&fs::read(&path).unwrap_or_else(|e| panic!("live77 golden {path:?} unavailable: {e}"))).unwrap()
    }
    fn row(values: &Value) -> Vec<Option<f32>> {
        values.as_array().unwrap().iter().map(|v| v.as_f64().map(|v| v as f32)).collect()
    }
    fn floats(values: &Value) -> Vec<f64> {
        values.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
    }
    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
    }
    /// Unavailable JSON with nothing that could be read as a verdict.
    fn assert_no_substitute(value: &Value) {
        assert_eq!((value["available"].as_bool(), value["model_executed"].as_bool()), (Some(false), Some(false)), "{value}");
        for field in ["label", "score", "top_classes", "is_attack", "explanation", "raw_probabilities", "attack_score", "attack_probability"] {
            assert!(value.get(field).is_none(), "unavailable output carries {field}: {value}");
        }
        assert!(value["error"].as_str().is_some_and(|e| !e.is_empty()));
    }

    #[test]
    fn missing_bundle_is_unavailable_and_never_replaced() {
        let models = Models::load(Path::new("no-such-native-live-runtime"));
        assert_no_substitute(&models.predict(&vec![None; 252], "fixture"));
        let mut window = Window::default();
        window.packets = 4;
        let (multiclass, binary) = models.predict_window(&window, "fixture");
        assert_no_substitute(&multiclass);
        assert!(binary.is_null(), "no live77 binary calibration exists; binary_prediction must stay null");
    }

    #[test]
    fn rejected_bundles_give_unavailable_json_without_substitute() {
        let bundle = root().join(LIVE77_BUNDLE);
        let bytes = fs::read(&bundle).expect("live77 bundle.json present in the workspace");
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), LIVE77_BUNDLE_SHA);
        let error = load_bundle(root(), &bundle, &"0".repeat(64)).err().expect("SHA mismatch must be refused");
        assert!(error.contains("SHA256"), "{error}");
        assert_no_substitute(&Models { model: Err(error) }.predict(&vec![None; 252], "fixture"));
        let manifest: Value = serde_json::from_slice(&bytes).unwrap();
        let folder = std::env::temp_dir().join(format!("netsec-live77-models-{}", std::process::id()));
        fs::create_dir_all(&folder).unwrap();
        let cases: Vec<(&str, Box<dyn Fn(&mut Value)>, &str)> = vec![
            ("class order swapped 1<->2", Box::new(|m| m["classes"].as_array_mut().unwrap().swap(1, 2)), "class order"),
            ("class order swapped 70<->71", Box::new(|m| m["classes"].as_array_mut().unwrap().swap(70, 71)), "class order"),
            ("class removed", Box::new(|m| { m["classes"].as_array_mut().unwrap().pop(); }), "class order"),
            ("merge missing", Box::new(|m| { m.as_object_mut().unwrap().remove("class_merge"); }), "class merge"),
            ("merge null", Box::new(|m| m["class_merge"] = Value::Null), "class merge"),
            ("merge reversed", Box::new(|m| { m["class_merge"]["source_class"] = json!("recon_scan"); m["class_merge"]["target_class"] = json!("recon_ping-sweep"); }), "class merge"),
            ("merge spec differs", Box::new(|m| m["class_merge"]["spec_sha256"] = json!("0".repeat(64))), "class merge"),
            ("window 1 s", Box::new(|m| m["window_seconds"] = json!(1)), "window"),
            ("window missing", Box::new(|m| { m.as_object_mut().unwrap().remove("window_seconds"); }), "window"),
            ("window length 1 s", Box::new(|m| m["window"]["length_seconds"] = json!(1)), "window"),
            ("test evaluated", Box::new(|m| m["test_evaluated"] = json!(true)), "provenance"),
            ("live validated", Box::new(|m| m["live_validation"] = json!(true)), "provenance"),
            ("features swapped", Box::new(|m| m["feature_names"].as_array_mut().unwrap().swap(0, 1)), "feature"),
            ("benign index", Box::new(|m| m["benign_index"] = json!(1)), "benign"),
            ("checkpoint differs", Box::new(|m| m["checkpoint_sha256"] = json!("0".repeat(64))), "checkpoint"),
            ("vector calibration", Box::new(|m| m["calibration"]["kind"] = json!("vector")), "calibration"),
            ("logit adjustment", Box::new(|m| m["logit_adjustment"] = json!([0.0])), "calibration"),
            ("other model id", Box::new(|m| m["model_id"] = json!("live_partial_xgboost")), "model id"),
            ("binary probability", Box::new(|m| m["output"]["binary_attack_probability"] = json!(0.5)), "binary"),
            ("label missing", Box::new(|m| m["label"] = json!("")), "label"),
        ];
        for (name, mutate, reason) in cases {
            let mut changed = manifest.clone();
            mutate(&mut changed);
            let text = serde_json::to_vec_pretty(&changed).unwrap();
            let path = folder.join("bundle.json");
            fs::write(&path, &text).unwrap();
            // The mutated bundle's own hash is "approved" here so each schema guard is reached independently of the SHA pin.
            let error = load_bundle(&folder, &path, &format!("{:x}", Sha256::digest(&text))).err().unwrap_or_else(|| panic!("{name}: accepted"));
            assert!(error.to_lowercase().contains(reason), "{name}: {error}");
            assert_no_substitute(&Models { model: Err(error) }.predict(&vec![None; 252], "fixture"));
        }
        let _ = fs::remove_dir_all(&folder);
    }

    #[test]
    fn approved_bundle_rejects_wrong_row_width() {
        let models = Models::load(root());
        assert!(models.model.is_ok(), "{:?}", models.model.as_ref().err());
        assert_no_substitute(&models.predict(&vec![None; 251], "fixture"));
        assert_no_substitute(&models.predict(&vec![Some(0.0); 253], "fixture"));
        let (multiclass, binary) = models.predict_window(&Window::default(), "fixture");
        assert_no_substitute(&multiclass);
        assert!(binary.is_null());
    }

    #[test]
    fn predict_window_scores_the_projected_live77_row() {
        let models = Models::load(root());
        let names = models.feature_names().unwrap().to_vec();
        let device = [2, 0, 0, 0, 0, 0x50];
        let fresh = || Window { live77: super::super::features::Live77Features::new(Some(device)), ..Window::default() };
        // A sent SYN with MSS 1460 (frames_v1.json tcp_syn_with_and_without_mss, first frame).
        let hex = "02000000000102000000005008004500002c0001400040066ef0c0a801320a000001c7380050000003e8000000006002faf000000000020405b4";
        let syn = (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect::<Vec<_>>();
        let mut window = fresh();
        window.add_live77_frame(&syn, syn.len() as u32);
        let row = window.live77_row(&names).unwrap();
        let (out, binary) = models.predict_window(&window, "fixture");
        assert!(binary.is_null());
        assert_eq!((out["available"].as_bool(), out["model_id"].as_str()), (Some(true), Some(LIVE77_MODEL_ID)), "{out}");
        assert_eq!(out["features"], json!(row));
        assert_eq!(out["features"][names.iter().position(|n| n == "network_mss_max").unwrap()], 1460.0);
        // An empty window with a known device is scored as a zero-packet row, as in training.
        let (empty, _) = models.predict_window(&fresh(), "empty");
        assert_eq!(empty["model_executed"], true, "{empty}");
        assert_eq!(empty["features"][names.iter().position(|n| n == "quality_no_packets_observed").unwrap()], 1.0);
        // Capture loss is not scored; an unknown device MAC gives no row and no substitute.
        window.capture_drops = 1;
        assert_no_substitute(&models.predict_window(&window, "drop").0);
        assert_no_substitute(&models.predict_window(&Window::default(), "no-mac").0);
    }

    #[test]
    fn golden_replay_matches_python_cpu_oracle() {
        let model = load(root()).unwrap();
        let golden = golden();
        assert_eq!((golden["partition"].as_str(), golden["test_loaded"].as_bool()), (Some("validation"), Some(false)));
        assert_eq!(golden["feature_names"], json!(model.names));
        assert_eq!(golden["classes"], json!(model.classes));
        let rows = golden["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 128);
        let (mut raw_max, mut cal_max, mut merged_max, mut margin_max, mut shap_sum_max, mut shap_max) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        let (mut flips, mut scan) = (0, 0);
        for (i, golden_row) in rows.iter().enumerate() {
            let mut input = row(&golden_row["values"]);
            assert_eq!(input.len(), 252);
            // NaN is missing: a Some(NaN) row must route exactly like the null row.
            if let Some(slot) = input.iter_mut().find(|v| v.is_none()) {
                *slot = Some(f32::NAN);
            }
            let out = model.predict(&input, "golden").unwrap();
            let input = input.into_iter().map(|v| v.filter(|v| !v.is_nan())).collect::<Vec<_>>();
            assert_eq!(out["features"], json!(input));
            let raw = floats(&out["raw_probabilities"]);
            raw_max = raw_max.max(max_diff(&raw, &floats(&golden_row["raw"])));
            cal_max = cal_max.max(max_diff(&floats(&out["calibrated_probabilities"]), &floats(&golden_row["calibrated"])));
            merged_max = merged_max.max(max_diff(&floats(&out["probabilities"]), &floats(&golden_row["merged"])));
            let decided = golden_row["decision"].as_u64().unwrap() as usize;
            assert_eq!(out["label"], json!(model.classes[decided]), "row {i}: decision differs");
            assert_eq!(out["raw_label"], json!(model.classes[golden_row["raw_decision"].as_u64().unwrap() as usize]), "row {i}");
            assert_eq!(out["score"].as_f64().unwrap(), floats(&out["probabilities"])[decided]);
            assert_eq!(out["top_classes"][0]["label"], out["label"]);
            assert_eq!((out["score_kind"].as_str(), out["model_id"].as_str()), (Some("calibrated_probability_after_class_merge"), Some(LIVE77_MODEL_ID)));
            assert!(out.get("attack_score").is_none() && out.get("attack_probability").is_none());
            assert_eq!(out["is_attack"], json!(decided != 0));
            assert_eq!((out["test_evaluated"].as_bool(), out["live_validation"].as_bool()), (Some(false), Some(false)));
            let margins = model.booster.margins(&input).unwrap().into_iter().map(f64::from).collect::<Vec<_>>();
            margin_max = margin_max.max(max_diff(&margins, &floats(&golden_row["margins"])));
            let explanation = &out["explanation"];
            assert_eq!((explanation["method"].as_str(), explanation["target_label"].as_str()), (Some("exact_tree_shap"), Some(model.classes[decided].as_str())), "row {i}: {}", out["explanation_error"]);
            shap_sum_max = shap_sum_max.max(explanation["quality"]["all_class_max_reconstruction_error"].as_f64().unwrap());
            let all = model.booster.contributions(&input).unwrap();
            for (class, expected) in golden_row["contributions"].as_object().unwrap() {
                let class = class.parse::<usize>().unwrap();
                shap_max = shap_max.max(max_diff(&all[class].iter().map(|v| f64::from(*v)).collect::<Vec<_>>(), &floats(expected)));
            }
            // The decided class's reported contributions are the Rust TreeSHAP values, keyed by feature name.
            let reported = explanation["features"].as_array().unwrap();
            assert_eq!(reported.len(), 252);
            for feature in reported {
                let index = model.names.iter().position(|n| n == feature["feature"].as_str().unwrap()).unwrap();
                assert_eq!(feature["contribution"].as_f64().unwrap(), f64::from(all[decided][index]));
                assert_eq!(feature["is_missing"], json!(input[index].is_none()));
            }
            if decided == 70 {
                scan += 1;
                let source = &out["merged_source_explanation"];
                assert_eq!((source["method"].as_str(), source["target_label"].as_str()), (Some("exact_tree_shap"), Some("recon_ping-sweep")));
                assert!(explanation["merge_note"].as_str().is_some_and(|n| n.contains("ping-sweep")));
                assert_eq!(out["merge_margins"]["units"], "raw class margin, not probability");
                assert_eq!(out["merge_margins"]["recon_scan"].as_f64().unwrap(), margins[70]);
                assert_eq!(out["merge_margins"]["recon_ping-sweep"].as_f64().unwrap(), margins[71]);
            } else {
                assert!(out.get("merged_source_explanation").is_none() && out.get("merge_margins").is_none());
            }
            flips += usize::from(golden_row["merge_flip"] == true);
        }
        println!("live77 golden replay over {} validation rows ({scan} recon_scan, {flips} merge flips): raw {raw_max:e}, calibrated {cal_max:e}, merged {merged_max:e}, margins {margin_max:e}, SHAP reconstruction {shap_sum_max:e}, SHAP vs Python {shap_max:e}", rows.len());
        assert!(raw_max <= 1e-6 && cal_max <= 1e-6 && merged_max <= 1e-6, "{raw_max} {cal_max} {merged_max}");
        assert!(margin_max <= 1e-4 && shap_sum_max <= 1e-4 && shap_max <= 1e-4, "{margin_max} {shap_sum_max} {shap_max}");
        assert!(flips > 0 && scan > 0, "golden must cover the class merge");
    }

    #[test]
    fn live77_predict_and_shap_timing_is_logged() {
        let model = load(root()).unwrap();
        let golden = golden();
        let rows = golden["rows"].as_array().unwrap().iter().take(16).map(|r| row(&r["values"])).collect::<Vec<_>>();
        let started = Instant::now();
        for input in &rows {
            model.booster.predict(input).unwrap();
        }
        let predict = started.elapsed().as_secs_f64() * 1e3 / rows.len() as f64;
        let started = Instant::now();
        for input in &rows {
            model.booster.contributions(input).unwrap();
        }
        let shap = started.elapsed().as_secs_f64() * 1e3 / rows.len() as f64;
        let started = Instant::now();
        for input in &rows {
            model.predict(input, "timing").unwrap();
        }
        let window = started.elapsed().as_secs_f64() * 1e3 / rows.len() as f64;
        println!("live77 CPU timing ({} build, {} trees, {} rows): predict {predict:.2} ms, all-class TreeSHAP {shap:.2} ms, full window output {window:.2} ms per 10 s window",
            if cfg!(debug_assertions) { "debug" } else { "release" }, model.booster.num_trees(), rows.len());
        assert!(predict.is_finite() && shap.is_finite() && window.is_finite());
    }
}
