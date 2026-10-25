#![forbid(unsafe_code)]
use netsec_lime_spike::{explain, reference::ReferenceData, regression, Error, Result};
use netsec_xgboost_spike::Booster;
use serde::Deserialize;
use std::io::Read;
use std::path::Path;

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Explain {
        checkpoint: String,
        checkpoint_sha256: Option<String>,
        reference: String,
        feature_names: Vec<String>,
        class_names: Vec<String>,
        cases: Vec<Case>,
    },
    Fit {
        data: Vec<Vec<f64>>,
        labels: Vec<f64>,
        weights: Vec<f64>,
        maximum: usize,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    values: Vec<Option<f32>>,
    #[serde(default)]
    packet: bool,
    #[serde(default = "seed")]
    seed: u64,
    #[serde(default = "samples")]
    samples: usize,
}

fn seed() -> u64 {
    42
}
fn samples() -> usize {
    2048
}

fn main() {
    if let Err(error) = run() {
        eprintln!("LIME spike failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut bytes = Vec::new();
    match args.as_slice() {
        [] => std::io::stdin()
            .take(16 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?,
        [flag, path] if flag == "--request" => std::fs::File::open(path)?
            .take(16 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?,
        _ => {
            return Err(Error::Invalid(
                "Usage: netsec-lime-spike [--request request.json] (otherwise JSON from stdin)"
                    .into(),
            ))
        }
    };
    if bytes.len() > 16 * 1024 * 1024 {
        return Err(Error::Invalid("Request exceeds 16 MiB".into()));
    }
    let request: Request = serde_json::from_slice(&bytes)?;
    let output = match request {
        Request::Explain {
            checkpoint,
            checkpoint_sha256,
            reference,
            feature_names,
            class_names,
            cases,
        } => {
            let booster =
                Booster::load_checkpoint(Path::new(&checkpoint), checkpoint_sha256.as_deref())
                    .map_err(|error| {
                        Error::Invalid(format!("XGBoost model unavailable: {error}"))
                    })?;
            if booster.num_features() != feature_names.len()
                || booster.num_classes() != class_names.len()
                || cases.len() > 64
            {
                return Err(Error::Invalid(
                    "Model, schema, class labels or bounded case count differ".into(),
                ));
            }
            let reference = ReferenceData::load(Path::new(&reference), &feature_names)?;
            let mut outputs = Vec::new();
            for case in cases {
                let start = std::time::Instant::now();
                let lime = explain::explain(
                    &reference,
                    &case.values,
                    &class_names,
                    case.packet,
                    case.seed,
                    case.samples,
                    |row| booster.predict(row).map_err(|error| error.to_string()),
                )?;
                outputs.push(serde_json::json!({ "id": case.id, "lime": lime, "elapsed_ms": start.elapsed().as_secs_f64() * 1000.0 }));
            }
            serde_json::json!({ "runtime": "pure Rust LIME and pure Rust XGBoost UBJ inference", "llm_executed": false, "cases": outputs })
        }
        Request::Fit {
            data,
            labels,
            weights,
            maximum,
        } => {
            let selected = regression::select_features(&data, &labels, &weights, maximum)?;
            let subset: Vec<Vec<f64>> = data
                .iter()
                .map(|row| selected.iter().map(|i| row[*i]).collect())
                .collect();
            let fit = regression::ridge(&subset, &labels, &weights, 1.0)?;
            let local_pred = fit.intercept
                + fit
                    .coefficients
                    .iter()
                    .zip(&subset[0])
                    .map(|(b, x)| b * x)
                    .sum::<f64>();
            serde_json::json!({ "selected_features": selected, "fit": fit, "surrogate_prediction": local_pred })
        }
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}
