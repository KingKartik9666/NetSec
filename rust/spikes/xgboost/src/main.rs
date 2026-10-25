//! Offline spike CLI: --checkpoint PATH [--sha256 HASH] --rows JSON.
use anyhow::{anyhow, Context, Result};
use netsec_xgboost_spike::Booster;
use serde_json::Value;
use std::{env, fs, path::PathBuf};

fn main() -> Result<()> {
    let mut checkpoint = None;
    let mut rows = None;
    let mut expected = None;
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| anyhow!("Missing value for {argument}"))?;
        match argument.as_str() {
            "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
            "--rows" => rows = Some(PathBuf::from(value)),
            "--sha256" => expected = Some(value),
            _ => return Err(anyhow!("Unknown spike option {argument}")),
        }
    }
    let checkpoint = checkpoint.ok_or_else(|| anyhow!("--checkpoint required"))?;
    let rows_path = rows.ok_or_else(|| anyhow!("--rows required"))?;
    if fs::metadata(&rows_path)?.len() > 8 * 1024 * 1024 {
        return Err(anyhow!("Rows exceed 8 MiB spike limit"));
    }
    let inputs: Value = serde_json::from_slice(&fs::read(rows_path)?).context("Read rows JSON")?;
    let rows: Vec<Vec<Option<f32>>> = serde_json::from_value(
        inputs
            .get("rows")
            .cloned()
            .ok_or_else(|| anyhow!("Expected rows array"))?,
    )?;
    if rows.is_empty() || rows.len() > 500 {
        return Err(anyhow!("Expected 1 to 500 rows"));
    }
    let booster = Booster::load_checkpoint(&checkpoint, expected.as_deref())?;
    let results = rows
        .iter()
        .map(|row| booster.evaluate(row))
        .collect::<Result<Vec<_>, _>>()?;
    println!(
        "{}",
        serde_json::to_string(
            &serde_json::json!({"engine":"Pure Rust numerical gbtree / TreeSHAP spike","features":booster.num_features(),"classes":booster.num_classes(),"trees":booster.num_trees(),"results":results})
        )?
    );
    Ok(())
}
