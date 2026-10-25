use anyhow::{Context, Result};
use netsec_llm_spike::{adapter, grounding, Request};
use serde_json::{json, Value};
use std::{env, fs};

fn main() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        anyhow::bail!("Usage: netsec-llm-spike native|child|grounding request.json output.json");
    }
    let input = fs::read_to_string(&args[2]).context("read spike request")?;
    let output = if args[1] == "grounding" {
        let cases: Value = serde_json::from_str(&input)?;
        let results = cases["cases"]
            .as_array()
            .context("grounding cases")?
            .iter()
            .map(|case| {
                let data = &case["input"];
                let result = grounding::check(
                    data["text"].as_str().unwrap_or(""),
                    &data["evidence"],
                    data["passages"]
                        .as_array()
                        .map(Vec::as_slice)
                        .unwrap_or(&[]),
                    data["finish_reason"].as_str(),
                )?;
                Ok(json!({"id":case["id"],"output":result}))
            })
            .collect::<Result<Vec<_>>>()?;
        json!({"cases":results})
    } else {
        let request: Request = serde_json::from_str(&input)?;
        let mut output = match args[1].as_str() {
            "native" => adapter::native(&request)?,
            "child" => adapter::child(&request)?,
            _ => anyhow::bail!("Unknown spike mode"),
        };
        output["grounding"] = grounding::check(
            output["text"].as_str().context("generated text")?,
            &request.evidence,
            &request.passages,
            output["finish_reason"].as_str(),
        )?;
        output
    };
    fs::write(&args[3], serde_json::to_vec_pretty(&output)?)?;
    Ok(())
}
