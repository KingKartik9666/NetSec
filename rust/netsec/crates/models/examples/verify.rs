//! Native CPU parity against the offline Python oracle. No Python is invoked.
use netsec_models::{ModelCatalog,PredictionCache};
use serde_json::{json,Value};
use std::{env,fs,path::Path,time::Instant};

fn main()->Result<(),Box<dyn std::error::Error>>{
    let arguments:Vec<String>=env::args().collect();
    let root=Path::new(arguments.get(1).ok_or("Expected native export directory")?);
    let catalog=ModelCatalog::load(root)?;
    let oracle=arguments.get(3).map(String::as_str).unwrap_or("golden.json");
    let golden:Value=serde_json::from_slice(&fs::read(root.join(oracle))?)?;
    let rows=golden["rows"].as_array().ok_or("Missing golden rows")?;
    let outputs=golden["outputs"].as_object().ok_or("Missing golden outputs")?;
    let mut reports=Vec::new();let mut success=true;
    let mut caches:Vec<PredictionCache>=(0..rows.len()).map(|_|PredictionCache::default()).collect();
    // Check every unique branch before the hybrids, matching extensive inference.
    let mut ids:Vec<_>=outputs.keys().collect();
    ids.sort_by_key(|id|(catalog.info(id).map(|i|i.kind=="hybrid").unwrap_or(true),*id));
    for id in ids{
        let expected=&outputs[id];
        eprintln!("VERIFY {id} {} rows CPU",rows.len());
        let started=Instant::now();let model=catalog.load_model(id)?;
        let load_seconds=started.elapsed().as_secs_f64();let started=Instant::now();
        let mut raw_difference=0.0f64;let mut calibrated_difference=0.0f64;let mut confidence_difference=0.0f64;
        let mut changed=0usize;let mut outside=0usize;
        let mut mismatches=Vec::new();
        for (i,row) in rows.iter().enumerate(){
            let input:Vec<f64>=row["values"].as_array().ok_or("Missing row values")?.iter().map(|v|v.as_f64().unwrap_or(f64::NAN)).collect();
            let result=model.predict_cached(&input,&mut caches[i])?;let expected=&expected[i];
            if result.predicted!=expected["predicted"].as_u64().ok_or("Missing expected prediction")? as usize{changed+=1;}
            for (actual,key,maximum) in [(&result.raw_probabilities,"raw_probabilities",&mut raw_difference),
                                       (&result.probabilities,"probabilities",&mut calibrated_difference)]{
                if let Some(actual)=actual{
                    let reference=expected[key].as_array().ok_or("Missing oracle probability array")?;
                    if actual.len()!=reference.len(){return Err("Oracle output dimension differs".into())}
                    for (a,b) in actual.iter().zip(reference){let b=b.as_f64().ok_or("Bad oracle probability")?;let delta=(*a-b).abs();*maximum=maximum.max(delta);
                        if delta>5e-5+5e-4*b.abs(){outside+=1}
                    }
                }
            }
            if let (Some(a),Some(b))=(result.confidence,expected["confidence"].as_f64()){
                confidence_difference=confidence_difference.max((a-b).abs());if (a-b).abs()>5e-5+5e-4*b.abs(){outside+=1;
                    mismatches.push(json!({"oracle_row":i,"row_id":row["id"],"row_index":row["row_index"],"actual":a,"expected":b,"hem":model.hem_details(&input)?}));}
            }
        }
        let passed=changed==0&&outside==0;success&=passed;
        reports.push(json!({"id":id,"rows":rows.len(),"passed":passed,"changed_decisions":changed,"outside_tolerance":outside,
            "max_raw_absolute_difference":raw_difference,"max_calibrated_absolute_difference":calibrated_difference,
            "max_confidence_absolute_difference":confidence_difference,"mismatches":mismatches,"load_seconds":load_seconds,"predict_seconds":started.elapsed().as_secs_f64()}));
    }
    let report=json!({"status":if success{"passed"}else{"failed"},"python_invoked":false,"inference_device":"cpu",
        "test_loaded":false,"oracle_file":oracle,"shared_branch_cache":true,"absolute_tolerance":5e-5,"relative_tolerance":5e-4,"models":reports});
    let serialized=serde_json::to_string_pretty(&report)?;
    if let Some(path)=arguments.get(2){fs::write(path,&serialized)?}
    println!("{serialized}");
    if !success{std::process::exit(1)}Ok(())
}
