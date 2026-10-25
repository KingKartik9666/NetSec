//! A package is admitted before serving UI or executing native libraries.
use crate::application::{hash_file,read_json};
use std::path::{Path,Component};

pub fn verify(root:&Path)->Result<(),String>{
    let manifest=read_json(&root.join("runtime_assets.json"),1024*1024)
        .map_err(|e|format!("Runtime assets are not sealed; run tools/stage_native_app.py after building: {e}"))?;
    if manifest["version"]!=1{return Err("Unsupported runtime asset manifest".into());}
    let files=manifest["files"].as_array().ok_or("Runtime asset list missing")?;
    if files.is_empty()||files.len()>2048{return Err("Runtime asset list outside its bounds".into());}
    for entry in files{
        if entry["verification"]=="model_load"{continue;}
        let relative=Path::new(entry["path"].as_str().ok_or("Runtime asset path missing")?);
        if relative.components().any(|c|!matches!(c,Component::Normal(_))){return Err("Runtime asset escapes package".into());}
        let path=root.join(relative).canonicalize().map_err(|e|format!("Missing runtime asset {}: {e}",relative.display()))?;
        if !path.starts_with(root){return Err("Runtime asset resolves outside package".into());}
        let expected=entry["sha256"].as_str().ok_or("Runtime asset digest missing")?;
        if hash_file(&path)?!=expected{return Err(format!("Runtime resource differs from its sealed package: {}",relative.display()));}
    }
    Ok(())
}
