//! A package is admitted before serving UI or executing native libraries.
//! "background" entries (the GGUF and CUDA runtime) are size-checked here and hashed by the language server's
//! background pass before any of them is executed, so startup never blocks on gigabytes of hashing.
use crate::application::{hash_file,read_json};
use std::path::{Path,Component};
pub(crate) const CUDA:&str="artifacts/rust_migration/phase1/llm/native-b11564-cuda";
/// Only `artifacts/llm/<name>.gguf` and direct children of the CUDA runtime folder may defer hashing.
pub(crate) fn background_path(relative:&Path)->bool{
    let parent=relative.parent().unwrap_or(Path::new(""));
    (parent==Path::new("artifacts/llm")&&relative.extension().is_some_and(|e|e=="gguf"))||parent==Path::new(CUDA)
}
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
        if entry["verification"]=="background"{
            if !background_path(relative){return Err(format!("Background verification is only allowed for the language model and CUDA runtime: {}",relative.display()));}
            if std::fs::metadata(&path).map_err(|e|e.to_string())?.len()!=entry["bytes"].as_u64().ok_or("Runtime asset size missing")?{return Err(format!("Runtime resource size differs from its sealed package: {}",relative.display()));}
            continue;
        }
        let expected=entry["sha256"].as_str().ok_or("Runtime asset digest missing")?;
        if hash_file(&path)?!=expected{return Err(format!("Runtime resource differs from its sealed package: {}",relative.display()));}
    }
    Ok(())
}
#[cfg(test)]mod tests{
    use super::*;
    use serde_json::json;
    use std::path::PathBuf;
    fn package(entries:&[(&str,&[u8],Option<&str>,&str)])->PathBuf{
        let mut r=[0u8;8];getrandom::fill(&mut r).unwrap();let root=std::env::temp_dir().join(format!("netsec-assets-{}",r.iter().map(|v|format!("{v:02x}")).collect::<String>()));
        let files=entries.iter().map(|(relative,bytes,digest,verification)|{let p=root.join(relative);std::fs::create_dir_all(p.parent().unwrap()).unwrap();std::fs::write(&p,bytes).unwrap();
            json!({"path":relative,"bytes":bytes.len(),"sha256":digest.map(str::to_string).unwrap_or_else(||hash_file(&p).unwrap()),"verification":verification})}).collect::<Vec<_>>();
        std::fs::write(root.join("runtime_assets.json"),json!({"version":1,"files":files}).to_string()).unwrap();root.canonicalize().unwrap()
    }
    const GGUF:&str="artifacts/llm/m.gguf";const CUDA:&str="artifacts/rust_migration/phase1/llm/native-b11564-cuda/ggml-cuda.dll";
    #[test]fn verify_does_not_hash_background_entries(){
        let wrong="0".repeat(64);
        let root=package(&[("bin/a.txt",b"startup",None,"startup"),(GGUF,b"weights",Some(&wrong),"background"),(CUDA,b"cuda",Some(&wrong),"background")]);
        assert_eq!(verify(&root),Ok(()),"a background entry is checked for size, not hashed, at startup");
        let mut manifest=read_json(&root.join("runtime_assets.json"),1024*1024).unwrap();manifest["files"][1]["bytes"]=json!(8);std::fs::write(root.join("runtime_assets.json"),manifest.to_string()).unwrap();
        assert!(verify(&root).is_err_and(|e|e.contains("m.gguf")),"a background entry whose size differs is rejected");
        std::fs::remove_file(root.join(GGUF)).unwrap();assert!(verify(&root).is_err_and(|e|e.contains("m.gguf")),"a missing background entry is rejected");
        std::fs::write(root.join("bin/a.txt"),b"changed").unwrap();std::fs::write(root.join(GGUF),b"weights").unwrap();
        manifest["files"][1]["bytes"]=json!(7);std::fs::write(root.join("runtime_assets.json"),manifest.to_string()).unwrap();assert!(verify(&root).is_err_and(|e|e.contains("a.txt")),"startup entries are still hashed");
        std::fs::remove_dir_all(&root).unwrap();
    }
    #[test]fn verify_rejects_background_outside_llm_paths(){
        for relative in ["bin/netsec-runtime.exe","artifacts/llm/model.json","artifacts/llm/sub/m.gguf","artifacts/rust_migration/phase1/llm/native-b11564/llama-server.exe",
            "artifacts/rust_migration/phase1/llm/native-b11564-cuda/sub/x.dll","artifacts/application/native_models_v1/catalog.json"]{
            let root=package(&[(relative,b"x",None,"background")]);
            assert!(verify(&root).is_err_and(|e|e.contains("Background")),"{relative}");std::fs::remove_dir_all(&root).unwrap();
        }
        let root=package(&[(GGUF,b"w",None,"background"),(CUDA,b"c",None,"background")]);assert_eq!(verify(&root),Ok(()));std::fs::remove_dir_all(&root).unwrap();
    }
}
