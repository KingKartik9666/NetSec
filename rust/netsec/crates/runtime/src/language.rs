//! Local verified generation: a loopback llama-server child writes plain English from one server-owned evidence packet;
//! every sentence is verified before it is shown, and a failure is replaced by labelled NetSec template points.
pub mod evidence;
pub mod explain;
pub mod packet;
pub mod server;
pub mod verify;
use crate::application::{Application,read_json};
use netsec_core::jobs::Job;
use serde::{Deserialize,Serialize};
use serde_json::{json,Value};
use server::Generate;
use std::{path::{Path,PathBuf},sync::atomic::AtomicBool};
const MANIFEST:&str="artifacts/llm/model.json";
const NATIVE:&str="artifacts/rust_migration/phase1/llm/native-b11564";
/// The installed GGUF is described by its manifest; no model is bundled by default.
#[derive(Deserialize)]
pub(crate) struct Model {pub name:String,pub file:String,pub sha256:String,pub prompt_template:String,#[serde(default)]pub license:String,#[serde(default)]pub device:Device,#[serde(default="token_ceiling")]pub max_tokens:u32}
fn token_ceiling()->u32{300}
/// Language-server device only; detectors always stay on CPU. `NETSEC_LLM_DEVICE=cpu` forces CPU (measurement/fallback).
#[derive(Serialize,Deserialize,Clone,Copy,Debug,Default,PartialEq,Eq)]#[serde(rename_all="lowercase")]
pub enum Device {#[default]Gpu,Cpu}
impl Device {
    pub(crate) fn requested(manifest:Device,env:Option<&str>)->Device{if env.is_some_and(|v|v.trim().eq_ignore_ascii_case("cpu")){Device::Cpu}else{manifest}}
    pub(crate) fn as_str(self)->&'static str{match self{Device::Gpu=>"gpu",Device::Cpu=>"cpu"}}
}
impl Model {
    pub(crate) fn device(&self)->Device{Device::requested(self.device,std::env::var("NETSEC_LLM_DEVICE").ok().as_deref())}
    pub(crate) fn cap(&self,kind:u32)->u32{kind.min(self.max_tokens)}
}
impl Model {
    pub(crate) fn path(&self,root:&Path)->Result<PathBuf,String>{
        let file=Path::new(&self.file);
        if file.components().count()!=1||file.extension().and_then(|e|e.to_str())!=Some("gguf"){return Err("Language model manifest names an invalid GGUF file".into());}
        Ok(root.join("artifacts/llm").join(file))
    }
}
pub(crate) fn installed(root:&Path)->Result<Model,String>{
    let manifest=read_json(&root.join(MANIFEST),64*1024).map_err(|_|"No local language model is installed; model and XAI evidence remain available".to_string())?;
    let model:Model=serde_json::from_value(manifest).map_err(|_|"Language model manifest is invalid".to_string())?;
    if !model.prompt_template.contains("{system}")||!model.prompt_template.contains("{user}"){return Err("Language model prompt template must contain {system} and {user}".into());}
    Ok(model)
}
/// `device` is the running child's device (`None` while unloaded); `requested_device` is the manifest/override choice. Detectors stay CPU-only.
pub fn status(root:&Path,device:Option<Device>)->Value{
    let model=installed(root);
    let ready=model.as_ref().is_ok_and(|m|m.path(root).is_ok_and(|p|p.is_file()))&&root.join(NATIVE).join("llama-server.exe").is_file();
    let device=device.map(Device::as_str);
    json!({"status":if ready{"ready"}else{"unavailable"},"model":model.as_ref().ok().map(|m|m.name.as_str()),"license":model.as_ref().ok().map(|m|m.license.as_str()),"provider":"Local llama.cpp b11564",
        "mode":"verified_generation","device":device,"language_device":device,"requested_device":model.as_ref().ok().map(|m|m.device().as_str()),"detectors_cpu_only":true,"cpu_threads":8,"offline":true,"automatic_generation":false,
        "reason":if ready{"The local model writes plain English on request; NetSec checks every sentence against the stored evidence and replaces failed output with labelled template text".to_string()}else{model.err().unwrap_or_else(||"Verified local GGUF/native runtime is missing".into())}})
}
/// Everything one explanation may use, built once from server-owned data: the packet the model sees, the rules its sentences must pass, the template points.
/// `names` are the catalog's class ids and feature names; the chat gate treats them as topic words and masks them before misuse matching.
pub struct Evidence {pub packet:packet::Packet,pub rules:verify::Rules,pub points:Vec<evidence::Point>,pub names:Vec<String>}
impl Evidence {
    pub fn new(prediction:&Value,label:&str,explanation:Option<&Value>,class_count:usize,classes:&[String],all_features:&[String])->Self{
        let packet=packet::build_with(prediction,label,class_count,explanation,classes,all_features);
        let rules=verify::rules(&packet,all_features,&packet::lexicon(classes));
        Evidence{points:evidence::points(&packet),packet,rules,names:classes.iter().chain(all_features).cloned().collect()}
    }
    /// The selected XAI method's preset; `None` without a selected XAI method, when the explainer does not start (owner, 3 Nov 2026).
    pub fn method(&self)->Option<evidence::Method>{evidence::Method::of(&self.packet)}
    /// `Err(CHOOSE_XAI)` unless an XAI method with a preset was selected; checked before any model call.
    pub fn require_xai(&self)->Result<evidence::Method,String>{self.method().ok_or_else(||evidence::CHOOSE_XAI.to_string())}
}
/// A live prediction's own class count, class list and feature names (live models are not in the catalog).
pub fn live_schema(prediction:&Value)->(usize,Vec<String>,Vec<String>){
    let strings=|key:&str|prediction[key].as_array().into_iter().flatten().filter_map(|v|v.as_str().map(String::from)).collect::<Vec<_>>();
    let classes=strings("classes");(prediction["class_count"].as_u64().map_or(classes.len(),|n|n as usize),classes,strings("feature_names"))
}
/// The manifest's `max_tokens` is a ceiling on every kind's token cap.
pub(crate) struct Capped<'a>{pub generator:&'a dyn Generate,pub model:&'a Model}
impl Generate for Capped<'_>{fn planned_device(&self)->Option<Device>{self.generator.planned_device()}fn stream(&self,prompt:&str,n_predict:u32,cancel:&AtomicBool,on_token:&mut dyn FnMut(&str)->bool)->Result<Device,String>{self.generator.stream(prompt,self.model.cap(n_predict),cancel,on_token)}}
/// Synchronous collector for the `/english` endpoints (chat streams turns through `explain::run` directly, see `chat::Context::start`).
pub fn answer(app:&Application,evidence:&Evidence,question:Option<&str>,prior:&[String],cancel:&AtomicBool)->Result<Value,String>{
    evidence.require_xai()?;let model=installed(&app.root)?;let mut sentences=Vec::new();
    let audit=explain::run(&Capped{generator:&app.llm,model:&model},&model.prompt_template,&evidence.packet,&evidence.rules,&evidence.points,question,prior,cancel,&mut |s|sentences.push(s))?;
    Ok(reply(&model.name,&evidence.points,&sentences,audit,app.llm.device()))
}
/// The device a turn's reply shows: null only when no model sentence was shown. A generator error after verified sentences
/// (deadline, early close, crash) returns no device, so the running child's device is used (`"unknown"` once it is gone).
pub(crate) fn shown_device(audit:&Value,running:Option<Device>)->Value{
    if audit["device"].is_null()&&audit["model_sentences"].as_u64().unwrap_or(0)>0{json!(running.map_or("unknown",Device::as_str))}else{audit["device"].clone()}
}
/// `citations` are the template points used; `llm_executed` and the device fields follow `shown_device`; `grounding` keeps the stream's own audit.
pub(crate) fn reply(model:&str,points:&[evidence::Point],sentences:&[explain::Sentence],audit:Value,running:Option<Device>)->Value{
    let citations=audit["template_ids"].as_array().into_iter().flatten().filter_map(|id|points.iter().find(|p|p.id==*id)).collect::<Vec<_>>();
    let device=shown_device(&audit,running);let mut warnings=vec!["Model-written sentences passed NetSec's evidence checks; template sentences are fixed NetSec text. A prediction does not confirm an attack or guarantee safety."];
    if device=="cpu"{warnings.push("The language model is running on CPU, so answers are slower.");}
    json!({"status":"completed","text":sentences.iter().map(|s|s.text.as_str()).collect::<Vec<_>>().join(" "),"sentences":sentences,"model":model,"provider":"Local llama.cpp b11564",
        "llm_executed":!device.is_null(),"inference_device":device,"device":device,"citations":citations,"grounding":audit,"warnings":warnings})
}
pub fn generate(app:&Application,job:&Job,model:&str,row:usize,explanation:Option<&Value>)->Result<Value,String>{
    let (_,prediction)=job.observation(model,row)?;let info=app.catalog.info(model).map_err(|e|e.to_string())?;
    generate_evidence(app,&prediction,&info.label,explanation,&job.cancelled,json!({"job_id":job.view().id,"input_id":job.input.id,"model_id":model,"row_index":row}),info.classes.len(),&info.classes,&info.feature_names)
}
#[allow(clippy::too_many_arguments)]
pub fn generate_evidence(app:&Application,prediction:&Value,label:&str,explanation:Option<&Value>,cancel:&AtomicBool,binding:Value,class_count:usize,classes:&[String],all_features:&[String])->Result<Value,String>{
    let evidence=Evidence::new(prediction,label,explanation,class_count,classes,all_features);
    let mut result=answer(app,&evidence,None,&[],cancel)?;
    if let Some(fields)=binding.as_object(){for(key,value)in fields{result[key]=value.clone();}}
    Ok(result)
}
/// Test data: the four real validation-row packets of the explainer pass-rate matrix (language/testdata/matrix_packets.json) with the real
/// 77-class catalog, built exactly as production builds them.
#[cfg(test)]pub(crate) fn matrix_evidence()->std::collections::BTreeMap<String,Evidence>{
    let dir=Path::new(env!("CARGO_MANIFEST_DIR"));
    let read=|p:PathBuf|serde_json::from_slice::<Value>(&std::fs::read(&p).unwrap_or_else(|e|panic!("{}: {e}",p.display()))).unwrap();
    let catalog=read(dir.join("../../../../artifacts/application/native_models_v1/catalog.json"));
    let strings=|key:&str|catalog[key].as_array().unwrap().iter().map(|v|v.as_str().unwrap().to_string()).collect::<Vec<_>>();
    let (classes,features)=(strings("classes"),strings("feature_names"));
    read(dir.join("src/language/testdata/matrix_packets.json")).as_object().unwrap().iter().filter(|(k,_)|!k.starts_with('_'))
        .map(|(k,v)|(k.clone(),Evidence::new(&v["prediction"],"XGBoost",Some(&v["explanation"]),classes.len(),&classes,&features))).collect()
}
/// Test data: one real matrix sample (benign, ddos, scan, unknown) with one of its six real offline XAI explanations
/// (language/testdata/matrix_methods.json) and the real 77-class catalog; `method` "none" builds the prediction without XAI.
#[cfg(test)]pub(crate) fn matrix_method_evidence(sample:&str,method:&str)->Evidence{
    let dir=Path::new(env!("CARGO_MANIFEST_DIR"));
    let read=|p:PathBuf|serde_json::from_slice::<Value>(&std::fs::read(&p).unwrap_or_else(|e|panic!("{}: {e}",p.display()))).unwrap();
    let catalog=read(dir.join("../../../../artifacts/application/native_models_v1/catalog.json"));
    let strings=|key:&str|catalog[key].as_array().unwrap().iter().map(|v|v.as_str().unwrap().to_string()).collect::<Vec<_>>();
    let (classes,features)=(strings("classes"),strings("feature_names"));
    let data=read(dir.join("src/language/testdata/matrix_methods.json"));let v=&data[sample];assert!(v.is_object(),"unknown sample {sample}");
    let explanation=(method!="none").then(||v["explanations"][method].clone());assert!(explanation.as_ref().is_none_or(|x|x.is_object()),"unknown method {method}");
    Evidence::new(&v["prediction"],"XGBoost",explanation.as_ref(),classes.len(),&classes,&features)
}
#[cfg(test)]mod basecopy;
#[cfg(test)]mod probes;
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn missing_model_is_reported_not_substituted(){let s=status(Path::new("does-not-exist"),None);assert_eq!(s["status"],"unavailable");assert!(s["model"].is_null());assert!(s["reason"].as_str().unwrap().contains("No local language model"));}
    #[test]fn manifest_file_cannot_escape_llm_folder(){for file in ["../x.gguf","a/b.gguf","x.bin"]{let m=Model{name:"m".into(),file:file.into(),sha256:String::new(),prompt_template:String::new(),license:String::new(),device:Device::Gpu,max_tokens:300};assert!(m.path(Path::new(".")).is_err(),"{file}");}}
    #[test]fn manifest_reads_license_device_and_token_ceiling(){
        let full:Model=serde_json::from_value(json!({"name":"m","file":"m.gguf","sha256":"s","bytes":1,"url":"https://x","prompt_template":"{system}{user}","license":"Apache-2.0","license_file":"L.txt","device":"cpu","max_tokens":120})).unwrap();
        assert_eq!((full.license.as_str(),full.device,full.max_tokens),("Apache-2.0",Device::Cpu,120));
        let minimal:Model=serde_json::from_value(json!({"name":"m","file":"m.gguf","sha256":"s","prompt_template":"{system}{user}"})).unwrap();
        assert_eq!((minimal.license.as_str(),minimal.device,minimal.max_tokens),("",Device::Gpu,300));
        assert!(serde_json::from_value::<Model>(json!({"name":"m","file":"m.gguf","sha256":"s","prompt_template":"","device":"tpu"})).is_err());
    }
    #[test]fn cpu_override_forces_cpu(){
        for (manifest,env,expected) in [(Device::Gpu,None,Device::Gpu),(Device::Gpu,Some("cpu"),Device::Cpu),(Device::Gpu,Some(" CPU "),Device::Cpu),(Device::Gpu,Some("gpu"),Device::Gpu),(Device::Gpu,Some("other"),Device::Gpu),(Device::Cpu,None,Device::Cpu),(Device::Cpu,Some("gpu"),Device::Cpu)]{assert_eq!(Device::requested(manifest,env),expected,"{manifest:?} {env:?}");}
        assert_eq!((Device::Gpu.as_str(),Device::Cpu.as_str()),("gpu","cpu"));
    }
    #[test]fn status_reports_verified_generation_and_running_device(){
        let s=status(Path::new("does-not-exist"),Some(Device::Cpu));
        assert_eq!((s["mode"].as_str(),s["device"].as_str(),s["language_device"].as_str(),s["cpu_threads"].as_u64(),s["detectors_cpu_only"].as_bool()),(Some("verified_generation"),Some("cpu"),Some("cpu"),Some(8),Some(true)));
        let s=status(Path::new("does-not-exist"),None);assert!(s["device"].is_null()&&s["requested_device"].is_null(),"{s}");
    }
    /// A benign prediction with a selected SHAP explanation (the explainer needs a selected XAI method).
    fn shap_evidence()->Evidence{Evidence::new(&json!({"label":"benign","score":0.8}),"X",Some(&json!({"method":"shap","units":"raw margin","features":[{"feature":"flow_bytes_total","value":3.0,"contribution":0.4}]})),2,
        &["benign".into(),"recon_scan".into()],&["flow_bytes_total".into()])}
    #[test]fn reply_labels_device_and_cites_only_templates_used(){
        let e=shap_evidence();
        let sentence=|text:&str,source|explain::Sentence{text:text.into(),source,part:"answer",violation:None};
        let r=reply("m",&e.points,&[sentence("A person should verify this result.","model")],json!({"device":"cpu","template_ids":[]}),None);
        assert_eq!((r["llm_executed"].as_bool(),r["device"].as_str(),r["inference_device"].as_str()),(Some(true),Some("cpu"),Some("cpu")));assert_eq!(r["citations"],json!([]));
        assert!(r["warnings"].to_string().contains("slower"),"{r}");assert_eq!(r["sentences"][0]["source"],"model");
        let r=reply("m",&e.points,&[sentence("t","template")],json!({"device":null,"template_ids":["S1","W1","X3"]}),None);
        assert_eq!(r["llm_executed"],false);assert_eq!(r["citations"].as_array().unwrap().iter().map(|p|p["id"].as_str().unwrap()).collect::<Vec<_>>(),["S1","W1"]);
        assert_eq!(live_schema(&json!({"class_count":3,"classes":["a","b"],"feature_names":["f"]})),(3,vec!["a".to_string(),"b".into()],vec!["f".to_string()]));
    }
    #[test]fn reply_labels_model_sentences_after_mid_stream_error(){
        // A deadline after one verified sentence: the stream returns no device, yet the sentence shown is model text.
        struct Cut;impl Generate for Cut{fn stream(&self,_:&str,_:u32,_:&AtomicBool,on_token:&mut dyn FnMut(&str)->bool)->Result<Device,String>{on_token("It shows the model's own reasoning. The");Err("answer deadline".into())}}
        let e=shap_evidence();let mut sentences=Vec::new();
        let audit=explain::run(&Cut,"{system}{user}",&e.packet,&e.rules,&e.points,Some("Why?"),&[],&AtomicBool::new(false),&mut |s|sentences.push(s)).unwrap();
        assert!(audit["device"].is_null()&&audit["model_sentences"]==1&&audit["first_violation"]=="deadline"&&sentences.len()>1,"{audit}");
        for (running,device) in [(Some(Device::Cpu),"cpu"),(Some(Device::Gpu),"gpu"),(None,"unknown")]{
            let r=reply("m",&e.points,&sentences,audit.clone(),running);
            assert_eq!((r["llm_executed"].as_bool(),r["device"].as_str(),r["inference_device"].as_str()),(Some(true),Some(device),Some(device)),"{r}");
            assert_eq!(r["sentences"][0]["source"],"model");let rest=&r["sentences"].as_array().unwrap()[1..];
            assert!(rest[..rest.len()-1].iter().all(|s|s["source"]=="template")&&rest.last().unwrap()["source"]=="netsec","{r}");
            assert_eq!(r["grounding"]["template_label"],"Template (model unavailable)");assert_eq!(r["warnings"].to_string().contains("slower"),device=="cpu","{r}");
        }
        // No model sentence was shown: not an LLM reply, whatever is running.
        let mut none=audit.clone();none["model_sentences"]=json!(0);let r=reply("m",&e.points,&sentences[1..],none,Some(Device::Gpu));
        assert_eq!((r["llm_executed"].as_bool(),r["device"].is_null(),r["inference_device"].is_null()),(Some(false),true,true),"{r}");
    }
    #[test]fn token_cap_never_exceeds_manifest_ceiling(){let m=Model{name:"m".into(),file:"m.gguf".into(),sha256:String::new(),prompt_template:String::new(),license:String::new(),device:Device::Gpu,max_tokens:200};assert_eq!((m.cap(120),m.cap(220),m.cap(300)),(120,200,200));}
}
