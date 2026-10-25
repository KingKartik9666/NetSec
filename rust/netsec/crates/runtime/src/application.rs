use netsec_core::jobs::JobStore;
use netsec_explain::{FeatureGroup,Reference};
use netsec_models::ModelCatalog;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,fs,path::{Path,PathBuf},sync::{Arc,Mutex,atomic::AtomicBool}};
use tokio::sync::{Notify,Semaphore};

pub struct Application {
    pub root:PathBuf, pub catalog:Arc<ModelCatalog>, pub jobs:Arc<JobStore>, pub samples:Value,
    pub reference:Reference, pub explanations:Mutex<BTreeMap<String,Value>>,
    pub inference_slots:Arc<Semaphore>, pub explanation_slots:Arc<Semaphore>,pub language_slots:Arc<Semaphore>,
    pub stopping:AtomicBool, pub shutdown:Notify,pub live:crate::live::Live,
    pub window:Mutex<Value>,
}
impl Application {
    pub fn load(root:PathBuf)->Result<Arc<Self>,String>{
        let root=root.canonicalize().map_err(|e|e.to_string())?;
        crate::assets::verify(&root)?;
        let exported=root.join("artifacts/application/native_models_v1");
        let catalog=Arc::new(ModelCatalog::load(&exported).map_err(|e|e.to_string())?);
        let samples=read_json(&exported.join("samples.json"),4*1024*1024)?;
        let reference_path=exported.join("reference.json");let raw=read_json(&reference_path,4*1024*1024)?;
        let names:Vec<String>=serde_json::from_value(raw["feature_names"].clone()).map_err(|e|e.to_string())?;
        if raw["partition"]!="train"{return Err("Explanation reference must be an approved training reference".into());}
        let entries=raw["rows"].as_array().ok_or("Reference observations missing")?;
        let rows=entries.iter().map(|v|nullable_row(&v["values"])).collect::<Result<Vec<_>,_>>()?;
        let labels=entries.iter().map(|v|v["label_index"].as_u64().map(|v|v as usize).ok_or("Reference class missing".into())).collect::<Result<Vec<_>,String>>()?;
        let groups=raw["feature_groups"].as_array().ok_or("Reference groups missing")?.iter().enumerate().map(|(i,g)|{
            let indices:Vec<usize>=serde_json::from_value(g.clone()).map_err(|e|e.to_string())?;
            let first=*indices.first().ok_or("Empty reference group")?;
            let name=names.get(first).ok_or("Reference group outside schema")?.clone();
            Ok(FeatureGroup{name:format!("{name} / group {}",i+1),kind:if indices.len()==1&&!name.contains("__"){"numeric"}else{"linked"}.into(),indices})
        }).collect::<Result<Vec<_>,String>>()?;
        let info=catalog.summaries().first().cloned().ok_or("No visible models")?;
        if names!=info.feature_names||samples["feature_names"]!=raw["feature_names"]||samples["partition"]!="validation" {return Err("Sample/reference and model schemas differ".into());}
        let reference=Reference{feature_names:names,rows,labels,groups,sha256:hash_file(&reference_path)?};
        Ok(Arc::new(Self{root:root.clone(),catalog,jobs:Arc::new(JobStore::new(16)),samples,reference,
            explanations:Mutex::new(BTreeMap::new()),inference_slots:Arc::new(Semaphore::new(1)),
            explanation_slots:Arc::new(Semaphore::new(1)),language_slots:Arc::new(Semaphore::new(1)),
            stopping:AtomicBool::new(false),shutdown:Notify::new(),live:crate::live::Live::with_root(root.clone()),
            window:Mutex::new(json!({"desktop":std::env::var_os("NETSEC_DESKTOP").is_some(),"maximized":false,"exitRequested":false,"closing":false}))}))
    }
    pub fn catalog_view(&self)->Value {
        let models=self.catalog.summaries();let schema=models.first().map(|v|json!({"feature_names":v.feature_names,"classes":v.classes})).unwrap_or(Value::Null);
        json!({"models":models.iter().map(|m|{let availability=self.catalog.availability(&m.id).expect("Catalog summary has an entry");json!({"id":m.id,"label":m.label,"kind":m.kind,"status":availability.status,"reason":availability.reason,
            "feature_count":m.feature_names.len(),"class_count":m.classes.len(),"artifact_sha256":m.artifact_sha256,
            "supports_partial":m.supports_partial,"live_compatible":false,"capabilities":{"prediction":availability.status=="ready",
            "explanations":["shap","lime","anchors","counterfactual","permutation","ale"],"english":true},
            "limitations":["Exploratory source-validation results; live inputs are incompatible with this full schema","Calibration can regress; inspect method-specific evidence"]})}).collect::<Vec<_>>(),
            "schema":schema,"runtime":{"backend":"rust","cpu_only":true,"offline":true,"python_inference":false},
            "language":crate::language::status(&self.root)})
    }
    pub fn samples_view(&self)->Value {
        json!({"samples":self.samples["rows"].as_array().into_iter().flatten().map(|v|json!({"id":v["id"],
            "display_name":format!("{} · observation {}",display_label(v["label"].as_str().unwrap_or("unknown")),v["row_index"]),
            "label":v["label"],"source_row_id":v["id"],"origin":"validation","row":v["values"]})).collect::<Vec<_>>()})
    }
}
pub fn nullable_row(value:&Value)->Result<Vec<f64>,String>{
    value.as_array().ok_or("Expected ordered observation values")?.iter().map(|v|if v.is_null(){Ok(f64::NAN)}else{
        v.as_f64().filter(|v|v.is_finite()).ok_or("Observation values must be finite numbers or null".into())
    }).collect()
}
pub fn read_json(path:&Path,limit:u64)->Result<Value,String>{
    if fs::metadata(path).map_err(|e|e.to_string())?.len()>limit{return Err("JSON artifact exceeds its size bound".into());}
    serde_json::from_slice(&fs::read(path).map_err(|e|e.to_string())?).map_err(|e|e.to_string())
}
pub fn hash_file(path:&Path)->Result<String,String>{
    use std::io::Read;
    let mut file=fs::File::open(path).map_err(|e|e.to_string())?;let mut hash=Sha256::new();let mut buffer=[0;65536];
    loop{let n=file.read(&mut buffer).map_err(|e|e.to_string())?;if n==0{break;}hash.update(&buffer[..n]);}
    Ok(format!("{:x}",hash.finalize()))
}
pub fn display_label(label:&str)->String{
    if label=="benign"{"Normal network activity".into()}else{label.replace('_'," ").replace('-'," ")}
}
