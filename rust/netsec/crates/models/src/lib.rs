//! CPU-only native inference of the completed, versioned research candidates.
#![forbid(unsafe_code)]
mod tensor;
mod preprocess;
mod neural;
mod forest;
mod hem;
pub use hem::HemDetails;

use std::{collections::BTreeMap,fs,path::{Path,PathBuf}};
use serde::{Deserialize,Serialize};
use serde_json::Value;
use tensor::{checked_bytes,local_path,Tensors,MAX_BLOB};
use netsec_xgboost_spike::Booster;

#[derive(Debug,thiserror::Error)]
pub enum ModelError {
    #[error("{0}")] Invalid(String),
    #[error("Model artifact I/O: {0}")] Io(#[from]std::io::Error),
    #[error("Model manifest JSON: {0}")] Json(#[from]serde_json::Error),
    #[error("Native XGBoost: {0}")] Xgboost(#[from]netsec_xgboost_spike::Error),
}
pub(crate) fn invalid(message:impl Into<String>)->ModelError{ModelError::Invalid(message.into())}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct ModelInfo {
    pub id:String,pub label:String,pub kind:String,pub feature_names:Vec<String>,pub classes:Vec<String>,
    pub artifact_sha256:String,pub supports_partial:bool,pub calibration:Value,
}
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct ModelOutput {
    pub predicted:usize,pub probabilities:Option<Vec<f64>>,pub raw_probabilities:Option<Vec<f64>>,
    pub confidence:Option<f64>,pub attack_probability:Option<f64>,pub confidence_kind:String,
}
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct TreeExplanation {
    pub contributions:Vec<f64>,pub bias:f64,pub margin:f64,pub units:String,pub method:String,pub reconstruction_error:f64,
}

#[derive(Clone)]
struct Entry { path:String,sha256:String,info:ModelInfo,auxiliary:bool,availability:ModelAvailability }
#[derive(Clone)]
pub struct ModelCatalog { root:PathBuf,entries:BTreeMap<String,Entry>,merge:Option<ClassMerge> }

/// Catalog-level output merge (artifacts/training/class_merge_20261101/spec.json): after calibration, one class's
/// probability is folded into another on the vector the app displays, and the decision is the argmax of the result
/// (lowest index on ties). Model weights are unchanged; a hard vote for the source class becomes the target class.
#[derive(Clone,Debug,PartialEq,Eq)]
pub struct ClassMerge { pub source:usize,pub target:usize }
impl ClassMerge {
    /// Reads `{"source_class":..,"target_class":..}`; an absent (null) field means no merge.
    pub fn parse(value:&Value,classes:&[String])->Result<Option<Self>,ModelError>{
        if value.is_null(){return Ok(None)}
        let index=|key:&str|value[key].as_str().and_then(|name|classes.iter().position(|c|c==name))
            .ok_or_else(||invalid(format!("Class merge {key} is not a catalog class")));
        let (source,target)=(index("source_class")?,index("target_class")?);
        if source==target{return Err(invalid("Class merge source and target are the same class"))}
        Ok(Some(Self{source,target}))
    }
    pub fn apply(&self,probabilities:&mut [f64]){
        probabilities[self.target]+=probabilities[self.source];probabilities[self.source]=0.0;
    }
    pub fn vote(&self,predicted:usize)->usize{if predicted==self.source{self.target}else{predicted}}
}

/// Metadata readiness is scoped to one model. Weight hashes are checked on load.
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct ModelAvailability { pub status:String,pub reason:Option<String> }

fn manifest(root:&Path,path:&str,sha:&str,info:&ModelInfo)->Result<Value,ModelError>{
    let path=local_path(root,path)?;
    let value:Value=serde_json::from_slice(&checked_bytes(&path,sha,16*1024*1024)?)?;
    if value["version"]!=1||value["id"]!=info.id||strings(&value["feature_names"])?!=info.feature_names
        ||strings(&value["classes"])?!=info.classes||value["benign_index"]!=0||value["kind"]!=info.kind{
        return Err(invalid("Model and catalog schemas differ"))
    }
    if !["random_forest","xgboost","kan","hem","tabm_ne","tabm_hier","tabm_hier_ne","tabpack","hybrid"].contains(&info.kind.as_str()){
        return Err(invalid("Unsupported model kind"))
    }
    let files=value["files"].as_array().filter(|v|v.len()<=4).ok_or_else(||invalid("Invalid model assets"))?;
    let folder=path.parent().ok_or_else(||invalid("Missing model folder"))?;
    let mut names=std::collections::BTreeSet::new();
    for file in files{
        let name=file["path"].as_str().ok_or_else(||invalid("Missing model asset path"))?;
        if !["weights.bin","model.ubj"].contains(&name)||!names.insert(name){return Err(invalid("Invalid or duplicate model asset"))}
        let actual=fs::metadata(local_path(folder,name)?)?.len();
        let expected=file["bytes"].as_u64().ok_or_else(||invalid("Missing model asset size"))?;
        if actual!=expected||actual>MAX_BLOB{return Err(invalid("Model asset size differs or exceeds bound"))}
        let hash=file["sha256"].as_str().ok_or_else(||invalid("Missing asset SHA256"))?;
        if hash.len()!=64||!hash.bytes().all(|b|b.is_ascii_hexdigit()){return Err(invalid("Invalid asset SHA256"))}
    }
    if info.kind!="hybrid"&&!names.contains("weights.bin"){return Err(invalid("Missing native model weights"))}
    if info.kind=="xgboost"&&!names.contains("model.ubj"){return Err(invalid("Missing XGBoost checkpoint"))}
    Ok(value)
}

fn dependencies(id:&str,manifests:&BTreeMap<String,Value>,entries:&BTreeMap<String,Entry>,path:&mut Vec<String>)->Result<(),String>{
    if path.iter().any(|p|p==id){return Err(format!("Hybrid dependency cycle: {} -> {id}",path.join(" -> ")))}
    if path.len()>=8{return Err("Hybrid dependency depth exceeds bound".into())}
    let entry=entries.get(id).ok_or_else(||format!("Unknown hybrid dependency {id}"))?;
    if entry.availability.status!="ready"{return Err(format!("Dependency {id} unavailable: {}",entry.availability.reason.as_deref().unwrap_or("metadata failed")))}
    let value=manifests.get(id).ok_or_else(||format!("Missing dependency metadata for {id}"))?;
    if entry.info.kind!="hybrid"{
        if entry.info.kind=="hem"{return Err("HEM hard votes cannot enter a probability blend".into())}
        return Ok(())
    }
    let branches=strings(&value["branches"]).map_err(|e|e.to_string())?;
    let cfg=&value["combiner"];
    if branches.is_empty()||branches.len()>8||strings(&cfg["names"]).map_err(|e|e.to_string())?!=branches
        ||cfg["n_classes"].as_u64()!=Some(entry.info.classes.len() as u64){return Err("Invalid hybrid branch or class schema".into())}
    let kind=cfg["kind"].as_str().ok_or("Missing hybrid combiner kind")?;
    if !["global","classwise"].contains(&kind){return Err("Unsupported hybrid combiner kind".into())}
    let weights=cfg["weights"].as_array().filter(|w|w.len()==branches.len()).ok_or("Hybrid weight count differs")?;
    let mut totals=vec![0.0;entry.info.classes.len()];
    for weight in weights{
        if kind=="classwise"&&weight.as_array().map_or(true,|v|v.len()!=totals.len()){return Err("Hybrid weight class dimensions differ".into())}
        for (c,total) in totals.iter_mut().enumerate(){
            let w=if kind=="global"{weight.as_f64()}else{weight.get(c).and_then(Value::as_f64)}
                .filter(|v|v.is_finite()&&*v>=0.0&&*v<=1.0+1e-9).ok_or("Invalid hybrid weight")?;
            *total+=w;
        }
    }
    if totals.iter().any(|v|(v-1.0).abs()>1e-6){return Err("Hybrid weights must sum to one across branches".into())}
    path.push(id.to_owned());
    for branch in &branches{dependencies(branch,manifests,entries,path)?;}
    path.pop();Ok(())
}

fn strings(value:&Value)->Result<Vec<String>,ModelError>{
    let values:Vec<String>=value.as_array().ok_or_else(||invalid("Expected schema array"))?.iter()
        .map(|s|s.as_str().filter(|s|!s.is_empty()).map(str::to_owned).ok_or_else(||invalid("Expected nonempty schema name"))).collect::<Result<_,_>>()?;
    let unique:std::collections::BTreeSet<_>=values.iter().collect();
    if unique.len()!=values.len(){return Err(invalid("Duplicate schema names"))}Ok(values)
}

impl ModelCatalog {
    pub fn load(export_root:&Path)->Result<Self,ModelError>{
        let root=export_root.canonicalize()?;
        let path=root.join("catalog.json");
        if fs::metadata(&path)?.len()>16*1024*1024{return Err(invalid("Catalog exceeds size bound"))}
        let catalog:Value=serde_json::from_slice(&fs::read(path)?)?;
        if catalog["version"]!=1{return Err(invalid("Unsupported native catalog version"))}
        let features=strings(&catalog["feature_names"])?;let classes=strings(&catalog["classes"])?;
        if features.is_empty()||features.len()>4096||classes.len()<2||classes.len()>512||catalog["benign_index"]!=0{return Err(invalid("Unsupported catalog schema"))}
        let merge=ClassMerge::parse(&catalog["class_merge"],&classes)?;
        let models=catalog["models"].as_array().filter(|v|!v.is_empty()&&v.len()<=32).ok_or_else(||invalid("Invalid catalog model count"))?;
        let mut entries=BTreeMap::new();let mut manifests=BTreeMap::new();
        for model in models{
            let id=model["id"].as_str().ok_or_else(||invalid("Missing model id"))?;
            let path=model["manifest"].as_str().ok_or_else(||invalid("Missing model manifest path"))?;
            let sha=model["sha256"].as_str().ok_or_else(||invalid("Missing model manifest SHA256"))?;
            let kind=model["kind"].as_str().ok_or_else(||invalid("Missing catalog model kind"))?;
            // Catalog metadata retains the visible entry even if its manifest is absent.
            let mut info=ModelInfo{id:id.into(),label:model["label"].as_str().unwrap_or(id).into(),kind:kind.into(),
                feature_names:features.clone(),classes:classes.clone(),artifact_sha256:sha.into(),
                supports_partial:false,calibration:Value::Null};
            let availability=match manifest(&root,path,sha,&info){
                Ok(manifest)=>{
                    info.label=manifest["label"].as_str().unwrap_or(&info.label).to_owned();
                    info.supports_partial=manifest["supports_partial"].as_bool().unwrap_or(false);
                    info.calibration=manifest["calibration"].clone();manifests.insert(id.to_owned(),manifest);
                    ModelAvailability{status:"ready".into(),reason:None}
                }
                Err(error)=>{let status=if matches!(&error,ModelError::Io(e) if e.kind()==std::io::ErrorKind::NotFound){"unavailable"}else{"incompatible"};
                    ModelAvailability{status:status.into(),reason:Some(error.to_string())}}
            };
            let entry=Entry{path:path.into(),sha256:sha.into(),info,auxiliary:model["auxiliary"].as_bool().unwrap_or(false),availability};
            if entries.insert(id.to_owned(),entry).is_some(){return Err(invalid("Duplicate model id"))}
        }
        let failures:Vec<_>=entries.iter().filter(|(_,entry)|entry.info.kind=="hybrid"&&entry.availability.status=="ready")
            .filter_map(|(id,_)|dependencies(id,&manifests,&entries,&mut Vec::new()).err().map(|error|(id.clone(),error))).collect();
        for (id,error) in failures{if let Some(entry)=entries.get_mut(&id){entry.availability=ModelAvailability{status:"unavailable".into(),reason:Some(error)}}}
        Ok(Self{root,entries,merge})
    }
    pub fn summaries(&self)->Vec<ModelInfo>{self.entries.values().filter(|entry|!entry.auxiliary).map(|entry|entry.info.clone()).collect()}
    pub fn info(&self,id:&str)->Result<ModelInfo,ModelError>{self.entries.get(id).map(|entry|entry.info.clone()).ok_or_else(||invalid(format!("Unknown model {id}")))}
    pub fn availability(&self,id:&str)->Result<ModelAvailability,ModelError>{self.entries.get(id).map(|entry|entry.availability.clone()).ok_or_else(||invalid(format!("Unknown model {id}")))}
    pub fn statuses(&self)->BTreeMap<String,ModelAvailability>{self.entries.iter().map(|(id,entry)|(id.clone(),entry.availability.clone())).collect()}
    pub fn root(&self)->&Path{&self.root}
    pub fn class_merge(&self)->Option<&ClassMerge>{self.merge.as_ref()}
    pub fn load_model(&self,id:&str)->Result<LoadedModel,ModelError>{
        let entry=self.entries.get(id).ok_or_else(||invalid(format!("Unknown model {id}")))?;
        if entry.availability.status!="ready"{return Err(invalid(format!("Model {id} unavailable: {}",entry.availability.reason.as_deref().unwrap_or("metadata validation failed"))))}
        let path=local_path(&self.root,&entry.path)?;
        let bytes=checked_bytes(&path,&entry.sha256,16*1024*1024)?;
        let manifest:Value=serde_json::from_slice(&bytes)?;
        let folder=path.parent().ok_or_else(||invalid("Missing bundle folder"))?;
        let files=manifest["files"].as_array().ok_or_else(||invalid("Missing bundle assets"))?;
        let mut blob=Vec::new();let mut booster=None;
        for file in files{
            let filename=file["path"].as_str().ok_or_else(||invalid("Missing asset path"))?;
            let hash=file["sha256"].as_str().ok_or_else(||invalid("Missing asset SHA256"))?;
            let asset=local_path(folder,filename)?;
            if filename=="weights.bin"{blob=checked_bytes(&asset,hash,MAX_BLOB)?}
            else if filename=="model.ubj"{booster=Some(Booster::load_checkpoint(&asset,Some(hash))?)}
            else{return Err(invalid("Unsupported model asset"))}
        }
        let tensors=Tensors::load(&manifest,&blob)?;
        drop(blob);
        // Retain each hybrid branch once, including for repeated XAI perturbations.
        let mut branches=BTreeMap::new();
        if entry.info.kind=="hybrid"{
            for branch in strings(&manifest["branches"])?{branches.insert(branch.clone(),self.load_model(&branch)?);}
        }
        Ok(LoadedModel{info:entry.info.clone(),manifest,tensors,booster,catalog:self.clone(),branches})
    }
    pub fn predict_many(&self,ids:&[String],row:&[f64])->Vec<(String,Result<ModelOutput,ModelError>)>{
        let mut cache=PredictionCache::default();
        ids.iter().map(|id|(id.clone(),self.load_model(id).and_then(|model|model.predict_cached(row,&mut cache)))).collect()
    }
}

#[derive(Clone)]
enum Raw { Probabilities(Vec<f64>),Vote{predicted:usize,confidence:f64} }
/// A cache belongs to ONE input row. Reusing it for another row clears it automatically.
#[derive(Default)]
pub struct PredictionCache { values:BTreeMap<String,Raw>,row_bits:Vec<u64> }
impl PredictionCache {
    pub fn clear(&mut self){self.values.clear();self.row_bits.clear()}
    fn bind(&mut self,row:&[f64]){
        let bits:Vec<u64>=row.iter().map(|v|v.to_bits()).collect();
        if bits!=self.row_bits{self.values.clear();self.row_bits=bits;}
    }
}

pub struct LoadedModel { info:ModelInfo,manifest:Value,tensors:Tensors,booster:Option<Booster>,catalog:ModelCatalog,branches:BTreeMap<String,LoadedModel> }

pub fn argmax(values:&[f64])->usize{
    let mut result=0;
    for i in 1..values.len(){if values[i]>values[result]{result=i}}
    result
}

/// Calibrated class probabilities and binary attack score. Legacy bundles carry only `temperature`
/// (+ `binary_platt`); cascade v3 bundles add `kind`, the smoothing weight `alpha` and `log_floor`,
/// i.e. `z = ln(max((1-alpha)p + alpha/K, floor))`, then `softmax(z/T)` or `softmax(scale*z + bias)`.
pub fn calibrated(raw:&[f64],calibration:&Value)->Result<(Vec<f64>,f64),ModelError>{
    let alpha=calibration["alpha"].as_f64().unwrap_or(0.0);
    let floor=calibration["log_floor"].as_f64().unwrap_or(1e-12);
    if !(0.0..1.0).contains(&alpha)||!(floor>0.0&&floor<1.0){return Err(invalid("Invalid calibration smoothing"))}
    let classes=raw.len() as f64;
    let z=raw.iter().map(|v|((1.0-alpha)*v+alpha/classes).clamp(floor,1.0).ln());
    let mut p:Vec<f64>=match calibration["kind"].as_str().unwrap_or("temperature"){
        "temperature"|"identity"=>{
            let temperature=calibration["temperature"].as_f64().unwrap_or(1.0);
            if !temperature.is_finite()||temperature<=0.0{return Err(invalid("Invalid calibration temperature"))}
            z.map(|v|v/temperature).collect()
        }
        "vector"=>{
            let scale=calibration["scale"].as_f64().filter(|v|v.is_finite()&&*v>0.0).ok_or_else(||invalid("Invalid vector calibration scale"))?;
            let bias=calibration["bias"].as_array().filter(|b|b.len()==raw.len()).ok_or_else(||invalid("Vector calibration bias dimension differs"))?;
            z.zip(bias).map(|(v,b)|b.as_f64().filter(|b|b.is_finite()).map(|b|scale*v+b).ok_or_else(||invalid("Invalid vector calibration bias"))).collect::<Result<_,_>>()?
        }
        _=>return Err(invalid("Unsupported calibration kind"))
    };
    let largest=p.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    for value in &mut p{*value=(*value-largest).exp()}
    let total=p.iter().sum::<f64>();for value in &mut p{*value/=total}
    let attack=(1.0-raw[0]).clamp(1e-6,1.0-1e-6);
    let slope=calibration["binary_platt"]["slope"].as_f64().unwrap_or(1.0);
    let intercept=calibration["binary_platt"]["intercept"].as_f64().unwrap_or(0.0);
    let attack=neural::sigmoid(slope*(attack/(1.0-attack)).ln()+intercept);
    Ok((p,attack))
}

/// Predicted class with its calibrated outputs. Temperature never moves the argmax, so legacy and
/// temperature bundles keep the raw decision; a vector calibrator's bias is a decision rule by design.
pub fn decision(raw:&[f64],calibration:&Value)->Result<(usize,Vec<f64>,f64),ModelError>{
    let (probabilities,attack)=calibrated(raw,calibration)?;
    let predicted=if calibration["kind"]=="vector"{argmax(&probabilities)}else{argmax(raw)};
    Ok((predicted,probabilities,attack))
}

impl LoadedModel {
    pub fn info(&self)->&ModelInfo{&self.info}
    pub fn predict(&self,row:&[f64])->Result<ModelOutput,ModelError>{self.predict_cached(row,&mut PredictionCache::default())}
    pub fn predict_cached(&self,row:&[f64],cache:&mut PredictionCache)->Result<ModelOutput,ModelError>{
        if row.len()!=self.info.feature_names.len()||row.iter().any(|v|v.is_infinite()||(v.is_finite()&&v.abs()>f32::MAX as f64)){
            return Err(invalid("Expected ordered finite-or-missing model feature row"))
        }
        cache.bind(row);
        match self.raw(row,cache)? {
            Raw::Vote{predicted,confidence}=>Ok(ModelOutput{predicted:self.catalog.merge.as_ref().map_or(predicted,|m|m.vote(predicted)),probabilities:None,raw_probabilities:None,
                confidence:Some(confidence),attack_probability:None,confidence_kind:"vote_correctness".into()}),
            Raw::Probabilities(raw)=>{
                if raw.len()!=self.info.classes.len()||raw.iter().any(|v|!v.is_finite()||*v<0.0)|| (raw.iter().sum::<f64>()-1.0).abs()>1e-4{
                    return Err(invalid("Invalid model probability distribution"))
                }
                let (mut predicted,mut probabilities,attack)=decision(&raw,&self.manifest["calibration"])?;
                if let Some(merge)=&self.catalog.merge{merge.apply(&mut probabilities);predicted=argmax(&probabilities);}
                Ok(ModelOutput{predicted,confidence:Some(probabilities[predicted]),attack_probability:Some(attack),
                    probabilities:Some(probabilities),raw_probabilities:Some(raw),confidence_kind:"class_probability".into()})
            }
        }
    }
    fn raw(&self,row:&[f64],cache:&mut PredictionCache)->Result<Raw,ModelError>{
        let key=format!("{}:{}",self.info.id,self.info.artifact_sha256);
        if let Some(output)=cache.values.get(&key){return Ok(output.clone())}
        let output=match self.info.kind.as_str(){
            "hybrid"=>{
                let branches=strings(&self.manifest["branches"])?;
                if branches.is_empty()||branches.len()>8||strings(&self.manifest["combiner"]["names"])?!=branches{return Err(invalid("Invalid hybrid branch schema"))}
                let mut probabilities=Vec::new();
                for branch in branches{
                    let branch_key=format!("{branch}:{}",self.catalog.info(&branch)?.artifact_sha256);
                    let raw=if let Some(raw)=cache.values.get(&branch_key){raw.clone()}else{
                        self.branches.get(&branch).ok_or_else(||invalid("Missing loaded hybrid dependency"))?.raw(row,cache)?
                    };
                    match raw{Raw::Probabilities(p)=>probabilities.push(p),_=>return Err(invalid("Hard-vote branch cannot enter probability blend"))}
                }
                let cfg=&self.manifest["combiner"];let weights=cfg["weights"].as_array().filter(|w|w.len()==probabilities.len()).ok_or_else(||invalid("Invalid hybrid weights"))?;
                let mut result=vec![0.0f64;self.info.classes.len()];
                for (m,p) in probabilities.iter().enumerate(){
                    if p.len()!=result.len(){return Err(invalid("Hybrid class dimensions differ"))}
                    for c in 0..result.len(){
                        let w=if cfg["kind"]=="global"{weights[m].as_f64()}else{weights[m].get(c).and_then(Value::as_f64)}
                            .filter(|v|v.is_finite()&&*v>=0.0).ok_or_else(||invalid("Invalid hybrid weight"))?;
                        result[c]+=w*p[c];
                    }
                }
                let total=result.iter().sum::<f64>();if total<=0.0{return Err(invalid("Zero hybrid probability mass"))}
                for p in &mut result{*p=((*p/total) as f32) as f64}
                Raw::Probabilities(result)
            }
            "random_forest"=>Raw::Probabilities(forest::predict(&self.manifest,&self.tensors,row)?),
            "xgboost"=>{
                let input:Vec<Option<f32>>=row.iter().map(|v|if v.is_nan(){None}else{Some(*v as f32)}).collect();
                Raw::Probabilities(self.booster.as_ref().ok_or_else(||invalid("Missing XGBoost booster"))?.predict(&input)?.iter().map(|p|*p as f64).collect())
            }
            family=>{
                let x=preprocess::transform(row,&self.manifest["preprocessor"],&self.tensors)?;
                match family{
                    "kan"=>Raw::Probabilities(neural::kan(&self.manifest,&self.tensors,x)?),
                    "hem"=>{let details=hem::predict(&self.manifest,&self.tensors,x)?;Raw::Vote{predicted:details.predicted,confidence:details.confidence}},
                    "tabpack"=>Raw::Probabilities(neural::tabpack(&self.manifest,&self.tensors,x)?),
                    "tabm_ne"|"tabm_hier"|"tabm_hier_ne"=>Raw::Probabilities(neural::tabm(&self.manifest,&self.tensors,x)?),
                    _=>return Err(invalid("Unsupported model family"))
                }
            }
        };
        cache.values.insert(key,output.clone());Ok(output)
    }
    pub fn tree_explanation(&self,row:&[f64],target:usize)->Result<Option<TreeExplanation>,ModelError>{
        if target>=self.info.classes.len(){return Err(invalid("Explanation target outside class schema"))}
        if let Some(booster)=&self.booster{
            let values:Vec<Option<f32>>=row.iter().map(|v|if v.is_nan(){None}else{Some(*v as f32)}).collect();
            let margins=booster.margins(&values)?;let all=booster.contributions(&values)?;
            let selected=all.get(target).ok_or_else(||invalid("Missing target SHAP vector"))?;
            let bias=*selected.last().ok_or_else(||invalid("Empty SHAP vector"))? as f64;
            let contributions:Vec<f64>=selected[..selected.len()-1].iter().map(|v|*v as f64).collect();
            let margin=margins[target] as f64;let reconstruction_error=(bias+contributions.iter().sum::<f64>()-margin).abs();
            Ok(Some(TreeExplanation{contributions,bias,margin,units:"raw_class_margin".into(),method:"exact_tree_shap".into(),reconstruction_error}))
        }else{Ok(None)}
    }
    /// Exact submodel evidence for numerical audits and faithful HEM explanations.
    pub fn hem_details(&self,row:&[f64])->Result<Option<HemDetails>,ModelError>{
        if self.info.kind!="hem"{return Ok(None)}
        if row.len()!=self.info.feature_names.len()||row.iter().any(|v|v.is_infinite()){return Err(invalid("Invalid HEM input row"))}
        let x=preprocess::transform(row,&self.manifest["preprocessor"],&self.tensors)?;
        Ok(Some(hem::predict(&self.manifest,&self.tensors,x)?))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    fn close(a:&[f64],b:&[f64])->bool{a.len()==b.len()&&a.iter().zip(b).all(|(x,y)|(x-y).abs()<1e-12)}
    #[test]
    fn legacy_temperature_calibration_is_unchanged() {
        let raw=[0.7,0.2,0.1,0.0];let t=0.85;
        let mut expected:Vec<f64>=raw.iter().map(|v:&f64|(v.clamp(1e-12,1.0).ln()/t).exp()).collect();
        let total:f64=expected.iter().sum();for v in &mut expected{*v/=total}
        let (predicted,p,_)=crate::decision(&raw,&json!({"temperature":t})).unwrap();
        assert_eq!(predicted,0);assert!(close(&p,&expected));
    }
    #[test]
    fn smoothed_temperature_matches_python_csv_v3_apply() {
        let (predicted,p,_)=crate::decision(&[0.7,0.2,0.1,0.0],&json!({"kind":"temperature","temperature":0.85,"alpha":0.01,"log_floor":1e-7})).unwrap();
        assert_eq!(predicted,0);
        assert!(close(&p,&[0.7481042221075982,0.17316215400904042,0.07773765025247581,0.0009959736308856324]));
    }
    #[test]
    fn vector_calibration_decides_by_calibrated_argmax() {
        let cfg=json!({"kind":"vector","scale":1.5,"alpha":0.01,"log_floor":1e-7,"bias":[0.0,2.5,0.0,-2.5]});
        let (predicted,p,_)=crate::decision(&[0.7,0.2,0.1,0.0],&cfg).unwrap();
        assert_eq!(predicted,1);
        assert!(close(&p,&[0.33997149791756387,0.6410687064440338,0.018953781538811998,6.014099590402529e-06]));
    }
    #[test]
    fn invalid_calibration_is_rejected() {
        for cfg in [json!({"kind":"vector","scale":1.0,"bias":[0.0]}),json!({"kind":"vector","scale":-1.0,"bias":[0.0,0.0]}),
                    json!({"kind":"temperature","temperature":1.0,"alpha":1.0}),json!({"kind":"unknown"})]{
            assert!(crate::decision(&[0.6,0.4],&cfg).is_err(),"{cfg}");
        }
    }
    #[test]
    fn repeated_quantiles_use_mean_of_both_interpolation_directions() {
        let result = crate::preprocess::quantile_uniform(1.0, &[0.0, 1.0, 1.0, 2.0], &[0.0, 0.25, 0.75, 1.0]);
        assert!((result - 0.5).abs() < 1e-14);
    }
    #[test]
    fn spline_partition_and_boundary_follow_half_open_intervals() {
        let b = crate::neural::bspline(0.25, &[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0], 2);
        assert_eq!(b.len(), 4);
        assert!((b.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
    #[test]
    fn normal_quantiles_are_symmetric_and_accurate() {
        let q = crate::preprocess::normal_quantile(0.975);
        assert!((q - 1.959963984540054).abs() < 5e-9);
        assert!((q + crate::preprocess::normal_quantile(0.025)).abs() < 1e-12);
    }
    #[test]
    fn pca_preserves_float32_accumulation_before_centering() {
        use crate::tensor::{Data,Tensor,Tensors};
        let mut values=std::collections::BTreeMap::new();
        values.insert("components".into(),Tensor{shape:vec![1,3],data:Data::F32(vec![1e8,1.0,-1e8])});
        values.insert("mean".into(),Tensor{shape:vec![3],data:Data::F32(vec![0.0;3])});
        let spec=serde_json::json!({"kind":"pipeline","steps":[{"kind":"pca","components":"components","mean":"mean"}]});
        let result=crate::preprocess::transform(&[1.0,1.0,1.0],&spec,&Tensors{values}).unwrap();
        assert_eq!(result,vec![0.0]);
    }
}
