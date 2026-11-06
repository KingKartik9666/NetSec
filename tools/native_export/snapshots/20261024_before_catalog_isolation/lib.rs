//! CPU-only native inference of the completed, versioned research candidates.
#![forbid(unsafe_code)]
mod tensor;
mod preprocess;
mod neural;
mod forest;
mod hem;

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
struct Entry { path:String,sha256:String,info:ModelInfo,auxiliary:bool }
#[derive(Clone)]
pub struct ModelCatalog { root:PathBuf,entries:BTreeMap<String,Entry> }

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
        let models=catalog["models"].as_array().filter(|v|!v.is_empty()&&v.len()<=32).ok_or_else(||invalid("Invalid catalog model count"))?;
        let mut entries=BTreeMap::new();
        for model in models{
            let id=model["id"].as_str().ok_or_else(||invalid("Missing model id"))?;
            let path=model["manifest"].as_str().ok_or_else(||invalid("Missing model manifest path"))?;
            let sha=model["sha256"].as_str().ok_or_else(||invalid("Missing model manifest SHA256"))?;
            let bytes=checked_bytes(&local_path(&root,path)?,sha,16*1024*1024)?;
            let manifest:Value=serde_json::from_slice(&bytes)?;
            if manifest["version"]!=1||manifest["id"]!=id||strings(&manifest["feature_names"])?!=features
                ||strings(&manifest["classes"])?!=classes||manifest["benign_index"]!=0{return Err(invalid("Model and catalog schemas differ"))}
            let kind=manifest["kind"].as_str().ok_or_else(||invalid("Missing model kind"))?;
            if !["random_forest","xgboost","kan","hem","tabm_ne","tabm_hier","tabm_hier_ne","tabpack","hybrid"].contains(&kind){return Err(invalid("Unsupported model kind"))}
            let info=ModelInfo{id:id.into(),label:manifest["label"].as_str().unwrap_or(id).into(),kind:kind.into(),
                feature_names:features.clone(),classes:classes.clone(),artifact_sha256:sha.into(),
                supports_partial:manifest["supports_partial"].as_bool().unwrap_or(false),calibration:manifest["calibration"].clone()};
            let entry=Entry{path:path.into(),sha256:sha.into(),info,auxiliary:model["auxiliary"].as_bool().unwrap_or(false)};
            if entries.insert(id.to_owned(),entry).is_some(){return Err(invalid("Duplicate model id"))}
        }
        Ok(Self{root,entries})
    }
    pub fn summaries(&self)->Vec<ModelInfo>{self.entries.values().filter(|entry|!entry.auxiliary).map(|entry|entry.info.clone()).collect()}
    pub fn info(&self,id:&str)->Result<ModelInfo,ModelError>{self.entries.get(id).map(|entry|entry.info.clone()).ok_or_else(||invalid(format!("Unknown model {id}")))}
    pub fn root(&self)->&Path{&self.root}
    pub fn load_model(&self,id:&str)->Result<LoadedModel,ModelError>{
        let entry=self.entries.get(id).ok_or_else(||invalid(format!("Unknown model {id}")))?;
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
        Ok(LoadedModel{info:entry.info.clone(),manifest,tensors,booster,catalog:self.clone()})
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

pub struct LoadedModel { info:ModelInfo,manifest:Value,tensors:Tensors,booster:Option<Booster>,catalog:ModelCatalog }

pub fn argmax(values:&[f64])->usize{
    let mut result=0;
    for i in 1..values.len(){if values[i]>values[result]{result=i}}
    result
}

pub fn calibrated(raw:&[f64],calibration:&Value)->Result<(Vec<f64>,f64),ModelError>{
    let temperature=calibration["temperature"].as_f64().unwrap_or(1.0);
    if !temperature.is_finite()||temperature<=0.0{return Err(invalid("Invalid calibration temperature"))}
    let mut p:Vec<f64>=raw.iter().map(|v|v.clamp(1e-12,1.0).ln()/temperature).collect();
    let largest=p.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    for value in &mut p{*value=(*value-largest).exp()}
    let total=p.iter().sum::<f64>();for value in &mut p{*value/=total}
    let attack=(1.0-raw[0]).clamp(1e-6,1.0-1e-6);
    let slope=calibration["binary_platt"]["slope"].as_f64().unwrap_or(1.0);
    let intercept=calibration["binary_platt"]["intercept"].as_f64().unwrap_or(0.0);
    let attack=neural::sigmoid(slope*(attack/(1.0-attack)).ln()+intercept);
    Ok((p,attack))
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
            Raw::Vote{predicted,confidence}=>Ok(ModelOutput{predicted,probabilities:None,raw_probabilities:None,
                confidence:Some(confidence),attack_probability:None,confidence_kind:"vote_correctness".into()}),
            Raw::Probabilities(raw)=>{
                if raw.len()!=self.info.classes.len()||raw.iter().any(|v|!v.is_finite()||*v<0.0)|| (raw.iter().sum::<f64>()-1.0).abs()>1e-4{
                    return Err(invalid("Invalid model probability distribution"))
                }
                let predicted=argmax(&raw);let (probabilities,attack)=calibrated(&raw,&self.manifest["calibration"])?;
                Ok(ModelOutput{predicted,confidence:Some(probabilities[predicted]),attack_probability:Some(attack),
                    probabilities:Some(probabilities),raw_probabilities:Some(raw),confidence_kind:"class_probability".into()})
            }
        }
    }
    fn raw(&self,row:&[f64],cache:&mut PredictionCache)->Result<Raw,ModelError>{
        if let Some(output)=cache.values.get(&self.info.id){return Ok(output.clone())}
        let output=match self.info.kind.as_str(){
            "hybrid"=>{
                let branches=strings(&self.manifest["branches"])?;
                if branches.is_empty()||branches.len()>8||strings(&self.manifest["combiner"]["names"])?!=branches{return Err(invalid("Invalid hybrid branch schema"))}
                let mut probabilities=Vec::new();
                for branch in branches{
                    let raw=if let Some(raw)=cache.values.get(&branch){raw.clone()}else{self.catalog.load_model(&branch)?.raw(row,cache)?};
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
                    "hem"=>{let (predicted,confidence)=hem::predict(&self.manifest,&self.tensors,x)?;Raw::Vote{predicted,confidence}},
                    "tabpack"=>Raw::Probabilities(neural::tabpack(&self.manifest,&self.tensors,x)?),
                    "tabm_ne"|"tabm_hier"|"tabm_hier_ne"=>Raw::Probabilities(neural::tabm(&self.manifest,&self.tensors,x)?),
                    _=>return Err(invalid("Unsupported model family"))
                }
            }
        };
        cache.values.insert(self.info.id.clone(),output.clone());Ok(output)
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
}

#[cfg(test)]
mod tests {
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
}
