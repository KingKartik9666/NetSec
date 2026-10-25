//! Bounded, model-independent explanations on an approved training reference.
//! Methods are native grouped approximations; no causal or detector claims.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};

mod constraints;
mod global;
mod local;
mod math;
mod rules;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureGroup {
    pub name: String,
    pub indices: Vec<usize>,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reference {
    pub feature_names: Vec<String>,
    pub rows: Vec<Vec<f64>>,
    pub labels: Vec<usize>,
    pub groups: Vec<FeatureGroup>,
    pub sha256: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Method { Shap, Lime, Anchors, Counterfactual, Permutation, Ale }

impl Method {
    pub fn parse(id: &str) -> Result<Self, String> {
        match id { "shap" => Ok(Self::Shap), "lime" => Ok(Self::Lime),
            "anchors" => Ok(Self::Anchors), "counterfactual" => Ok(Self::Counterfactual),
            "permutation" => Ok(Self::Permutation), "ale" => Ok(Self::Ale),
            _ => Err(format!("Unknown explanation method: {id}")) }
    }

    pub fn id(self) -> &'static str {
        match self { Self::Shap=>"shap",Self::Lime=>"lime",Self::Anchors=>"anchors",
            Self::Counterfactual=>"counterfactual",Self::Permutation=>"permutation",Self::Ale=>"ale" }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub budget: usize,
    pub seed: u64,
    pub target_index: usize,
    pub target_label: String,
    pub units: String,
}

impl Default for Settings {
    fn default() -> Self { Self { budget: 64, seed: 42, target_index: 0,
        target_label: "target".into(), units: "probability".into() } }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Evaluation {
    /// The same chosen target's score for every evaluated row.
    pub score: f64,
    /// The model's actual decision, independently of the chosen score.
    pub class_index: usize,
}

/// Primary API: one genuine inference supplies both the fixed target score and
/// actual model decision. The reference's provenance/hash must be verified by
/// its loader; this crate cannot authenticate a caller-provided digest string.
pub fn explain_prediction(method: Method, row: &[f64], reference: &Reference,
    settings: &Settings, predict: &dyn Fn(&[f64])->Result<Evaluation,String>,
    cancel: &AtomicBool) -> Result<Value,String> {
    explain_inner(method,row,reference,settings,predict,cancel,"actual_model_decision")
}

/// Compatibility API for scalar scores. Anchors and counterfactuals concern a
/// declared score >= 0.5 predicate here, never a verified multiclass decision.
pub fn explain(method: Method, row: &[f64], reference: &Reference,
    settings: &Settings, score: &dyn Fn(&[f64])->Result<f64,String>,
    cancel: &AtomicBool) -> Result<Value,String> {
    let predict=|x:&[f64]|score(x).map(|score|Evaluation{score,class_index:usize::from(score>=0.5)});
    explain_inner(method,row,reference,settings,&predict,cancel,"scalar_threshold_compatibility")
}

fn explain_inner(method:Method,row:&[f64],reference:&Reference,settings:&Settings,
    predict:&dyn Fn(&[f64])->Result<Evaluation,String>,cancel:&AtomicBool,
    semantics:&'static str)->Result<Value,String> {
    check_cancel(cancel)?;
    let prepared=constraints::prepare(row,reference,cancel)?;
    let limit=if settings.budget==0 {64}else{settings.budget.clamp(16,256)};
    let probability_units=settings.units.to_lowercase().contains("probab")
        ||settings.units.to_lowercase().contains("indicator");
    let mut engine=Engine{predict,cancel,limit,used:0,probability_units,semantics};
    let mut rng=math::Rng::new(settings.seed);
    let mut out=match method {
        Method::Shap=>local::shap(row,reference,&prepared,&mut engine,&mut rng),
        Method::Lime=>local::lime(row,reference,&prepared,&mut engine,&mut rng),
        Method::Anchors=>rules::anchors(row,reference,&prepared,&mut engine,&mut rng),
        Method::Counterfactual=>rules::counterfactual(row,reference,&prepared,&mut engine,&mut rng),
        Method::Permutation=>global::permutation(row,reference,&prepared,settings,&mut engine,&mut rng),
        Method::Ale=>global::ale(row,reference,&prepared,&mut engine,&mut rng),
    }?;
    check_cancel(cancel)?;
    out["method"]=json!(method.id());
    out["target_label"]=json!(settings.target_label);
    out["units"]=json!(if method==Method::Permutation {"Brier loss increase"}else{&settings.units});
    out["quality"]["evaluations"]=json!(engine.used);
    out["quality"]["evaluation_budget"]=json!(limit);
    out["quality"]["requested_budget"]=json!(settings.budget);
    out["details"]["seed"]=json!(settings.seed);
    out["details"]["rng"]=json!("SplitMix64; local deterministic state");
    out["details"]["reference_sha256"]=json!(reference.sha256);
    out["details"]["reference_rows"]=json!(reference.rows.len());
    out["details"]["reference_role"]=json!("Caller-approved training reference; not independent held-out validation");
    out["details"]["target_index"]=json!(settings.target_index);
    out["details"]["model_score_units"]=json!(settings.units);
    out["details"]["decision_semantics"]=json!(semantics);
    let warnings=out["warnings"].as_array_mut().ok_or("Invalid internal explanation envelope")?;
    warnings.extend(prepared.warnings.iter().map(|s|json!(s)));
    if semantics=="scalar_threshold_compatibility" && matches!(method,Method::Anchors|Method::Counterfactual) {
        warnings.push(json!("Only scalar score >= 0.5 predicate agreement/change is available; multiclass decision changes are not verified."));
    }
    if settings.budget!=0 && settings.budget!=limit {
        warnings.push(json!("Requested evaluation budget was clamped to the supported range 16..256."));
    }
    Ok(out)
}

pub(crate) fn check_cancel(cancel:&AtomicBool)->Result<(),String> {
    if cancel.load(Ordering::Relaxed) {Err("Explanation cancelled".into())}else{Ok(())}
}

pub(crate) struct Engine<'a> {
    predict:&'a dyn Fn(&[f64])->Result<Evaluation,String>,
    cancel:&'a AtomicBool,
    limit:usize,
    used:usize,
    probability_units:bool,
    semantics:&'static str,
}

impl Engine<'_> {
    fn check(&self)->Result<(),String>{check_cancel(self.cancel)}
    fn remaining(&self)->usize {self.limit-self.used}
    fn evaluate(&mut self,row:&[f64])->Result<Evaluation,String>{
        self.check()?;
        if self.used>=self.limit{return Err("Explanation evaluation budget exhausted".into());}
        self.used+=1;
        let value=(self.predict)(row).map_err(|e|format!("Model evaluation failed: {e}"))?;
        self.check()?;
        if !value.score.is_finite(){return Err("Model returned a nonfinite target score".into());}
        if self.probability_units && !(0.0..=1.0).contains(&value.score){
            return Err("Model returned a score outside the declared probability/indicator range".into());
        }
        Ok(value)
    }
    fn score(&mut self,row:&[f64])->Result<f64,String>{Ok(self.evaluate(row)?.score)}
}

pub(crate) fn envelope(label:&str,scope:&str)->Value {
    json!({"method_label":label,"scope":scope,"features":[],"quality":{},"details":{},"warnings":[]})
}

pub(crate) fn unsupported(label:&str,scope:&str,state:&str,reason:&str)->Value {
    let mut out=envelope(label,scope);
    out["details"]=json!({"state":state,"reason":reason});out
}

pub(crate) fn feature(group:&FeatureGroup,row:&[f64],contribution:f64)->Value {
    let value=if group.indices.len()==1 {let v=row[group.indices[0]];v.is_finite().then_some(v)}else{None};
    json!({"feature":group.name,"value":value,"contribution":contribution,
        "is_group":group.indices.len()>1,"observed_count":group.indices.iter().filter(|&&i|row[i].is_finite()).count(),"group_size":group.indices.len()})
}

pub(crate) fn group_details(groups:&[FeatureGroup],reference:&Reference)->Value {
    json!(groups.iter().map(|g|json!({"group":g.name,"indices":g.indices,"kind":g.kind,
        "features":g.indices.iter().map(|&i|&reference.feature_names[i]).collect::<Vec<_>>()})).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests;
