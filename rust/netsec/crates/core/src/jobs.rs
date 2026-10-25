//! Immutable observations and bounded job state; no model or transport dependency.
use crate::input::InputSnapshot;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU64, Ordering}}, time::{SystemTime, UNIX_EPOCH}};

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum State { Queued, Running, Completed, Failed, Cancelled, Incompatible }
impl State { pub fn terminal(self) -> bool { !matches!(self, Self::Queued | Self::Running) } }

#[derive(Clone, Serialize)]
pub struct ModelRun {
    pub model_id: String, pub label: String, pub artifact_sha256: String,
    pub status: State, pub elapsed_ms: u64, pub predictions: Vec<Value>, pub error: Option<String>,
}
#[derive(Clone, Serialize)]
pub struct InputIdentity { pub id: String, pub row_count: usize, pub origin: String, pub schema_hash: String }
#[derive(Clone, Serialize)]
pub struct Progress { pub completed: usize, pub total: usize }
#[derive(Clone, Serialize)]
pub struct JobView {
    pub id: String, pub status: State, pub input: InputIdentity, pub models: Vec<ModelRun>,
    pub progress: Progress, pub created_ms: u64, pub agreement: Value,
}
pub struct Job {
    pub input: Arc<InputSnapshot>, pub cancelled: AtomicBool, view: Mutex<JobView>,
}
impl Job {
    pub fn view(&self) -> JobView { self.view.lock().unwrap_or_else(|e| e.into_inner()).clone() }
    pub fn begin(&self, index: usize) -> bool {
        if self.cancelled.load(Ordering::Acquire) { return false; }
        let mut view = self.view.lock().unwrap_or_else(|e| e.into_inner());
        if view.status.terminal() || index >= view.models.len() { return false; }
        view.status = State::Running; view.models[index].status = State::Running; true
    }
    pub fn record(&self, index: usize, result: Result<Vec<Value>, String>, elapsed_ms: u64) {
        let mut view = self.view.lock().unwrap_or_else(|e| e.into_inner());
        if index >= view.models.len() || self.cancelled.load(Ordering::Acquire) { return; }
        let model = &mut view.models[index]; model.elapsed_ms = elapsed_ms;
        match result {
            Ok(predictions) => { model.predictions = predictions; model.status = State::Completed; }
            Err(reason) => { model.status = State::Failed; model.error = Some(reason); }
        }
        view.progress.completed = view.models.iter().filter(|m| m.status.terminal()).count();
        if view.progress.completed == view.progress.total {
            view.status = if view.models.iter().any(|m| m.status == State::Completed) { State::Completed } else { State::Failed };
        }
        view.agreement = agreement(&view.models);
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        let mut view = self.view.lock().unwrap_or_else(|e| e.into_inner());
        if view.status.terminal() { return; }
        view.status = State::Cancelled;
        for model in &mut view.models { if !model.status.terminal() { model.status = State::Cancelled; } }
        view.progress.completed = view.models.len();
    }
    pub fn observation(&self, model: &str, row: usize) -> Result<(Vec<f64>, Value), String> {
        let view = self.view();
        let run = view.models.iter().find(|m| m.model_id == model).ok_or("Model did not run in this job")?;
        if run.status != State::Completed { return Err("The requested model result is not complete".into()); }
        let prediction = run.predictions.get(row).ok_or("Observation index is outside this job")?.clone();
        let values = self.input.rows.get(row).ok_or("Observation is no longer available")?.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
        Ok((values, prediction))
    }
}
fn agreement(models: &[ModelRun]) -> Value {
    let mut counts = BTreeMap::<String, usize>::new();
    for model in models.iter().filter(|m| m.status == State::Completed) {
        if let Some(label) = model.predictions.first().and_then(|v| v["label"].as_str()) {
            *counts.entry(label.into()).or_default() += 1;
        }
    }
    serde_json::json!({"counts":counts,"models_completed":models.iter().filter(|m|m.status==State::Completed).count(),
        "scope":"First observation; descriptive model agreement only, not calibrated confidence or ground truth"})
}
pub struct JobStore { jobs: Mutex<BTreeMap<String, Arc<Job>>>, sequence: AtomicU64, capacity: usize }
impl JobStore {
    pub fn new(capacity: usize) -> Self { Self { jobs: Mutex::new(BTreeMap::new()), sequence: AtomicU64::new(0), capacity: capacity.clamp(1,32) } }
    pub fn get(&self, id: &str) -> Result<Arc<Job>, String> {
        self.jobs.lock().unwrap_or_else(|e|e.into_inner()).get(id).cloned().ok_or("This analysis expired or does not belong to this app session".into())
    }
    pub fn create(&self, input: InputSnapshot, models: Vec<(String,String,String)>) -> Result<(Arc<Job>, bool), String> {
        if models.is_empty() || models.len() > 16 { return Err("Choose between one and sixteen compatible models".into()); }
        let mut jobs = self.jobs.lock().unwrap_or_else(|e|e.into_inner());
        for job in jobs.values() {
            let view = job.view();
            if !view.status.terminal() && view.input.id == input.id && view.models.iter().map(|m| &m.model_id).eq(models.iter().map(|m| &m.0)) {
                return Ok((job.clone(),false));
            }
        }
        while jobs.len() >= self.capacity {
            let oldest = jobs.iter().filter(|(_,j)|j.view().status.terminal()).min_by_key(|(_,j)|j.view().created_ms).map(|(id,_)|id.clone());
            if let Some(id) = oldest { jobs.remove(&id); } else { return Err("The bounded analysis queue is full; finish or cancel another job".into()); }
        }
        let created_ms = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e|e.to_string())?.as_millis() as u64;
        let id = format!("{:x}",Sha256::digest(format!("{}:{}:{}",input.id,created_ms,self.sequence.fetch_add(1,Ordering::Relaxed))));
        let total = models.len();
        let view = JobView { id:id.clone(), status:State::Queued,
            input:InputIdentity{id:input.id.clone(),row_count:input.rows.len(),origin:input.origin.clone(),schema_hash:input.schema_hash.clone()},
            models:models.into_iter().map(|(model_id,label,artifact_sha256)|ModelRun {model_id,label,artifact_sha256,status:State::Queued,elapsed_ms:0,predictions:vec![],error:None}).collect(),
            progress:Progress{completed:0,total},created_ms,agreement:Value::Null };
        let job = Arc::new(Job {input:Arc::new(input),cancelled:AtomicBool::new(false),view:Mutex::new(view)});
        jobs.insert(id,job.clone()); Ok((job,true))
    }
    pub fn cancel_all(&self) { for job in self.jobs.lock().unwrap_or_else(|e|e.into_inner()).values() { job.cancel(); } }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn input() -> InputSnapshot { crate::input::snapshot(vec![vec![Some(0.)]],&["count".into()],"sample").unwrap() }
    fn models() -> Vec<(String,String,String)> { vec![("a".into(),"A".into(),"hash-a".into()),("b".into(),"B".into(),"hash-b".into())] }
    #[test] fn failure_is_scoped_and_completed_result_survives_cancellation() {
        let store=JobStore::new(4); let (job,_) = store.create(input(),models()).unwrap();
        assert!(job.begin(0)); job.record(0,Ok(vec![serde_json::json!({"label":"benign"})]),1);
        assert!(job.begin(1)); job.cancel(); job.record(1,Ok(vec![]),2);
        let view=job.view(); assert_eq!(view.status,State::Cancelled);assert_eq!(view.models[0].status,State::Completed);assert_eq!(view.models[1].status,State::Cancelled);
        assert!(!job.begin(1)); assert!(job.observation("a",0).is_ok());assert!(job.observation("b",0).is_err());
    }
    #[test] fn duplicate_active_submission_reuses_job_and_mixed_results_finish() {
        let store=JobStore::new(4);let (a,new)=store.create(input(),models()).unwrap();assert!(new);
        let (b,new)=store.create(input(),models()).unwrap();assert!(!new);assert!(Arc::ptr_eq(&a,&b));
        a.begin(0);a.record(0,Err("missing artifact".into()),1);a.begin(1);a.record(1,Ok(vec![serde_json::json!({"label":"attack"})]),2);
        assert_eq!(a.view().status,State::Completed);assert!(a.view().models[0].error.is_some());assert_eq!(a.view().models[1].predictions.len(),1);
    }
    #[test] fn active_jobs_are_not_evicted_and_unknown_ids_are_refused() {
        let store=JobStore::new(1);let (a,_)=store.create(input(),models()).unwrap();
        assert!(store.create(input(),vec![("c".into(),"C".into(),"c".into())]).is_err());
        assert!(store.get("other-session").is_err());a.cancel();assert!(store.create(input(),models()).is_ok());
    }
}
