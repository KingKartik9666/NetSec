//! Phase 1-only local LLM experiment; no app routes or production registration.
pub mod adapter;
pub mod grounding;

use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum SpikeError {
    #[error("{0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Dynamic(#[from] libloading::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Regex(#[from] regex::Error),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub model: PathBuf,
    pub native_dir: PathBuf,
    pub prompt_file: PathBuf,
    pub evidence: Value,
    pub passages: Vec<Value>,
    pub timeout_seconds: u64,
}
