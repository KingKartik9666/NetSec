//! Phase 1 only: seeded, observed-column LIME with an injected probability port.
//! This is not the Python `lime` package and is not wired into the application.
#![forbid(unsafe_code)]

pub mod explain;
pub mod neighborhood;
pub mod reference;
pub mod regression;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("Reference I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("Reference JSON failed: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
