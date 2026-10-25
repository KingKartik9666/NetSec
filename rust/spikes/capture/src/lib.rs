//! Isolated Phase 1 spike. Does not start capture, infer, train or change the app.
use serde::Serialize;

pub mod replay;
#[cfg(windows)]
pub mod windows_adapter;

pub const MISSING_MESSAGE: &str = "Install Npcap and restart the app for packet inspection. Windows may require running the app as Administrator.";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(String),
    #[error("{api} failed with Windows code {code}")]
    Windows { api: &'static str, code: u32 },
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Adapter {
    pub name: String,
    pub id: String,
    pub addresses: Vec<String>,
    pub is_up: bool,
    pub is_loopback: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct Connection {
    pub family: String,
    pub protocol: String,
    pub local_ip: String,
    pub local_port: u16,
    pub remote_ip: Option<String>,
    pub remote_port: Option<u16>,
    pub status: String,
    pub pid: u32,
}

#[derive(Debug, Serialize)]
pub struct CaptureCapability {
    pub packet_capture_available: bool,
    pub packet_capture_message: String,
    pub status: String,
    pub dll: Option<String>,
    pub devices: Vec<String>,
    pub error: Option<String>,
    pub capture_started: bool,
    pub permission_checked: bool,
}

impl CaptureCapability {
    pub fn missing() -> Self {
        Self {
            packet_capture_available: false,
            packet_capture_message: MISSING_MESSAGE.into(),
            status: "not_installed".into(),
            dll: None,
            devices: Vec::new(),
            error: None,
            capture_started: false,
            permission_checked: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_is_explicit_and_has_no_substitute() {
        let value = CaptureCapability::missing();
        assert!(!value.packet_capture_available);
        assert_eq!(value.packet_capture_message, MISSING_MESSAGE);
        assert_eq!(value.status, "not_installed");
        assert!(!value.capture_started);
        assert!(!value.permission_checked);
        assert!(value.devices.is_empty());
    }
}
