//! Explicit native observation and independently identified live-compatible models.
mod models;
#[cfg(windows)]
mod native;
mod packet;
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

pub struct Live {
    inner: Arc<Inner>,
}
struct Inner {
    root: PathBuf,
    stop: AtomicBool,
    generation: AtomicU64,
    control: Mutex<()>,
    state: Mutex<Value>,
    last: Mutex<Option<Instant>>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}
impl Default for Live {
    fn default() -> Self {
        Self::new()
    }
}
impl Live {
    pub fn new() -> Self {
        Self::with_root(std::env::current_dir().unwrap_or_default())
    }
    pub fn with_root(root: PathBuf) -> Self {
        Self {
            inner: Arc::new(Inner {
                root,
                stop: AtomicBool::new(true),
                generation: AtomicU64::new(0),
                control: Mutex::new(()),
                state: Mutex::new(stopped()),
                last: Mutex::new(None),
                thread: Mutex::new(None),
            }),
        }
    }
    pub fn cancellation(&self) -> &AtomicBool {
        &self.inner.stop
    }
    pub fn evidence(&self, observation_id: &str, model_id: &str) -> Result<Value, String> {
        let state = self.status();
        if state["status"] != "running"
            || state["mode"] != "packets"
            || state["observation"]["id"] != observation_id
        {
            return Err("The requested live observation is no longer current; previous evidence cannot be reused".into());
        }
        let field = match model_id {
            "live_partial_xgboost" => "multiclass_prediction",
            "live_binary_xgboost" => "binary_prediction",
            _ => {
                return Err("Select one of the separately identified live-compatible models".into())
            }
        };
        let evidence = state[field].clone();
        if evidence["available"] != true
            || evidence["model_executed"] != true
            || evidence["observation_id"] != observation_id
        {
            return Err(
                "This live model has no completed prediction for the current observation".into(),
            );
        }
        Ok(evidence)
    }
    pub fn status(&self) -> Value {
        let mut state = self
            .inner
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        if let Some(last) = *self.inner.last.lock().unwrap_or_else(|e| e.into_inner()) {
            let age = last.elapsed().as_secs_f64();
            state["observation_age_seconds"] = json!(age);
            if state["running"] == true
                && age
                    > if state["mode"] == "packets" {
                        25.0
                    } else {
                        8.0
                    }
            {
                state["status"] = json!("stale");
                state["message"] =
                    json!("No fresh observations are available. Stop this session and retry.");
                for field in [
                    "observation",
                    "multiclass_prediction",
                    "binary_prediction",
                    "assessment",
                ] {
                    state[field] = Value::Null;
                }
                state["model_executed"] = json!(false);
            }
        }
        state
    }
    pub fn adapters(&self) -> Result<Value, String> {
        #[cfg(windows)]
        {
            let adapters =
                netsec_capture::windows_adapter::adapters().map_err(|e| e.to_string())?;
            let cap = netsec_capture::windows_adapter::npcap(false).map_err(|e| e.to_string())?;
            Ok(
                json!({"adapters":adapters.into_iter().filter(|v|!v.is_loopback&&!v.addresses.is_empty()).collect::<Vec<_>>(),"connections_available":true,
                "packet_capture_available":cap.packet_capture_available,"packet_capture_message":cap.packet_capture_message,"packet_capture_status":cap.status,
                "capture_started":false,"permission_checked":false,"models":{"full_csv_models":"incompatible: live windows do not supply 655 prepared features",
                    "packet_partial":"Separate approved 170-feature partial-input XGBoost; seven counts and 163 missing inputs","packet_binary":"Separate experimental seven-count XGBoost; no labelled Windows-PC accuracy"}}),
            )
        }
        #[cfg(not(windows))]
        {
            Err("Native Windows observation adapters are unavailable on this platform".into())
        }
    }
    pub fn start(&self, adapter: &str, mode: &str) -> Result<Value, String> {
        let _control = self.inner.control.lock().unwrap_or_else(|e| e.into_inner());
        if !["connections", "packets"].contains(&mode) {
            return Err("Choose connections or packets explicitly".into());
        }
        if self
            .inner
            .thread
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|h| !h.is_finished())
        {
            return Err(
                "Stop the active observation session and wait for cleanup before starting another"
                    .into(),
            );
        }
        if let Some(handle) = self
            .inner
            .thread
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
        {
            let _ = handle.join();
        }
        #[cfg(windows)]
        {
            let chosen = netsec_capture::windows_adapter::adapters()
                .map_err(|e| e.to_string())?
                .into_iter()
                .find(|v| v.id == adapter && v.is_up && !v.is_loopback && !v.addresses.is_empty())
                .ok_or("Choose a currently active native adapter")?;
            if mode == "packets" {
                let cap =
                    netsec_capture::windows_adapter::npcap(false).map_err(|e| e.to_string())?;
                if !cap.packet_capture_available {
                    return Err(cap.packet_capture_message);
                }
            }
            self.inner.stop.store(false, Ordering::Release);
            let generation = self.inner.generation.fetch_add(1, Ordering::AcqRel) + 1;
            *self.inner.state.lock().unwrap_or_else(|e| e.into_inner()) = json!({"status":"starting","running":true,"mode":mode,"adapter":chosen.name,"observation":null,"model_executed":false});
            *self.inner.last.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
            let (ready, synced) = std::sync::mpsc::sync_channel(1);
            let inner = self.inner.clone();
            let mode = mode.to_owned();
            let handle = std::thread::spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    monitor(&inner, generation, chosen, &mode, ready)
                }));
                let error = match result {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => Some(error),
                    Err(_) => Some("Native observation worker panicked".into()),
                };
                if inner.generation.load(Ordering::Acquire) == generation {
                    inner.stop.store(true, Ordering::Release);
                    *inner.last.lock().unwrap_or_else(|e| e.into_inner()) = None;
                    *inner.state.lock().unwrap_or_else(|e| e.into_inner()) = if let Some(error) =
                        error
                    {
                        json!({"status":"failed","running":false,"observation":null,"model_executed":false,"error":error,"message":error})
                    } else {
                        stopped()
                    };
                }
            });
            *self.inner.thread.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
            match synced.recv_timeout(Duration::from_secs(4)) {
                Ok(Ok(())) => Ok(self.status()),
                Ok(Err(error)) => {
                    self.inner.stop.store(true, Ordering::Release);
                    Err(error)
                }
                Err(_) => {
                    self.inner.stop.store(true, Ordering::Release);
                    Err("Native observation did not become ready within four seconds; capture was cancelled".into())
                }
            }
        }
        #[cfg(not(windows))]
        {
            let _ = adapter;
            Err("Native Windows observation port unavailable".into())
        }
    }
    pub fn stop(&self) -> Value {
        let _control = self.inner.control.lock().unwrap_or_else(|e| e.into_inner());
        self.inner.stop.store(true, Ordering::Release);
        self.inner.generation.fetch_add(1, Ordering::AcqRel);
        *self.inner.last.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.inner.state.lock().unwrap_or_else(|e| e.into_inner()) = stopped();
        let deadline = Instant::now() + Duration::from_secs(6);
        loop {
            let finished = self
                .inner
                .thread
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .is_none_or(|thread| thread.is_finished());
            if finished {
                if let Some(handle) = self
                    .inner
                    .thread
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take()
                {
                    let _ = handle.join();
                }
                break;
            }
            if Instant::now() >= deadline {
                *self.inner.state.lock().unwrap_or_else(|e| e.into_inner()) = json!({"status":"stopping","running":true,"observation":null,"model_executed":false,
                "message":"Stop requested; native cleanup is still pending. Previous verdicts were cleared. Wait before starting another session."});
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        self.status()
    }
}
impl Drop for Live {
    fn drop(&mut self) {
        self.stop();
    }
}
fn stopped() -> Value {
    json!({"status":"stopped","running":false,"observation":null,"binary_prediction":null,"multiclass_prediction":null,"model_executed":false,"message":"Monitoring is off; previous observations and model verdicts are cleared"})
}
fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(windows)]
fn monitor(
    inner: &Arc<Inner>,
    generation: u64,
    chosen: netsec_capture::Adapter,
    mode: &str,
    ready: std::sync::mpsc::SyncSender<Result<(), String>>,
) -> Result<(), String> {
    use std::collections::VecDeque;
    let mut previous = match native::counters(&chosen.name) {
        Ok(v) if v.is_up => v,
        Ok(_) => {
            let e = "The selected adapter is no longer active".to_owned();
            let _ = ready.send(Err(e.clone()));
            return Err(e);
        }
        Err(e) => {
            let _ = ready.send(Err(e.clone()));
            return Err(e);
        }
    };
    let window = Arc::new(Mutex::new(packet::Window::default()));
    let capture_error = Arc::new(Mutex::new(None));
    let mut capture_thread = None;
    if mode == "packets" {
        let (capture_ready, capture_synced) = std::sync::mpsc::sync_channel(1);
        let selected = chosen.clone();
        let running = inner.clone();
        let packets = window.clone();
        let errors = capture_error.clone();
        capture_thread = Some(std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                native::capture(&selected, &running.stop, &packets, capture_ready)
            }));
            if let Some(error) = match result {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(error),
                Err(_) => Some("Native capture worker panicked".into()),
            } {
                *errors.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
            }
        }));
        let outcome = match capture_synced.recv_timeout(Duration::from_secs(3)) {
            Ok(v) => v,
            Err(_) => Err("Passive Npcap capture did not become ready within three seconds".into()),
        };
        if let Err(error) = outcome {
            inner.stop.store(true, Ordering::Release);
            let _ = ready.send(Err(error.clone()));
            if let Some(handle) = capture_thread {
                let _ = handle.join();
            }
            return Err(error);
        }
    }
    let _ = ready.send(Ok(()));
    let started = Instant::now();
    let mut last = Instant::now();
    let mut history = VecDeque::new();
    let mut legacy = None;
    let mut sequence = 0u64;
    let result = (|| {
        while !inner.stop.load(Ordering::Acquire)
            && inner.generation.load(Ordering::Acquire) == generation
            && started.elapsed() < Duration::from_secs(3600)
        {
            let period = Duration::from_secs(if mode == "packets" { 10 } else { 2 });
            while last.elapsed() < period && !inner.stop.load(Ordering::Acquire) {
                if let Some(error) = capture_error
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
                {
                    return Err(error);
                }
                if capture_thread.as_ref().is_some_and(|h| h.is_finished()) {
                    return Err("Passive packet capture ended unexpectedly".into());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            if inner.stop.load(Ordering::Acquire) {
                break;
            }
            let now = Instant::now();
            let seconds = now.duration_since(last).as_secs_f64();
            last = now;
            // Rotate immediately; native reads continue independently during inference.
            let completed = if mode == "packets" {
                Some(std::mem::take(
                    &mut *window.lock().unwrap_or_else(|e| e.into_inner()),
                ))
            } else {
                None
            };
            let current = native::counters(&chosen.name)?;
            if !current.is_up {
                return Err(
                    "The selected adapter became unavailable; previous verdicts were cleared"
                        .into(),
                );
            }
            let reset = current.packets < previous.packets || current.bytes < previous.bytes;
            let packets = current.packets.saturating_sub(previous.packets);
            let bytes = current.bytes.saturating_sub(previous.bytes);
            previous = current.clone();
            let connections = netsec_capture::windows_adapter::connections()
                .map_err(|e| e.to_string())?
                .into_iter()
                .filter(|v| {
                    chosen.addresses.contains(&v.local_ip)
                        && v.remote_ip.as_ref().is_some_and(|v| {
                            v.parse::<std::net::IpAddr>()
                                .is_ok_and(|ip| !ip.is_loopback())
                        })
                })
                .take(1000)
                .collect::<Vec<_>>();
            sequence += 1;
            let id = format!("live-{generation}-{}-{sequence}", now_ms());
            let (count, size, assessment, partial, binary, recent, counts) = if let Some(window) =
                completed
            {
                let predictions = match window.counts(seconds) {
                    Ok(_) => {
                        if legacy.is_none() {
                            legacy = Some(models::Models::load(Path::new(&inner.root)));
                        }
                        legacy
                            .as_ref()
                            .ok_or("Live model loader missing")?
                            .predict(&window, seconds, &id)
                    }
                    Err(error) => {
                        let skip = json!({"available":false,"model_executed":false,"status":"skipped","error":error,"observation_id":id});
                        (skip.clone(), skip)
                    }
                };
                (
                    window.packets,
                    window.bytes,
                    window.assessment(seconds),
                    predictions.0,
                    predictions.1,
                    json!(window.recent_packets),
                    json!(&window),
                )
            } else {
                let pending = connections
                    .iter()
                    .filter(|v| v.status == "SYN_SENT")
                    .count();
                let assessment = json!({"status":if pending>=40{"suspicious"}else if !connections.is_empty()||packets>0{"benign_looking"}else{"idle"},
                    "method":"Live behavior rules","evidence":if pending>=40{json!([{"rule":"connection_attempt_burst","observed":pending,"threshold":40}])}else{json!([])},
                    "limitations":"OS endpoint rules only; no saved intrusion model executes in connections mode. No alert does not prove safety."});
                (
                    packets,
                    bytes,
                    assessment,
                    Value::Null,
                    Value::Null,
                    json!([]),
                    Value::Null,
                )
            };
            let rate = if reset && mode == "connections" {
                None
            } else {
                Some(count as f64 / seconds)
            };
            let byte_rate = if reset && mode == "connections" {
                None
            } else {
                Some(size as f64 / seconds)
            };
            history.push_back(json!({"observation_id":id,"packets_per_second":rate,"bytes_per_second":byte_rate,"rule_status":assessment["status"]}));
            if history.len() > 120 {
                history.pop_front();
            }
            let executed = partial["model_executed"] == true || binary["model_executed"] == true;
            let snapshot = json!({"status":"running","running":true,"mode":mode,"adapter":chosen.name,"adapter_id":chosen.id,
                "observation":{"id":id,"observed_at_unix_ms":now_ms(),"seconds":seconds,"packets_per_second":rate,"bytes_per_second":byte_rate,"packets_observed":count,"active_connections":connections.len(),"session_elapsed_seconds":started.elapsed().as_secs(),"counters_reset":reset},
                "message":if mode=="packets"{"Actual own-adapter passive headers. Separate legacy models run only on complete compatible ten-second windows."}else{"Actual Windows traffic counters and endpoints. No intrusion model executes in connections mode."},
                "os_counters":current,"connections":connections.into_iter().take(256).collect::<Vec<_>>(),"assessment":assessment,"header_counts":counts,"recent_packets":recent,"history":history,
                "multiclass_prediction":partial,"binary_prediction":binary,"model_executed":executed,"capture":{"started":mode=="packets","permission_checked":mode=="packets","promiscuous":false,"payload_saved":false,"pcap_saved":false,"own_adapter_only":true},
                "model_compatibility":{"full_csv_models":"incompatible: 655 prepared sensor/log inputs are not supplied by live windows","partial_multiclass":"separate legacy 170-feature partial-input XGBoost","binary":"separate experimental seven-count XGBoost"},
                "limitations":["Scores and rules do not confirm incidents or guarantee safety","No labelled Windows-PC accuracy or unknown-attack evaluation","No automatic mitigation"]});
            if inner.stop.load(Ordering::Acquire)
                || inner.generation.load(Ordering::Acquire) != generation
            {
                break;
            }
            *inner.state.lock().unwrap_or_else(|e| e.into_inner()) = snapshot;
            *inner.last.lock().unwrap_or_else(|e| e.into_inner()) = Some(now);
        }
        Ok(())
    })();
    inner.stop.store(true, Ordering::Release);
    if let Some(handle) = capture_thread {
        let _ = handle.join();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn construction_is_passive_and_stop_is_idempotent() {
        let live = Live::new();
        assert_eq!(live.status()["running"], false);
        assert!(live.status()["observation"].is_null());
        assert_eq!(live.stop()["status"], "stopped");
        assert_eq!(live.stop()["model_executed"], false);
    }
    #[test]
    fn stale_observations_clear_predictions() {
        let live = Live::new();
        *live.inner.last.lock().unwrap() = Some(Instant::now() - Duration::from_secs(30));
        *live.inner.state.lock().unwrap() = json!({"running":true,"mode":"packets","status":"running","observation":{"id":"old"},"multiclass_prediction":{"label":"attack"},"model_executed":true});
        let value = live.status();
        assert_eq!(value["status"], "stale");
        assert!(value["observation"].is_null());
        assert!(value["multiclass_prediction"].is_null());
        assert_eq!(value["model_executed"], false);
    }
    #[test]
    fn evidence_requires_current_completed_model_and_is_invalid_after_stop() {
        let live = Live::new();
        *live.inner.last.lock().unwrap() = Some(Instant::now());
        *live.inner.state.lock().unwrap() = json!({"running":true,"mode":"packets","status":"running","observation":{"id":"fixture"},
            "multiclass_prediction":{"available":true,"model_executed":true,"observation_id":"fixture","label":"hypothesis"}});
        assert!(live.evidence("fixture", "live_partial_xgboost").is_ok());
        assert!(live.evidence("old", "live_partial_xgboost").is_err());
        assert!(live.evidence("fixture", "live_binary_xgboost").is_err());
        assert!(live.evidence("fixture", "xgboost").is_err());
        live.stop();
        assert!(live.evidence("fixture", "live_partial_xgboost").is_err());
    }
}
