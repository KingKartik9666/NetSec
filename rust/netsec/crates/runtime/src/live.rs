//! Explicit native observation and independently identified live-compatible models.
pub mod features;
mod models;
#[cfg(windows)]
mod native;
pub mod packet;
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Packets mode closes one tumbling window per live77 verdict (owner decision 3 Nov 2026: 10 s, overriding the plan's 1 s).
const PACKETS_PERIOD: Duration = Duration::from_secs(10);
/// A packets session is stale after more than two missed windows; previous verdicts are then cleared.
const PACKETS_STALE_SECONDS: f64 = 25.0;
/// Rate history entries (twenty minutes of 10 s windows).
const HISTORY_LIMIT: usize = 120;

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
            models::LIVE77_MODEL_ID => "multiclass_prediction",
            "live_partial_xgboost" | "live_binary_xgboost" => return Err(RETIRED_LIVE_MODEL.into()),
            _ => {
                return Err("Select one of the separately identified live-compatible models".into())
            }
        };
        let evidence = state[field].clone();
        if evidence["available"] != true
            || evidence["model_executed"] != true
            || evidence["observation_id"] != observation_id
            || evidence["model_id"] != model_id
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
                        PACKETS_STALE_SECONDS
                    } else {
                        8.0
                    }
            {
                state["status"] = json!("stale");
                state["message"] =
                    json!("No fresh observations are available. Stop this session and retry.");
                for field in [
                    "observation",
                    "feature_observation",
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
                "capture_started":false,"permission_checked":false,"models":model_compatibility()}),
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
const RETIRED_LIVE_MODEL: &str = "This live model is retired; select the live77 model (live77_xgboost)";
/// Plan step 7: the one explicit display label per live model id, used by live English (transport.rs) and live chat
/// (chat.rs). It equals the SHA-pinned bundle's `label`; retired and unknown ids are refused, never given a fallback label.
pub fn model_label(model_id: &str) -> Result<&'static str, String> {
    match model_id {
        models::LIVE77_MODEL_ID => Ok("Live XGBoost (77-class, 10 s windows)"),
        "live_partial_xgboost" | "live_binary_xgboost" => Err(RETIRED_LIVE_MODEL.into()),
        _ => Err(format!("Unknown live model id; only {} is live-compatible", models::LIVE77_MODEL_ID)),
    }
}
/// Which models live windows can feed (adapters() and every packets snapshot). The retired partial/binary bundles stay on
/// disk unused and their ids are refused by `evidence`. Wording must survive the Live page caveat filter.
fn model_compatibility() -> Value {
    json!({"full_csv_models":"incompatible: 655 prepared sensor/log inputs are not supplied by live windows",
        "live77":format!("{}: Live XGBoost (77-class, 10 s windows); 252 packet-header inputs from each 10 ± 1 s window; trained on DataSense windows of 1-10 s",models::LIVE77_MODEL_ID),
        "binary":"none: no live77 binary calibration exists, so no attack probability is shown",
        "retired":"live_partial_xgboost and live_binary_xgboost are retired and refused"})
}
fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}
/// Plan step 6: every completed packets window passes the live77 gate (10 ± 1 s, no capture loss) before the
/// SHA-pinned live77 bundle (loaded once, on first use) scores its row. Damaged-header windows are scored with those
/// inputs missing. A window with no packets from the device is not scored (owner, 4 Nov 2026: the model calls empty
/// windows mitm_impersonation, a false alert on an idle PC); it reports no_traffic. Skipped windows carry no
/// substitute verdict. binary_prediction stays null (no live77 binary calibration).
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
fn score_window(
    loaded: &mut Option<models::Models>,
    root: &Path,
    window: &packet::Window,
    seconds: f64,
    id: &str,
) -> (Value, Value) {
    match window.live77_gate(seconds) {
        Ok(()) if window.live77.packets() == 0 => (
            json!({"available":false,"model_executed":false,"status":"no_traffic",
                "message":"No traffic from this device in this window, so it was not scored.","observation_id":id}),
            Value::Null,
        ),
        Ok(()) => loaded
            .get_or_insert_with(|| models::Models::load(root))
            .predict_window(window, id),
        Err(error) => (
            json!({"available":false,"model_executed":false,"status":"skipped","error":error,"observation_id":id}),
            Value::Null,
        ),
    }
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
    let capture_error = Arc::new(Mutex::new(None));
    let mut capture_thread = None;
    let mut capture_snapshots = None;
    if mode == "packets" {
        let (capture_ready, capture_synced) = std::sync::mpsc::sync_channel(1);
        let selected = chosen.clone();
        let running = inner.clone();
        let (snapshot_request, snapshot_receiver) = std::sync::mpsc::sync_channel(1);
        capture_snapshots = Some(snapshot_request);
        let errors = capture_error.clone();
        capture_thread = Some(std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                native::capture(&selected, &running.stop, snapshot_receiver, capture_ready)
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
    let mut live77 = None;
    let mut sequence = 0u64;
    let result = (|| {
        while !inner.stop.load(Ordering::Acquire)
            && inner.generation.load(Ordering::Acquire) == generation
            && started.elapsed() < Duration::from_secs(3600)
        {
            let period = if mode == "packets" {
                PACKETS_PERIOD
            } else {
                Duration::from_secs(2)
            };
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
            let mut now = Instant::now();
            let mut seconds = now.duration_since(last).as_secs_f64();
            // Capture owns rotation and samples drop counters at this boundary.
            // Packet reads continue independently once the completed window arrives.
            let completed = if mode == "packets" {
                let (reply, response) = std::sync::mpsc::sync_channel(1);
                capture_snapshots
                    .as_ref()
                    .ok_or("Capture snapshot channel missing")?
                    .send(reply)
                    .map_err(|_| "Capture ended before window closure")?;
                let snapshot = response
                    .recv_timeout(Duration::from_secs(2))
                    .map_err(|_| "Capture did not verify window quality within two seconds")?;
                now = snapshot.closed_at;
                seconds = snapshot.seconds;
                Some(snapshot.window)
            } else {
                None
            };
            last = now;
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
            let (count, size, assessment, partial, binary, recent, counts, feature_observation) =
                if let Some(window) = completed {
                    let predictions =
                        score_window(&mut live77, Path::new(&inner.root), &window, seconds, &id);
                    (
                        window.packets,
                        window.bytes,
                        window.assessment(seconds),
                        predictions.0,
                        predictions.1,
                        json!(window.recent_packets),
                        json!(&window),
                        window.feature_observation(seconds),
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
            if history.len() > HISTORY_LIMIT {
                history.pop_front();
            }
            let executed = partial["model_executed"] == true || binary["model_executed"] == true;
            let snapshot = json!({"status":"running","running":true,"mode":mode,"adapter":chosen.name,"adapter_id":chosen.id,
                "observation":{"id":id,"observed_at_unix_ms":now_ms(),"seconds":seconds,"packets_per_second":rate,"bytes_per_second":byte_rate,"packets_observed":count,"active_connections":connections.len(),"session_elapsed_seconds":started.elapsed().as_secs(),"counters_reset":reset},
                "message":if mode=="packets"{"Actual own-adapter passive headers. The live77 model scores each completed ten-second window (10 ± 1 s); windows with capture loss are skipped."}else{"Actual Windows traffic counters and endpoints. No intrusion model executes in connections mode."},
                "os_counters":current,"connections":connections.into_iter().take(256).collect::<Vec<_>>(),"assessment":assessment,"header_counts":counts,"feature_observation":feature_observation,"recent_packets":recent,"history":history,
                "multiclass_prediction":partial,"binary_prediction":binary,"model_executed":executed,"capture":{"started":mode=="packets","permission_checked":mode=="packets","promiscuous":false,"payload_saved":false,"pcap_saved":false,"own_adapter_only":true},
                "model_compatibility":model_compatibility(),
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
        let running = json!({"running":true,"mode":"packets","status":"running","observation":{"id":"old"},"multiclass_prediction":{"label":"attack"},"model_executed":true,"feature_observation":{"features":{"old":1}}});
        // One 10 s window plus the 2 s closure wait and the 1 s gate tolerance is still current.
        let current = PACKETS_PERIOD + Duration::from_secs(3);
        *live.inner.last.lock().unwrap() = Some(Instant::now() - current);
        *live.inner.state.lock().unwrap() = running.clone();
        assert_eq!(live.status()["status"], "running");
        assert_eq!(live.status()["multiclass_prediction"]["label"], "attack");
        *live.inner.last.lock().unwrap() =
            Some(Instant::now() - Duration::from_secs_f64(PACKETS_STALE_SECONDS + 1.0));
        let value = live.status();
        assert_eq!(value["status"], "stale");
        assert!(value["observation"].is_null());
        assert!(value["feature_observation"].is_null());
        assert!(value["multiclass_prediction"].is_null());
        assert!(value["binary_prediction"].is_null());
        assert_eq!(value["model_executed"], false);
    }
    #[test]
    fn packets_cadence_matches_the_live77_window() {
        // Owner decision 3 Nov 2026: one 10 s tumbling window per verdict (overrides the plan's 1 s).
        assert_eq!(PACKETS_PERIOD.as_secs_f64(), features::LIVE77_NOMINAL_SECONDS);
        // Stale only after more than two missed windows (closure wait 2 s, gate tolerance 1 s).
        assert!(PACKETS_STALE_SECONDS > PACKETS_PERIOD.as_secs_f64() + 3.0);
        assert!(PACKETS_STALE_SECONDS < 3.0 * PACKETS_PERIOD.as_secs_f64());
        // The 3 s UI poll sees every 10 s window; history covers twenty minutes.
        assert_eq!(HISTORY_LIMIT as f64 * PACKETS_PERIOD.as_secs_f64(), 1200.0);
    }
    /// Live77 features holding one TCP SYN frame sent by `device`, so the window has traffic to score.
    fn traffic(device: [u8; 6]) -> features::Live77Features {
        let mut bytes = vec![0u8; 54];
        bytes[6..12].copy_from_slice(&device);
        bytes[12..14].copy_from_slice(&0x0800u16.to_be_bytes());
        bytes[14] = 0x45;
        bytes[16..18].copy_from_slice(&40u16.to_be_bytes());
        bytes[23] = 6;
        bytes[26..30].copy_from_slice(&[192, 0, 2, 10]);
        bytes[30..34].copy_from_slice(&[198, 51, 100, 20]);
        bytes[36..38].copy_from_slice(&443u16.to_be_bytes());
        bytes[46] = 0x50;
        bytes[47] = 0x02;
        let mut live77 = features::Live77Features::new(Some(device));
        live77.add_frame(&bytes, 54);
        live77
    }
    fn workspace() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(4)
            .unwrap()
            .to_path_buf()
    }
    #[test]
    fn packets_windows_are_gated_at_ten_seconds_then_scored_by_live77() {
        let root = workspace();
        let device = [2, 0, 0, 0, 0, 0x50];
        let window = |drops: u64| {
            packet::Window { live77: traffic(device), capture_drops: drops, ..packet::Window::default() }
        };
        let empty = || packet::Window { live77: features::Live77Features::new(Some(device)), ..packet::Window::default() };
        let mut loaded = None;
        // A delayed window is skipped before any model loads, with no substitute and no binary slot.
        let (late, binary) = score_window(&mut loaded, &root, &window(0), 12.5, "late");
        assert_eq!(
            (late["status"].as_str(), late["available"].as_bool(), late["model_executed"].as_bool()),
            (Some("skipped"), Some(false), Some(false))
        );
        assert!(late["error"].as_str().unwrap().contains("ten seconds"));
        assert!(binary.is_null() && loaded.is_none() && late.get("label").is_none());
        // A window with no packets from the device is not scored (owner, 4 Nov 2026) and has no substitute verdict.
        for seconds in [9.0, 10.0, 11.0] {
            let (out, binary) = score_window(&mut loaded, &root, &empty(), seconds, "idle");
            assert_eq!((out["status"].as_str(), out["model_executed"].as_bool()), (Some("no_traffic"), Some(false)), "{out}");
            assert!(out.get("label").is_none() && binary.is_null());
        }
        // Windows with device traffic are scored by live77.
        for (seconds, id) in [(10.0, "busy"), (9.0, "early"), (11.0, "slow")] {
            let (out, binary) = score_window(&mut loaded, &root, &window(0), seconds, id);
            assert_eq!(out["model_executed"], true, "{seconds} s: {out}");
            assert_eq!(
                (out["model_id"].as_str(), out["observation_id"].as_str(), out["window_seconds"].as_f64()),
                (Some("live77_xgboost"), Some(id), Some(features::LIVE77_NOMINAL_SECONDS))
            );
            let names = out["feature_names"].as_array().unwrap();
            let at = |name: &str| names.iter().position(|n| n == name).unwrap();
            assert_eq!(out["features"][at("quality_no_packets_observed")], 0.0);
            assert_eq!(out["features"][at("window_duration_seconds")], 10.0);
            assert!(binary.is_null());
            let size = serde_json::to_vec(&out).unwrap().len();
            println!("live77 status prediction JSON for a {seconds} s window ({}): {size} bytes", out["label"]);
            assert!(size < 1 << 20);
        }
        // Unreadable headers no longer skip the window: the live77 row marks those inputs missing.
        let mut damaged = window(0);
        damaged.incomplete_headers = 1;
        assert_eq!(score_window(&mut loaded, &root, &damaged, 10.0, "damaged").0["model_executed"], true);
        // Capture loss is skipped by the gate; the model never sees it.
        let (lossy, binary) = score_window(&mut loaded, &root, &window(1), 10.0, "loss");
        assert_eq!((lossy["status"].as_str(), lossy["model_executed"].as_bool()), (Some("skipped"), Some(false)));
        assert!(lossy["error"].as_str().unwrap().contains("Capture loss") && binary.is_null());
        // A missing bundle stays unavailable, never replaced.
        let mut missing = None;
        let (out, _) = score_window(&mut missing, Path::new("no-such-native-live-runtime"), &window(0), 10.0, "none");
        assert_eq!((out["available"].as_bool(), out["model_executed"].as_bool()), (Some(false), Some(false)));
        assert!(out.get("label").is_none());
    }
    #[test]
    fn compatibility_strings_describe_live77_not_the_retired_models() {
        let text = model_compatibility().to_string();
        assert!(text.contains(models::LIVE77_MODEL_ID) && text.contains("252") && text.contains("10 ± 1 s"), "{text}");
        for retired in ["170-feature", "seven-count", "partial-input"] {
            assert!(!text.to_lowercase().contains(retired), "{retired}: {text}");
        }
        // The Live page hides strings matching its caveat filter; these descriptions must stay visible.
        for hidden in ["unverif", "untested", "provisional", "limitation", "caveat", "experimental", "accuracy_verified", "not verified", "not_verified", "not-verified", "notverified", "unvalidated"] {
            assert!(!text.to_lowercase().contains(hidden), "{hidden}: {text}");
        }
    }
    #[test]
    fn evidence_requires_current_completed_model_and_is_invalid_after_stop() {
        let live = Live::new();
        *live.inner.last.lock().unwrap() = Some(Instant::now());
        let running = |model_id: &str| json!({"running":true,"mode":"packets","status":"running","observation":{"id":"fixture"},
            "multiclass_prediction":{"available":true,"model_executed":true,"observation_id":"fixture","model_id":model_id,"label":"hypothesis"}});
        *live.inner.state.lock().unwrap() = running("live77_xgboost");
        assert_eq!(live.evidence("fixture", "live77_xgboost").unwrap()["model_id"], "live77_xgboost");
        assert!(live.evidence("old", "live77_xgboost").is_err());
        // The retired legacy ids are refused, never mapped onto live77 output.
        for retired in ["live_partial_xgboost", "live_binary_xgboost"] {
            assert!(live.evidence("fixture", retired).unwrap_err().contains("retired"), "{retired}");
        }
        assert!(live.evidence("fixture", "xgboost").is_err());
        // The stored prediction must come from the requested model.
        *live.inner.state.lock().unwrap() = running("live_partial_xgboost");
        assert!(live.evidence("fixture", "live77_xgboost").is_err());
        *live.inner.state.lock().unwrap() = running("live77_xgboost");
        live.stop();
        assert!(live.evidence("fixture", "live77_xgboost").is_err());
    }
    const LIVE77_LABEL: &str = "Live XGBoost (77-class, 10 s windows)";
    #[test]
    fn live_model_labels_are_explicit_and_other_ids_are_refused() {
        // Plan step 7: one explicit label per live id (used by live English in transport.rs and live chat in chat.rs).
        assert_eq!(model_label(models::LIVE77_MODEL_ID), Ok(LIVE77_LABEL));
        let bundle: Value = serde_json::from_slice(&std::fs::read(workspace().join("artifacts/application/live_models_v1/live77_xgboost/bundle.json")).unwrap()).unwrap();
        assert_eq!(bundle["label"], LIVE77_LABEL, "the label must match the SHA-pinned live77 bundle");
        for retired in ["live_partial_xgboost", "live_binary_xgboost"] {
            assert!(model_label(retired).unwrap_err().contains("retired"), "{retired}");
        }
        // Nothing falls through to the old 'Experimental packet-only XGBoost' label.
        for unknown in ["", "xgboost", "xgboost_app", "LIVE77_XGBOOST", "live77_xgboost ", "live77"] {
            assert!(model_label(unknown).unwrap_err().contains("Unknown live model"), "{unknown:?}");
        }
        // The Live page caveat filter must not hide the label.
        for hidden in ["unverif", "untested", "provisional", "limitation", "caveat", "experimental", "accuracy_verified", "not verified", "not_verified", "not-verified", "notverified", "unvalidated"] {
            assert!(!LIVE77_LABEL.to_lowercase().contains(hidden), "{hidden}");
        }
    }
    #[test]
    fn live_chat_context_carries_the_live77_label_and_refuses_other_ids() {
        // Builds the app with a language server that never starts (failed integrity); no capture and no llama-server.
        let root = workspace();
        let app = crate::application::Application::open(root.clone(), crate::language::server::LlamaServer::unverified()).unwrap();
        let window = packet::Window { live77: traffic([2, 0, 0, 0, 0, 0x50]), ..packet::Window::default() };
        let (prediction, _) = score_window(&mut None, &root, &window, 10.0, "chat-window");
        let bundle: Value = serde_json::from_slice(&std::fs::read(root.join("artifacts/application/live_models_v1/live77_xgboost/bundle.json")).unwrap()).unwrap();
        assert_eq!((&prediction["model_label"], &prediction["provenance_note"]), (&json!(LIVE77_LABEL), &bundle["provenance_note"]));
        // language::live_schema reads the 77 classes and the 252 names from the live prediction itself.
        let (count, classes, names) = crate::language::live_schema(&prediction);
        assert_eq!((count, classes.len(), names.len()), (77, 77, 252));
        assert_eq!((classes[0].as_str(), classes[70].as_str(), classes[71].as_str()), ("benign", "recon_scan", "recon_ping-sweep"));
        *app.live.inner.last.lock().unwrap() = Some(Instant::now());
        *app.live.inner.state.lock().unwrap() = json!({"running":true,"mode":"packets","status":"running","observation":{"id":"chat-window"},"multiclass_prediction":prediction});
        let request = |model_id: &str| crate::chat::ContextRequest { model_id: model_id.into(), job_id: None, observation_id: Some("chat-window".into()), row_index: 0, explanation_id: None };
        let view = app.chat.create(&app, request(models::LIVE77_MODEL_ID)).unwrap();
        assert_eq!((view["model_id"].as_str(), view["model_label"].as_str(), view["source"].as_str()), (Some(models::LIVE77_MODEL_ID), Some(LIVE77_LABEL), Some("live")));
        assert_eq!(view["title"], format!("{LIVE77_LABEL} · pinned live window"));
        assert!(!view.to_string().contains("packet-only") && !view.to_string().contains("Partial-input"), "{view}");
        for other in ["live_partial_xgboost", "live_binary_xgboost", "xgboost_app", ""] {
            assert!(app.chat.create(&app, request(other)).is_err(), "{other:?}");
        }
        app.llm.shutdown();
    }
}
