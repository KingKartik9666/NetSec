//! Offline header-prefix replay through the exact native parser and aggregator.
//! Input is one bounded JSON object on stdin; this program never opens Npcap.
use netsec_runtime::live::features::{Live77Features, LIVE77_NOMINAL_SECONDS};
use netsec_runtime::live::packet::{parse_at, LocalAddresses, Window};
use serde::Deserialize;
use serde_json::json;
use std::io::{self, Read};
use std::net::IpAddr;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Frame {
    hex: String,
    original_length: u32,
    timestamp_ns: Option<i64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    frames: Vec<Frame>,
    local_ips: Vec<IpAddr>,
    local_mac: Option<String>,
    seconds: f64,
    #[serde(default)]
    capture_drops: u64,
    /// When given (the bundle's names), every frame also feeds the live77 projection with
    /// `local_mac` as the device MAC, and the row is returned in this order.
    #[serde(default)]
    live77_names: Option<Vec<String>>,
}

fn decode(text: &str) -> Result<Vec<u8>, String> {
    if !text.len().is_multiple_of(2) || !text.is_ascii() {
        return Err("Even-length ASCII hex required".into());
    }
    (0..text.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&text[i..i + 2], 16).map_err(|_| "Invalid hexadecimal byte".into())
        })
        .collect()
}
fn run() -> Result<(), String> {
    let mut input = String::new();
    io::stdin()
        .take(64 * 1024 * 1024 + 1)
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    if input.len() > 64 * 1024 * 1024 {
        return Err("Replay input exceeds 64 MiB".into());
    }
    let input: Input = serde_json::from_str(&input).map_err(|e| e.to_string())?;
    if input.frames.len() > 100_000 || input.local_ips.len() > 256 {
        return Err("Replay count exceeds its explicit bound".into());
    }
    let local_mac = input
        .local_mac
        .map(|text| {
            let raw = decode(&text.replace(':', ""))?;
            <[u8; 6]>::try_from(raw).map_err(|_| "Local MAC must contain six bytes".to_owned())
        })
        .transpose()?;
    let own = LocalAddresses {
        ips: input.local_ips.into_iter().collect(),
        mac: local_mac,
    };
    if input.live77_names.as_ref().is_some_and(|n| n.len() > 4096) {
        return Err("Replay live77 name count exceeds its explicit bound".into());
    }
    let mut window = Window {
        live77: Live77Features::new(own.mac),
        ..Window::default()
    };
    let mut not_admitted = 0;
    for frame in input.frames {
        if frame.hex.len() > 384 {
            return Err("Frame exceeds native 192-byte header-prefix bounds".into());
        }
        let bytes = decode(&frame.hex)?;
        if bytes.len() > 192
            || bytes.len() > frame.original_length as usize
            || frame.original_length > 4 * 1024 * 1024
        {
            return Err("Frame exceeds native 192-byte header-prefix bounds".into());
        }
        if input.live77_names.is_some() {
            window.add_live77_frame(&bytes, frame.original_length);
        }
        if let Some(header) = parse_at(&bytes, frame.original_length, frame.timestamp_ns, &own) {
            window.add(header);
        } else {
            not_admitted += 1;
        }
    }
    window.capture_drops = input.capture_drops;
    let counts = window.counts(input.seconds);
    let mut output = json!({"feature_observation":window.feature_observation(input.seconds),
        "legacy_counts":counts.as_ref().ok(),"legacy_error":counts.err(),
        "not_admitted_frames":not_admitted,"offline_replay":true,"capture_performed":false});
    if let Some(names) = &input.live77_names {
        let row = window.live77_row(names);
        output["live77"] = json!({"names":names,"row":row.as_ref().ok(),"error":row.err(),
            "admitted_frames":window.live77.packets(),"nominal_seconds":LIVE77_NOMINAL_SECONDS,
            "publisher_equivalence_verified":false});
    }
    println!("{output}");
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
