use netsec_capture_spike::{Error, replay};
use serde_json::json;
use std::{env, io::Write};

fn run() -> Result<serde_json::Value, Error> {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("probe") if args.len() == 1 || (args.len() == 2 && args[1] == "--simulate-missing-driver") => {
            #[cfg(windows)]
            {
                use netsec_capture_spike::windows_adapter;
                let start = std::time::Instant::now();
                let adapters = windows_adapter::adapters()?;
                let adapter_ms = start.elapsed().as_secs_f64() * 1000.0;
                let start = std::time::Instant::now();
                let connections = windows_adapter::connections()?;
                let connection_ms = start.elapsed().as_secs_f64() * 1000.0;
                let start = std::time::Instant::now();
                let capability = windows_adapter::npcap(args.len() == 2)?;
                let npcap_ms = start.elapsed().as_secs_f64() * 1000.0;
                Ok(json!({"schema_version":1, "synthetic":false,
                    "adapters":adapters, "connections":connections, "npcap":capability,
                    "timings_ms":{"adapters":adapter_ms,"connections":connection_ms,"npcap":npcap_ms},
                    "limits":{"native_buffer_bytes":16777216,"rows":100000,"adapters":512,"attempts":3},
                    "limitations":"Enumeration only; capture permission and header capture were not executed. No detector, payload storage or traffic transmission."}))
            }
            #[cfg(not(windows))]
            { Err(Error::Invalid("Windows enumeration not executed on this platform".into())) }
        }
        Some("replay") if args.len() == 3 => replay::golden(&args[1], &args[2]),
        _ => Err(Error::Invalid("Usage: netsec-capture-spike probe [--simulate-missing-driver] | replay <pcap> <windows.json>".into())),
    }
}

fn main() {
    let result = run();
    let (value, code) = match result {
        Ok(value) => (value, 0),
        Err(error) => (json!({"error":error.to_string(),"substitute":false}), 1),
    };
    if serde_json::to_writer(std::io::stdout().lock(), &value).is_err()
        || std::io::stdout().write_all(b"\n").is_err()
    {
        std::process::exit(2);
    }
    std::process::exit(code);
}
