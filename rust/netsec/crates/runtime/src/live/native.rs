//! Windows owns OS counters; an explicitly started Npcap handle owns passive reads.
//! There are no send/injection APIs. Unsafe calls stay at this native boundary.
use super::features::Live77Features;
use super::packet::{self, LocalAddresses, Window};
use libloading::os::windows::Library;
use netsec_capture::Adapter;
use serde::Serialize;
use std::{
    ffi::{c_char, c_int, c_void, CString},
    mem,
    path::PathBuf,
    ptr,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, SyncSender},
    },
    thread,
    time::{Duration, Instant},
};
use windows_sys::Win32::{
    NetworkManagement::IpHelper::{ConvertInterfaceAliasToLuid, GetIfEntry2, MIB_IF_ROW2},
    System::LibraryLoader::{LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32},
};

#[derive(Clone, Debug, Serialize)]
pub struct Counters {
    pub packets: u64,
    pub bytes: u64,
    pub mac: Option<[u8; 6]>,
    pub is_up: bool,
}
pub fn counters(alias: &str) -> Result<Counters, String> {
    if alias.encode_utf16().count() > 256 || alias.contains('\0') {
        return Err("Invalid native adapter alias".into());
    }
    let alias: Vec<u16> = alias.encode_utf16().chain([0]).collect();
    // SAFETY: Windows SDK POD row is initialized; both API outputs point into it.
    let mut row: MIB_IF_ROW2 = unsafe { mem::zeroed() };
    let code = unsafe { ConvertInterfaceAliasToLuid(alias.as_ptr(), &mut row.InterfaceLuid) };
    if code != 0 {
        return Err(format!("The selected adapter disappeared or is unavailable (ConvertInterfaceAliasToLuid code {code})"));
    }
    let code = unsafe { GetIfEntry2(&mut row) };
    if code != 0 {
        return Err(format!(
            "Windows adapter counters failed (GetIfEntry2 code {code})"
        ));
    }
    let packets = row
        .InUcastPkts
        .checked_add(row.InNUcastPkts)
        .and_then(|v| v.checked_add(row.OutUcastPkts))
        .and_then(|v| v.checked_add(row.OutNUcastPkts))
        .ok_or("Windows packet counters overflowed")?;
    let bytes = row
        .InOctets
        .checked_add(row.OutOctets)
        .ok_or("Windows byte counters overflowed")?;
    let mac = if row.PhysicalAddressLength == 6 {
        Some(
            <[u8; 6]>::try_from(&row.PhysicalAddress[..6])
                .map_err(|_| "Invalid native MAC address")?,
        )
    } else {
        None
    };
    Ok(Counters {
        packets,
        bytes,
        mac,
        is_up: row.OperStatus == 1,
    })
}

/// BPF filter. Owner decision (3 Nov 2026): the live77 window counts every frame to or from the
/// device MAC, including non-IP frames such as LLDP, so that MAC is admitted without an EtherType
/// test. The IP/ARP frames of the selected addresses stay admitted as before. Snaplen stays 192
/// bytes and no frame or payload byte is retained.
fn capture_filter(ips: &[std::net::IpAddr], mac: Option<[u8; 6]>) -> Result<String, String> {
    let hosts = ips
        .iter()
        .map(|ip| format!("host {ip}"))
        .collect::<Vec<_>>()
        .join(" or ");
    let ether = mac.map(|mac| {
        let text = mac.iter().map(|v| format!("{v:02x}")).collect::<Vec<_>>();
        format!("ether host {}", text.join(":"))
    });
    match (hosts.is_empty(), ether) {
        (true, None) => {
            Err("The selected adapter has no verified own addresses for passive filtering".into())
        }
        (true, Some(ether)) => Ok(ether),
        (false, None) => Ok(format!("(ip or ip6 or arp) and ({hosts})")),
        (false, Some(ether)) => Ok(format!("({ether}) or ((ip or ip6 or arp) and ({hosts}))")),
    }
}

/// The frames the former `(ip or ip6 or arp) and (...)` filter passed, so the
/// npcap-network-v1 observation view reads exactly what it read before the widening.
fn observation_frame(bytes: &[u8]) -> bool {
    bytes.get(12..14).is_some_and(|kind| {
        matches!(
            u16::from_be_bytes([kind[0], kind[1]]),
            0x0800 | 0x86dd | 0x0806
        )
    })
}

#[repr(C)]
struct PacketHeader {
    seconds: i32,
    microseconds: i32,
    captured: u32,
    original: u32,
}

fn timestamp_ns(seconds: i64, microseconds: i32) -> Result<i64, String> {
    if seconds < 0 || !(0..1_000_000).contains(&microseconds) {
        return Err("Npcap returned an invalid microsecond timestamp".into());
    }
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|v| v.checked_add(i64::from(microseconds) * 1000))
        .ok_or_else(|| "Npcap timestamp exceeds signed nanosecond bounds".into())
}
#[repr(C)]
struct Program {
    length: u32,
    instructions: *mut c_void,
}
// Windows pcap_stat has six u32 fields, unlike the three-field Unix layout.
// Source: upstream pcap/pcap.h (_WIN32 ps_capt, ps_sent, ps_netdrop).
#[repr(C)]
#[derive(Default)]
struct Statistics {
    received: u32,
    dropped: u32,
    interface_dropped: u32,
    captured: u32,
    sent: u32,
    network_dropped: u32,
}
type Open = unsafe extern "C" fn(*const c_char, c_int, c_int, c_int, *mut c_char) -> *mut c_void;
type Close = unsafe extern "C" fn(*mut c_void);
type Next = unsafe extern "C" fn(*mut c_void, *mut *const PacketHeader, *mut *const u8) -> c_int;
type Link = unsafe extern "C" fn(*mut c_void) -> c_int;
type Nonblock = unsafe extern "C" fn(*mut c_void, c_int, *mut c_char) -> c_int;
type Compile = unsafe extern "C" fn(*mut c_void, *mut Program, *const c_char, c_int, u32) -> c_int;
type Filter = unsafe extern "C" fn(*mut c_void, *mut Program) -> c_int;
type Free = unsafe extern "C" fn(*mut Program);
type Stats = unsafe extern "C" fn(*mut c_void, *mut Statistics) -> c_int;
type GetError = unsafe extern "C" fn(*mut c_void) -> *const c_char;

struct Capture {
    _library: Library,
    handle: *mut c_void,
    close: Close,
    next: Next,
    stats: Stats,
    get_error: GetError,
}
impl Drop for Capture {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.close)(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

fn error_text(bytes: &[c_char]) -> String {
    let bytes = bytes
        .iter()
        .take_while(|v| **v != 0)
        .map(|v| *v as u8)
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).into_owned()
}

impl Capture {
    fn error(&self) -> String {
        // SAFETY: pcap_geterr returns a native-owned NUL string valid while the
        // matching live handle exists. Read at most PCAP_ERRBUF_SIZE bytes.
        let value = unsafe { (self.get_error)(self.handle) };
        if value.is_null() {
            return "Native packet capture failed".into();
        }
        let mut text = Vec::new();
        for i in 0..256 {
            let byte = unsafe { *value.add(i) };
            if byte == 0 {
                break;
            }
            text.push(byte as u8)
        }
        String::from_utf8_lossy(&text).into_owned()
    }
    fn open(adapter: &Adapter) -> Result<Self, String> {
        let capability =
            netsec_capture::windows_adapter::npcap(false).map_err(|e| e.to_string())?;
        if !capability.packet_capture_available {
            return Err(capability.packet_capture_message);
        }
        let id = adapter.id.to_ascii_lowercase();
        let device = capability
            .devices
            .iter()
            .find(|device| device.to_ascii_lowercase().ends_with(&id))
            .ok_or("Cannot match the selected Windows adapter to an Npcap device")?;
        let dll = PathBuf::from(capability.dll.ok_or("Npcap DLL identity missing")?);
        let name = CString::new(device.as_str()).map_err(|_| "Invalid native capture device")?;
        // SAFETY: the path comes from GetSystemDirectoryW/Npcap. Restrict DLL
        // dependency search to this DLL's directory and System32, never PATH.
        let library = unsafe {
            Library::load_with_flags(
                &dll,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
            )
        }
        .map_err(|e| e.to_string())?;
        // Resolve every used function before acquiring the handle. Function
        // pointers remain valid because Capture retains the loaded library.
        let (open, close, next, link, nonblock, compile, filter, free, stats, get_error) = unsafe {
            (
                *library
                    .get::<Open>(b"pcap_open_live\0")
                    .map_err(|e| e.to_string())?,
                *library
                    .get::<Close>(b"pcap_close\0")
                    .map_err(|e| e.to_string())?,
                *library
                    .get::<Next>(b"pcap_next_ex\0")
                    .map_err(|e| e.to_string())?,
                *library
                    .get::<Link>(b"pcap_datalink\0")
                    .map_err(|e| e.to_string())?,
                *library
                    .get::<Nonblock>(b"pcap_setnonblock\0")
                    .map_err(|e| e.to_string())?,
                *library
                    .get::<Compile>(b"pcap_compile\0")
                    .map_err(|e| e.to_string())?,
                *library
                    .get::<Filter>(b"pcap_setfilter\0")
                    .map_err(|e| e.to_string())?,
                *library
                    .get::<Free>(b"pcap_freecode\0")
                    .map_err(|e| e.to_string())?,
                *library
                    .get::<Stats>(b"pcap_stats\0")
                    .map_err(|e| e.to_string())?,
                *library
                    .get::<GetError>(b"pcap_geterr\0")
                    .map_err(|e| e.to_string())?,
            )
        };
        let mut error = [0 as c_char; 256];
        // snaplen is bounded to 192 bytes of transient native buffer; we parse
        // headers only and retain Header fields, never frame/payload bytes.
        let handle = unsafe { open(name.as_ptr(), 192, 0, 100, error.as_mut_ptr()) };
        if handle.is_null() {
            return Err(format!(
                "Npcap could not start passive capture: {}",
                error_text(&error)
            ));
        }
        let capture = Self {
            _library: library,
            handle,
            close,
            next,
            stats,
            get_error,
        };
        if unsafe { link(handle) } != 1 {
            return Err("This adapter does not provide supported Ethernet headers; raw/radiotap capture is unavailable".into());
        }
        if unsafe { nonblock(handle, 1, error.as_mut_ptr()) } != 0 {
            return Err(format!(
                "Bounded nonblocking capture is unavailable: {}",
                error_text(&error)
            ));
        }
        let ips = adapter
            .addresses
            .iter()
            .filter_map(|v| v.parse::<std::net::IpAddr>().ok())
            .collect::<Vec<_>>();
        let expression = CString::new(capture_filter(&ips, counters(&adapter.name)?.mac)?)
            .map_err(|_| "Invalid own-adapter capture filter")?;
        let mut program = Program {
            length: 0,
            instructions: ptr::null_mut(),
        };
        if unsafe { compile(handle, &mut program, expression.as_ptr(), 1, u32::MAX) } != 0 {
            return Err(format!(
                "Own-adapter capture filter failed: {}",
                capture.error()
            ));
        }
        let result = unsafe { filter(handle, &mut program) };
        unsafe { free(&mut program) };
        if result != 0 {
            return Err(format!(
                "Own-adapter capture filter could not be installed: {}",
                capture.error()
            ));
        }
        Ok(capture)
    }
    /// Reads at most one frame into `window`; `false` when none was waiting.
    fn next(&self, own: &LocalAddresses, window: &mut Window) -> Result<bool, String> {
        let mut header = ptr::null();
        let mut bytes = ptr::null();
        let code = unsafe { (self.next)(self.handle, &mut header, &mut bytes) };
        match code {
            0 => Ok(false),
            1 => {
                if header.is_null() || bytes.is_null() {
                    return Err("Npcap returned an invalid frame pointer".into());
                }
                let header = unsafe { &*header };
                if header.captured > 192
                    || header.original < header.captured
                    || header.original > 4 * 1024 * 1024
                {
                    return Err("Npcap frame length exceeds the header-only bound".into());
                }
                let data = unsafe { std::slice::from_raw_parts(bytes, header.captured as usize) };
                let timestamp = timestamp_ns(i64::from(header.seconds), header.microseconds)?;
                // live77 admits by the device MAC itself (all frame types, IPv6 left out).
                window.add_live77_frame(data, header.original);
                if observation_frame(data) {
                    if let Some(mut parsed) =
                        packet::parse_at(data, header.original, Some(timestamp), own)
                    {
                        // pcap_open_live uses the default microsecond timestamp API.
                        // Conversion changes the storage unit, not source precision.
                        parsed.facts.timestamp_resolution_ns = Some(1_000);
                        window.add(parsed);
                    }
                }
                Ok(true)
            }
            -2 => Err("Native packet capture stopped unexpectedly".into()),
            _ => Err(format!(
                "Native packet capture read failed: {}",
                self.error()
            )),
        }
    }
    fn drops(&self) -> Result<u64, String> {
        let mut stats = Statistics::default();
        if unsafe { (self.stats)(self.handle, &mut stats) } != 0 {
            return Err(format!("Capture statistics unavailable: {}", self.error()));
        }
        Ok(u64::from(stats.dropped) + u64::from(stats.interface_dropped))
    }
}

pub struct CompletedWindow {
    pub window: Window,
    pub seconds: f64,
    pub closed_at: Instant,
}

// The capture thread owns both packet aggregation and window closure. A monitor
// cannot rotate a shared Window between the last statistics poll and its update.
struct CaptureWindows {
    current: Window,
    previous_drops: u64,
    began: Instant,
    device: Option<[u8; 6]>,
}
impl CaptureWindows {
    fn new(began: Instant, device: Option<[u8; 6]>) -> Self {
        Self {
            current: Self::window(device),
            previous_drops: 0,
            began,
            device,
        }
    }
    /// Every window carries the device MAC so an empty window is still a scored zero-packet live77 row.
    fn window(device: Option<[u8; 6]>) -> Window {
        Window {
            live77: Live77Features::new(device),
            ..Window::default()
        }
    }
    fn finish(&mut self, total_drops: u64, closed_at: Instant) -> Result<CompletedWindow, String> {
        let drops = total_drops
            .checked_sub(self.previous_drops)
            .ok_or("Npcap drop counters reset or wrapped; window quality is unknown")?;
        self.current.capture_drops = drops;
        let seconds = closed_at.duration_since(self.began).as_secs_f64();
        self.previous_drops = total_drops;
        self.began = closed_at;
        Ok(CompletedWindow {
            window: mem::replace(&mut self.current, Self::window(self.device)),
            seconds,
            closed_at,
        })
    }
}

pub fn capture(
    adapter: &Adapter,
    stop: &AtomicBool,
    snapshots: Receiver<SyncSender<CompletedWindow>>,
    ready: SyncSender<Result<(), String>>,
) -> Result<(), String> {
    let started = Capture::open(adapter);
    let capture = match started {
        Ok(c) => c,
        Err(error) => {
            let _ = ready.send(Err(error.clone()));
            return Err(error);
        }
    };
    let own = LocalAddresses {
        ips: adapter
            .addresses
            .iter()
            .filter_map(|v| v.parse().ok())
            .collect(),
        mac: counters(&adapter.name)?.mac,
    };
    let mut windows = CaptureWindows::new(Instant::now(), own.mac);
    let _ = ready.send(Ok(()));
    while !stop.load(Ordering::Acquire) {
        if let Ok(reply) = snapshots.try_recv() {
            let closed_at = Instant::now();
            // Query at closure, before exposing the window. Statistics failure
            // ends the session instead of labelling the window complete.
            let completed = windows.finish(capture.drops()?, closed_at)?;
            reply
                .send(completed)
                .map_err(|_| "Window consumer stopped".to_owned())?;
        }
        if !capture.next(&own, &mut windows.current)? {
            thread::sleep(Duration::from_millis(5));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packet_timestamps_are_checked_before_forwarding() {
        assert_eq!(timestamp_ns(2, 123_456).unwrap(), 2_123_456_000);
        assert!(timestamp_ns(-1, 0).is_err());
        assert!(timestamp_ns(1, 1_000_000).is_err());
        assert!(timestamp_ns(i64::MAX, 0).is_err());
    }
    #[test]
    fn windows_pcap_header_abi_and_bpf_pointer_layout() {
        assert_eq!(mem::size_of::<PacketHeader>(), 16);
        assert_eq!(mem::size_of::<Program>(), 16);
        assert_eq!(mem::size_of::<Statistics>(), 24);
    }
    #[test]
    fn invalid_alias_is_rejected_without_capture() {
        assert!(counters("bad\0alias").is_err());
    }
    #[test]
    fn capture_filter_admits_every_device_mac_frame_and_keeps_the_observation_set() {
        let ips = [
            "192.0.2.10".parse().unwrap(),
            "2001:db8::10".parse().unwrap(),
        ];
        let mac = Some([0x02, 0, 0, 0, 0, 0x10]);
        assert_eq!(
            capture_filter(&ips, mac).unwrap(),
            "(ether host 02:00:00:00:00:10) or ((ip or ip6 or arp) and (host 192.0.2.10 or host 2001:db8::10))"
        );
        assert_eq!(
            capture_filter(&[], mac).unwrap(),
            "ether host 02:00:00:00:00:10"
        );
        assert_eq!(
            capture_filter(&ips[..1], None).unwrap(),
            "(ip or ip6 or arp) and (host 192.0.2.10)"
        );
        assert!(capture_filter(&[], None).is_err());
        // The observation view still reads exactly the frames the former (ip or ip6 or arp) filter passed.
        let frame = |kind: u16| {
            let mut bytes = vec![0; 60];
            bytes[12..14].copy_from_slice(&kind.to_be_bytes());
            bytes
        };
        for (kind, observed) in [
            (0x0800, true),
            (0x86dd, true),
            (0x0806, true),
            (0x88cc, false),
            (0x8100, false),
        ] {
            assert_eq!(observation_frame(&frame(kind)), observed, "{kind:#06x}");
        }
        assert!(!observation_frame(&[0; 13]));
    }
    #[test]
    fn rotated_windows_keep_the_device_mac_for_the_live77_row() {
        let began = Instant::now();
        let device = Some([0x02, 0, 0, 0, 0, 0x10]);
        let mut windows = CaptureWindows::new(began, device);
        assert_eq!(windows.current.live77.device(), device);
        let first = windows.finish(0, began + Duration::from_secs(10)).unwrap();
        assert_eq!(
            (
                first.window.live77.device(),
                windows.current.live77.device()
            ),
            (device, device)
        );
        let names = ["quality_no_packets_observed".to_string()];
        assert_eq!(first.window.live77_row(&names).unwrap(), [Some(1.)]);
        assert!(CaptureWindows::new(began, None)
            .current
            .live77_row(&names)
            .is_err());
    }
    #[test]
    fn late_capture_loss_is_closed_with_its_own_window() {
        let began = Instant::now();
        let mut windows = CaptureWindows::new(began, None);
        windows.current.packets = 2;
        // A drop appears after a hypothetical 500 ms poll, before rotation.
        let first = windows.finish(3, began + Duration::from_secs(10)).unwrap();
        assert_eq!(first.window.capture_drops, 3);
        assert!(first.window.counts(first.seconds).is_err());
        assert!(first.window.live77_gate(first.seconds).is_err());
        assert_eq!(first.seconds, 10.0);
        windows.current.packets = 1;
        let second = windows.finish(3, began + Duration::from_secs(20)).unwrap();
        assert_eq!(second.window.capture_drops, 0);
        assert!(second.window.counts(second.seconds).is_ok());
        assert_eq!(second.window.live77_gate(second.seconds), Ok(()));
        // An empty 10 s window passes the live77 gate; a delayed rotation does not.
        let third = windows.finish(3, began + Duration::from_secs(30)).unwrap();
        assert_eq!((third.window.packets, third.window.live77_gate(third.seconds)), (0, Ok(())));
        let late = windows.finish(3, began + Duration::from_millis(41_500)).unwrap();
        assert!(late.window.live77_gate(late.seconds).is_err());
        assert!(windows.finish(1, began + Duration::from_secs(30)).is_err());
    }
    #[test]
    #[ignore = "Requires an active Windows adapter; reads OS counters and enumerates Npcap only"]
    fn native_counters_and_driver_discovery_without_starting_capture() {
        let adapters = netsec_capture::windows_adapter::adapters().unwrap();
        let chosen = adapters
            .iter()
            .find(|v| v.is_up && !v.is_loopback && !v.addresses.is_empty())
            .expect("An active Windows adapter is required");
        let first = counters(&chosen.name).unwrap();
        thread::sleep(Duration::from_millis(50));
        let second = counters(&chosen.name).unwrap();
        assert!(second.is_up);
        assert!(second.packets >= first.packets);
        assert!(second.bytes >= first.bytes);
        let capability = netsec_capture::windows_adapter::npcap(false).unwrap();
        assert!(!capability.capture_started);
        assert!(!capability.permission_checked);
        eprintln!(
            "Native adapter {}, packets {}, bytes {}, Npcap {}. No capture handle opened.",
            chosen.name, second.packets, second.bytes, capability.status
        );
    }
}
