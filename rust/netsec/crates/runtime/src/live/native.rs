//! Windows owns OS counters; an explicitly started Npcap handle owns passive reads.
//! There are no send/injection APIs. Unsafe calls stay at this native boundary.
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
        mpsc::SyncSender,
        Mutex,
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

#[repr(C)]
struct PacketHeader {
    seconds: i32,
    microseconds: i32,
    captured: u32,
    original: u32,
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
        let mut conditions = adapter
            .addresses
            .iter()
            .filter_map(|v| v.parse::<std::net::IpAddr>().ok())
            .map(|ip| format!("host {ip}"))
            .collect::<Vec<_>>();
        if let Some(mac) = counters(&adapter.name)?.mac {
            conditions.push(format!(
                "ether host {}",
                mac.iter()
                    .map(|v| format!("{v:02x}"))
                    .collect::<Vec<_>>()
                    .join(":")
            ));
        }
        if conditions.is_empty() {
            return Err(
                "The selected adapter has no verified own addresses for passive filtering".into(),
            );
        }
        let expression = CString::new(format!(
            "(ip or ip6 or arp) and ({})",
            conditions.join(" or ")
        ))
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
    fn next(&self, own: &LocalAddresses) -> Result<Option<packet::Header>, String> {
        let mut header = ptr::null();
        let mut bytes = ptr::null();
        let code = unsafe { (self.next)(self.handle, &mut header, &mut bytes) };
        match code {
            0 => Ok(None),
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
                Ok(packet::parse(data, header.original, own))
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

pub fn capture(
    adapter: &Adapter,
    stop: &AtomicBool,
    window: &Mutex<Window>,
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
    let mut previous_drops = capture.drops()?;
    let mut last_stats = Instant::now();
    let _ = ready.send(Ok(()));
    while !stop.load(Ordering::Acquire) {
        if let Some(header) = capture.next(&own)? {
            window.lock().unwrap_or_else(|e| e.into_inner()).add(header);
        } else {
            thread::sleep(Duration::from_millis(5));
        }
        if last_stats.elapsed() >= Duration::from_millis(500) {
            let drops = capture.drops()?;
            window
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .capture_drops += drops.saturating_sub(previous_drops);
            previous_drops = drops;
            last_stats = Instant::now();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
