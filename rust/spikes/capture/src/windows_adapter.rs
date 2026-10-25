//! Native-only adapter. All unsafe is confined here; buffers and traversal are bounded.
//! Windows owns the OS ABI and Npcap owns its returned list. Rust never opens a
//! capture handle or calls a send/injection function.
use crate::{Adapter, CaptureCapability, Connection, Error};
use libloading::os::windows::Library;
use std::{
    ffi::{c_char, c_int, c_void},
    mem::size_of,
    net::{Ipv4Addr, Ipv6Addr},
    path::PathBuf,
    ptr,
};
use windows_sys::Win32::{
    Foundation::{ERROR_BUFFER_OVERFLOW, ERROR_INSUFFICIENT_BUFFER, ERROR_NO_DATA},
    NetworkManagement::IpHelper::*,
    Networking::WinSock::{AF_INET, AF_INET6, AF_UNSPEC, SOCKADDR, SOCKADDR_IN, SOCKADDR_IN6},
    System::{
        LibraryLoader::{LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32},
        SystemInformation::GetSystemDirectoryW,
    },
};

const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_ROWS: usize = 100_000;
const MAX_ADAPTERS: usize = 512;

struct Buffer {
    words: Vec<u64>,
    len: usize,
}
impl Buffer {
    fn new(bytes: usize) -> Result<Self, Error> {
        if !(4..=MAX_BYTES).contains(&bytes) {
            return Err(Error::Invalid(
                "Native buffer outside 4..16 MiB bound".into(),
            ));
        }
        Ok(Self {
            words: vec![0; bytes.div_ceil(8)],
            len: bytes,
        })
    }
    fn base(&self) -> *const u8 {
        self.words.as_ptr().cast()
    }
    fn mutable(&mut self) -> *mut c_void {
        self.words.as_mut_ptr().cast()
    }
    fn contains<T>(&self, item: *const T) -> bool {
        let start = self.base() as usize;
        let address = item as usize;
        address >= start
            && address
                .checked_add(size_of::<T>())
                .is_some_and(|end| end <= start + self.len)
    }
    fn read<T: Copy>(&self, item: *const T) -> Result<T, Error> {
        if !self.contains(item) {
            return Err(Error::Invalid(
                "Windows returned pointer outside owned buffer".into(),
            ));
        }
        // SAFETY: range was checked, T is an SDK POD structure and unaligned reads are allowed.
        Ok(unsafe { ptr::read_unaligned(item) })
    }
    fn text16(&self, item: *const u16) -> Result<String, Error> {
        if item.is_null() {
            return Ok(String::new());
        }
        let mut chars = Vec::new();
        for i in 0..4096 {
            let address = (item as usize)
                .checked_add(i * 2)
                .ok_or_else(|| Error::Invalid("String overflow".into()))?;
            let value = self.read(address as *const u16)?;
            if value == 0 {
                return Ok(String::from_utf16_lossy(&chars));
            }
            chars.push(value);
        }
        Err(Error::Invalid("Adapter string exceeds bound".into()))
    }
    fn text8(&self, item: *const u8) -> Result<String, Error> {
        let mut chars = Vec::new();
        for i in 0..4096 {
            let address = (item as usize)
                .checked_add(i)
                .ok_or_else(|| Error::Invalid("String overflow".into()))?;
            let value = self.read(address as *const u8)?;
            if value == 0 {
                return Ok(String::from_utf8_lossy(&chars).into_owned());
            }
            chars.push(value);
        }
        Err(Error::Invalid("Adapter string exceeds bound".into()))
    }
}

fn query(
    api: &'static str,
    initial: usize,
    mut call: impl FnMut(*mut c_void, &mut u32) -> u32,
) -> Result<Buffer, Error> {
    let mut bytes = initial;
    for _ in 0..3 {
        let mut buffer = Buffer::new(bytes)?;
        let mut actual = bytes as u32;
        let code = call(buffer.mutable(), &mut actual);
        if code == 0 {
            buffer.len = actual as usize;
            if buffer.len > bytes {
                return Err(Error::Invalid(
                    "Returned buffer length exceeds allocation".into(),
                ));
            }
            return Ok(buffer);
        }
        if code == ERROR_NO_DATA {
            buffer.words.fill(0);
            buffer.len = 4;
            return Ok(buffer);
        }
        if code != ERROR_BUFFER_OVERFLOW && code != ERROR_INSUFFICIENT_BUFFER {
            return Err(Error::Windows { api, code });
        }
        bytes = actual as usize;
    }
    Err(Error::Invalid(format!(
        "{api} exceeded three resize attempts"
    )))
}

fn socket_address(
    buffer: &Buffer,
    item: *const SOCKADDR,
    bytes: i32,
) -> Result<Option<String>, Error> {
    let addr = buffer.read(item)?;
    match addr.sa_family {
        AF_INET if bytes as usize >= size_of::<SOCKADDR_IN>() => {
            let v = buffer.read(item.cast::<SOCKADDR_IN>())?;
            // SAFETY: SDK IN_ADDR represents network-order IPv4 bytes in S_addr.
            Ok(Some(
                Ipv4Addr::from(unsafe { v.sin_addr.S_un.S_addr }.to_ne_bytes()).to_string(),
            ))
        }
        AF_INET6 if bytes as usize >= size_of::<SOCKADDR_IN6>() => {
            let v = buffer.read(item.cast::<SOCKADDR_IN6>())?;
            // SAFETY: SDK IN6_ADDR Byte is the 16-byte address representation.
            Ok(Some(
                Ipv6Addr::from(unsafe { v.sin6_addr.u.Byte }).to_string(),
            ))
        }
        _ => Ok(None),
    }
}

pub fn adapters() -> Result<Vec<Adapter>, Error> {
    let buffer = query("GetAdaptersAddresses", 15000, |target, bytes| {
        // SAFETY: target is 8-byte aligned, owns bytes writable storage; reserved is NULL.
        unsafe {
            GetAdaptersAddresses(
                AF_UNSPEC as u32,
                GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER,
                ptr::null(),
                target.cast(),
                bytes,
            )
        }
    })?;
    if buffer.len == 4 {
        return Ok(Vec::new());
    }
    let mut item = buffer.base().cast::<IP_ADAPTER_ADDRESSES_LH>();
    let mut result = Vec::new();
    while !item.is_null() {
        if result.len() >= MAX_ADAPTERS {
            return Err(Error::Invalid("Adapter count exceeds bound".into()));
        }
        let value = buffer.read(item)?;
        let mut unicast = value.FirstUnicastAddress;
        let mut ips = Vec::new();
        let mut count = 0;
        while !unicast.is_null() {
            count += 1;
            if count > MAX_ADAPTERS {
                return Err(Error::Invalid("Unicast list exceeds bound".into()));
            }
            let address = buffer.read(unicast)?;
            if let Some(ip) = socket_address(
                &buffer,
                address.Address.lpSockaddr,
                address.Address.iSockaddrLength,
            )? {
                ips.push(ip);
            }
            unicast = address.Next;
        }
        ips.sort();
        ips.dedup();
        result.push(Adapter {
            name: buffer.text16(value.FriendlyName)?,
            id: buffer.text8(value.AdapterName)?,
            addresses: ips,
            is_up: value.OperStatus == 1,
            is_loopback: value.IfType == 24,
        });
        item = value.Next;
    }
    result.sort_by(|a, b| a.name.cmp(&b.name).then(a.id.cmp(&b.id)));
    Ok(result)
}

fn rows<T: Copy>(buffer: &Buffer) -> Result<Vec<T>, Error> {
    let count = buffer.read(buffer.base().cast::<u32>())? as usize;
    if count > MAX_ROWS || 4 + count * size_of::<T>() > buffer.len {
        return Err(Error::Invalid(
            "Invalid or oversized Windows table row count".into(),
        ));
    }
    (0..count)
        .map(|i| buffer.read((buffer.base() as usize + 4 + i * size_of::<T>()) as *const T))
        .collect()
}

fn state(value: u32) -> String {
    match value {
        1 => "CLOSED",
        2 => "LISTEN",
        3 => "SYN_SENT",
        4 => "SYN_RECV",
        5 => "ESTABLISHED",
        6 => "FIN_WAIT1",
        7 => "FIN_WAIT2",
        8 => "CLOSE_WAIT",
        9 => "CLOSING",
        10 => "LAST_ACK",
        11 => "TIME_WAIT",
        12 => "DELETE_TCB",
        _ => "UNKNOWN",
    }
    .into()
}
fn port(value: u32) -> u16 {
    u16::from_be(value as u16)
}
fn tcp(
    family: &str,
    local_ip: String,
    local_port: u32,
    remote_ip: String,
    remote_port: u32,
    tcp_state: u32,
    pid: u32,
) -> Connection {
    let remote_port = port(remote_port);
    Connection {
        family: family.into(),
        protocol: "TCP".into(),
        local_ip,
        local_port: port(local_port),
        remote_ip: (remote_port != 0).then_some(remote_ip),
        remote_port: (remote_port != 0).then_some(remote_port),
        status: state(tcp_state),
        pid,
    }
}
pub fn connections() -> Result<Vec<Connection>, Error> {
    let mut result = Vec::new();
    for family in [AF_INET, AF_INET6] {
        let tcp_buffer = query("GetExtendedTcpTable", 16384, |target, bytes| {
            // SAFETY: aligned writable owned buffer; SDK table class supports both AFs.
            unsafe {
                GetExtendedTcpTable(target, bytes, 0, family as u32, TCP_TABLE_OWNER_PID_ALL, 0)
            }
        })?;
        let udp_buffer = query("GetExtendedUdpTable", 16384, |target, bytes| {
            // SAFETY: aligned writable owned buffer, reserved field is zero.
            unsafe { GetExtendedUdpTable(target, bytes, 0, family as u32, UDP_TABLE_OWNER_PID, 0) }
        })?;
        if family == AF_INET {
            for row in rows::<MIB_TCPROW_OWNER_PID>(&tcp_buffer)? {
                result.push(tcp(
                    "IPv4",
                    Ipv4Addr::from(row.dwLocalAddr.to_ne_bytes()).to_string(),
                    row.dwLocalPort,
                    Ipv4Addr::from(row.dwRemoteAddr.to_ne_bytes()).to_string(),
                    row.dwRemotePort,
                    row.dwState,
                    row.dwOwningPid,
                ));
            }
            for row in rows::<MIB_UDPROW_OWNER_PID>(&udp_buffer)? {
                result.push(Connection {
                    family: "IPv4".into(),
                    protocol: "UDP".into(),
                    local_ip: Ipv4Addr::from(row.dwLocalAddr.to_ne_bytes()).to_string(),
                    local_port: port(row.dwLocalPort),
                    remote_ip: None,
                    remote_port: None,
                    status: "NONE".into(),
                    pid: row.dwOwningPid,
                });
            }
        } else {
            for row in rows::<MIB_TCP6ROW_OWNER_PID>(&tcp_buffer)? {
                result.push(tcp(
                    "IPv6",
                    Ipv6Addr::from(row.ucLocalAddr).to_string(),
                    row.dwLocalPort,
                    Ipv6Addr::from(row.ucRemoteAddr).to_string(),
                    row.dwRemotePort,
                    row.dwState,
                    row.dwOwningPid,
                ));
            }
            for row in rows::<MIB_UDP6ROW_OWNER_PID>(&udp_buffer)? {
                result.push(Connection {
                    family: "IPv6".into(),
                    protocol: "UDP".into(),
                    local_ip: Ipv6Addr::from(row.ucLocalAddr).to_string(),
                    local_port: port(row.dwLocalPort),
                    remote_ip: None,
                    remote_port: None,
                    status: "NONE".into(),
                    pid: row.dwOwningPid,
                });
            }
        }
    }
    result.sort();
    // psutil's Windows Python wrapper returns a set of normalized endpoint rows.
    // Duplicate raw UDP bindings are not distinct rows in that public contract.
    result.dedup();
    Ok(result)
}

fn system_dir() -> Result<PathBuf, Error> {
    let mut value = [0_u16; 32768];
    // SAFETY: value is a writable array with capacity matching the passed length.
    let count = unsafe { GetSystemDirectoryW(value.as_mut_ptr(), value.len() as u32) } as usize;
    if count == 0 || count >= value.len() {
        return Err(Error::Invalid(
            "Cannot obtain Windows system directory".into(),
        ));
    }
    Ok(PathBuf::from(String::from_utf16_lossy(&value[..count])))
}

#[repr(C)]
struct PcapIf {
    next: *mut PcapIf,
    name: *mut c_char,
    description: *mut c_char,
    addresses: *mut c_void,
    flags: u32,
}
type FindDevices = unsafe extern "C" fn(*mut *mut PcapIf, *mut c_char) -> c_int;
type FreeDevices = unsafe extern "C" fn(*mut PcapIf);

fn dll_devices(path: &std::path::Path) -> Result<Vec<String>, Error> {
    // SAFETY: absolute system Npcap path (or known system DLL in fault test), with
    // dependency search restricted to its directory and System32. No cwd/PATH search.
    let library = unsafe {
        Library::load_with_flags(
            path,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    }
    .map_err(|e| Error::Invalid(format!("Npcap DLL load failed: {e}")))?;
    // SAFETY: signatures and C ABI follow Npcap's exported pcap API; symbol lifetimes
    // stay within library. Missing exports return errors before any native call.
    let find = unsafe { library.get::<FindDevices>(b"pcap_findalldevs\0") }
        .map_err(|e| Error::Invalid(format!("Npcap enumeration export unavailable: {e}")))?;
    let free = unsafe { library.get::<FreeDevices>(b"pcap_freealldevs\0") }
        .map_err(|e| Error::Invalid(format!("Npcap cleanup export unavailable: {e}")))?;
    let mut list = ptr::null_mut();
    let mut error = [0_i8; 256];
    // SAFETY: both out pointers are initialized and writable; DLL remains loaded.
    let status = unsafe { find(&mut list, error.as_mut_ptr()) };
    if status != 0 {
        let message: Vec<u8> = error
            .iter()
            .take_while(|v| **v != 0)
            .map(|v| *v as u8)
            .collect();
        return Err(Error::Invalid(format!(
            "Npcap device enumeration failed: {}",
            String::from_utf8_lossy(&message)
        )));
    }
    // Always free the library-owned list, including traversal-bound failures.
    let result = (|| {
        let mut cursor = list;
        let mut devices = Vec::new();
        while !cursor.is_null() {
            if devices.len() >= MAX_ADAPTERS {
                return Err(Error::Invalid("Npcap device list exceeds bound".into()));
            }
            // SAFETY: library guarantees valid linked-list nodes until freealldevs.
            let value = unsafe { &*cursor };
            devices.push(native_text(value.name)?);
            cursor = value.next;
        }
        devices.sort();
        devices.dedup();
        Ok(devices)
    })();
    // SAFETY: exact list returned by find, freed once before unloading the library.
    unsafe { free(list) };
    result
}

fn native_text(item: *const c_char) -> Result<String, Error> {
    if item.is_null() {
        return Ok(String::new());
    }
    let mut chars = Vec::new();
    for i in 0..4096 {
        // SAFETY: Npcap ABI guarantees NUL-terminated strings; bound limits work.
        let value = unsafe { *item.add(i) };
        if value == 0 {
            return Ok(String::from_utf8_lossy(&chars).into_owned());
        }
        chars.push(value as u8);
    }
    Err(Error::Invalid("Npcap string exceeds bound".into()))
}

pub fn npcap(simulate_missing: bool) -> Result<CaptureCapability, Error> {
    if simulate_missing {
        return Ok(CaptureCapability::missing());
    }
    let dll = system_dir()?.join("Npcap").join("wpcap.dll");
    if !dll.is_file() {
        return Ok(CaptureCapability::missing());
    }
    let mut capability = CaptureCapability::missing();
    capability.dll = Some(dll.to_string_lossy().into_owned());
    match dll_devices(&dll) {
        Ok(devices) => {
            capability.packet_capture_available = true;
            capability.packet_capture_message =
                "Ready for packet inspection; capture permission is checked when you start.".into();
            capability.status = "available_enumeration_only".into();
            capability.devices = devices;
        }
        Err(error) => {
            capability.status = "unavailable".into();
            capability.packet_capture_message = "Npcap is installed but could not enumerate capture devices. Check driver and permissions.".into();
            capability.error = Some(error.to_string());
        }
    }
    Ok(capability)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    #[test]
    fn absent_driver_even_if_installed() {
        assert_eq!(npcap(true).unwrap().status, "not_installed");
    }
    #[test]
    fn missing_export_fails_without_os_substitution() {
        assert!(
            dll_devices(&system_dir().unwrap().join("kernel32.dll"))
                .unwrap_err()
                .to_string()
                .contains("export unavailable")
        );
    }
    #[test]
    fn oversized_or_invalid_table_is_rejected() {
        let mut buffer = Buffer::new(8).unwrap();
        buffer.words[0] = 100001;
        assert!(rows::<MIB_TCPROW_OWNER_PID>(&buffer).is_err());
        assert!(Buffer::new(MAX_BYTES + 1).is_err());
    }
    #[test]
    fn native_error_is_not_an_empty_success() {
        assert!(matches!(
            query("fake", 16, |_, _| 5),
            Err(Error::Windows { code: 5, .. })
        ));
        assert!(
            query("fake", 16, |_, bytes| {
                *bytes = 32;
                ERROR_INSUFFICIENT_BUFFER
            })
            .is_err()
        );
    }
    #[test]
    fn network_port_and_ipv4_byte_order() {
        assert_eq!(port(0xbb01), 443);
        assert_eq!(
            Ipv4Addr::from(0x0a0200c0_u32.to_ne_bytes()).to_string(),
            "192.0.2.10"
        );
    }
}
