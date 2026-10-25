//! Keep all native children inside the runtime's lifetime on Windows.
//!
//! The sole job handle is deliberately retained until process teardown. Closing
//! it in a Rust destructor would also terminate this process before normal exit.
//! Windows closes it on either clean exit or forced termination, retiring every
//! descendant even if Rust destructors cannot run. The handle is not inherited.
#[cfg(windows)]
pub fn own_descendants() -> Result<(), String> {
    use std::{io, mem::{size_of, zeroed}, ptr};
    use windows_sys::Win32::{Foundation::CloseHandle, System::{JobObjects::*, Threading::GetCurrentProcess}};
    // SAFETY: the SDK struct has the required layout; handles and sizes are
    // checked, and the process is assigned before it can launch any children.
    unsafe {
        let job = CreateJobObjectW(ptr::null(), ptr::null());
        if job.is_null() { return Err(io::Error::last_os_error().to_string()); }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if SetInformationJobObject(job, JobObjectExtendedLimitInformation,
            &info as *const _ as _, size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32) == 0
            || AssignProcessToJobObject(job, GetCurrentProcess()) == 0 {
            let error = io::Error::last_os_error().to_string();
            CloseHandle(job);
            return Err(format!("Cannot establish native process ownership: {error}"));
        }
        Ok(())
    }
}

#[cfg(not(windows))]
pub fn own_descendants() -> Result<(), String> { Ok(()) }
