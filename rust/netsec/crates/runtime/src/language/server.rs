//! One loopback `llama-server` child: verified pins, lazy start, GPU first with CPU fallback, idle unload.
//!
//! Nothing is executed before the background pass (started by `new`) has hashed the GGUF and every
//! server file against its pin. The child stays inside the runtime's KILL_ON_JOB_CLOSE job by inheritance
//! (`process_tree::own_descendants`); `stop`, idle unload and `Drop` also kill it explicitly.
pub use super::Device;
use crate::{application::{hash_file,read_json},assets::CUDA};
use serde_json::{Value,json};
use std::{collections::BTreeMap,io::{Read,Write},net::{Shutdown,SocketAddr,TcpListener,TcpStream},path::{Path,PathBuf},process::{Child,Command,Stdio},
    sync::{Arc,Condvar,Mutex,MutexGuard,Weak,atomic::{AtomicBool,AtomicUsize,Ordering::{Acquire,Release,SeqCst}},mpsc},thread,time::{Duration,Instant}};
const CPU:&str="artifacts/rust_migration/phase1/llm/native-b11564";
const HOST:&str="127.0.0.1";
/// Prompt token counts of one completion (Task 7 cache-miss row): `evaluated` = prompt_n and `cache_n` from the first event
/// carrying `timings`, `total` = tokens_evaluated from the final event; `None` when unknown (e.g. a verifier stop before the final event).
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct PromptTokens{pub evaluated:Option<u64>,pub total:Option<u64>,pub cache_n:Option<u64>}
impl PromptTokens {
    /// Tokens reused from the slot cache: cache_n when the server sent it, otherwise total - evaluated.
    pub fn cached(&self)->Option<u64>{self.cache_n.or_else(||Some(self.total?.checked_sub(self.evaluated?)?))}
    pub fn json(&self)->Value{json!({"evaluated":self.evaluated,"total":self.total,"cached":self.cached()})}
}
// `complete` runs on the caller's thread, so `explain::run` reads its counts here without changing `Generate`.
thread_local!{static PROMPT_TOKENS:std::cell::Cell<PromptTokens>=const{std::cell::Cell::new(PromptTokens{evaluated:None,total:None,cache_n:None})};}
pub(crate) fn record_prompt_tokens(tokens:PromptTokens){PROMPT_TOKENS.with(|c|c.set(tokens))}
/// The counts the last stream on this thread recorded; clears them.
pub fn take_prompt_tokens()->PromptTokens{PROMPT_TOKENS.with(|c|c.take())}
/// Streams completion tokens for one prompt; `on_token` returning `false` stops generation and closes the connection.
pub trait Generate {
    /// The device the next stream will run on, when known; `explain::prompt` bounds earlier answers on CPU (plan Task 7 Step 2).
    fn planned_device(&self)->Option<Device>{None}
    fn stream(&self,prompt:&str,n_predict:u32,cancel:&AtomicBool,on_token:&mut dyn FnMut(&str)->bool)->Result<Device,String>; }
/// One way to start the server; `{host}`/`{port}` in `args` are filled at spawn, and the API key goes in the child's environment.
#[derive(Clone)]
struct Launch{device:Device,program:PathBuf,args:Vec<String>,host:String,port:Option<u16>,health:Duration}
#[derive(Clone,Copy)]
struct Timing{first_token:Duration,total:Duration,idle:Duration,tick:Duration}
const LIVE:Timing=Timing{first_token:Duration::from_secs(15),total:Duration::from_secs(60),idle:Duration::from_secs(600),tick:Duration::from_millis(100)};
enum Integrity{Pending,Ready(Vec<Launch>),Failed(String)}
struct Running{child:Child,device:Device,port:u16,key:String,healthy:bool}
impl Drop for Running{fn drop(&mut self){let _=self.child.kill();let _=self.child.wait();}}
struct Inner{integrity:Mutex<Integrity>,verified:Condvar,running:Mutex<Option<Running>>,starting:Mutex<()>,timing:Timing,spawns:AtomicUsize,active:AtomicUsize,last_use:Mutex<Instant>,closed:AtomicBool,warming:AtomicBool}
pub struct LlamaServer{inner:Arc<Inner>}
fn lock<T>(m:&Mutex<T>)->MutexGuard<'_,T>{m.lock().unwrap_or_else(|e|e.into_inner())}
impl LlamaServer {
    /// Starts the one-per-process background hash pass; the server itself starts lazily (or on `warm`).
    pub fn new(root:PathBuf)->Self{
        let server=Self::scripted(Integrity::Pending,LIVE);let weak=Arc::downgrade(&server.inner);
        thread::spawn(move||{let result=verify_pins(&root);if let Some(inner)=weak.upgrade(){inner.set(match result{Ok(l)=>Integrity::Ready(l),Err(e)=>Integrity::Failed(e)});}});
        server
    }
    fn scripted(integrity:Integrity,timing:Timing)->Self{
        let inner=Arc::new(Inner{integrity:Mutex::new(integrity),verified:Condvar::new(),running:Mutex::new(None),starting:Mutex::new(()),timing,spawns:AtomicUsize::new(0),active:AtomicUsize::new(0),last_use:Mutex::new(Instant::now()),closed:AtomicBool::new(false),warming:AtomicBool::new(false)});
        let weak=Arc::downgrade(&inner);thread::spawn(move||idle_watcher(weak,timing.tick));
        LlamaServer{inner}
    }
    /// A server that never starts a child (its hash pass failed); for tests above the language layer.
    #[cfg(test)]pub(crate) fn unverified()->Self{Self::scripted(Integrity::Failed("test server".into()),LIVE)}
    #[cfg(test)]fn set_integrity(&self,integrity:Integrity){self.inner.set(integrity)}
    #[cfg(test)]fn spawns(&self)->usize{self.inner.spawns.load(SeqCst)}
    #[cfg(test)]fn pid(&self)->Option<u32>{lock(&self.inner.running).as_ref().map(|r|r.child.id())}
    /// Waits for the background hash pass; true only when it succeeded.
    pub fn wait_verified(&self,timeout:Duration)->bool{matches!(*self.inner.wait_verified(timeout),Integrity::Ready(_))}
    /// Starts the server in the background (when an explanation context is created) so the first answer finds it warm.
    /// Calls while one warm thread waits are coalesced, and nothing starts after `shutdown`.
    pub fn warm(&self){
        if self.inner.closed.load(Acquire)||self.inner.warming.swap(true,SeqCst){return;}
        let inner=self.inner.clone();thread::spawn(move||{drop(inner.wait_verified(Duration::from_secs(900)));if !inner.closed.load(Acquire){*lock(&inner.last_use)=Instant::now();let _=inner.start();}inner.warming.store(false,SeqCst);});
    }
    pub fn stop(&self){self.inner.stop()}
    /// Terminal `stop` for app exit: a waiting warm thread is released and no later warm, stream or start launches a child.
    pub fn shutdown(&self){self.inner.closed.store(true,Release);self.inner.verified.notify_all();self.inner.stop();}
    /// The device of the healthy running child, if any.
    pub fn device(&self)->Option<Device>{lock(&self.inner.running).as_mut().and_then(|r|(r.healthy&&matches!(r.child.try_wait(),Ok(None))).then_some(r.device))}
    /// `language::status` plus the background hash result; a failed pass is an integrity error, never a silent fallback.
    pub fn status(&self,root:&Path)->Value{
        let mut status=super::status(root,self.device());
        match &*lock(&self.inner.integrity){
            Integrity::Pending=>status["integrity"]=json!("verifying"),
            Integrity::Ready(_)=>status["integrity"]=json!("verified"),
            Integrity::Failed(e)=>{status["integrity"]=json!("failed");status["status"]=json!("unavailable");status["reason"]=json!(format!("Language model integrity check failed: {e}"));}
        }
        status
    }
}
impl Drop for LlamaServer{fn drop(&mut self){self.inner.closed.store(true,Release);self.inner.verified.notify_all();self.inner.stop();}}
fn idle_watcher(weak:Weak<Inner>,tick:Duration){loop{
    thread::sleep(tick);let Some(inner)=weak.upgrade() else{return};if inner.closed.load(Acquire){return;}
    // Checked under the child lock: a stream marks itself active before it reads the child, so a busy child is never unloaded.
    let mut running=lock(&inner.running);if running.is_some()&&inner.active.load(SeqCst)==0&&lock(&inner.last_use).elapsed()>=inner.timing.idle{*running=None;}
}}
impl Inner {
    fn set(&self,integrity:Integrity){*lock(&self.integrity)=integrity;self.verified.notify_all();}
    fn wait_verified(&self,timeout:Duration)->MutexGuard<'_,Integrity>{
        let deadline=Instant::now()+timeout;let mut integrity=lock(&self.integrity);
        while matches!(*integrity,Integrity::Pending)&&!self.closed.load(Acquire){let left=deadline.saturating_duration_since(Instant::now());if left.is_zero(){break;}integrity=self.verified.wait_timeout(integrity,left).unwrap_or_else(|e|e.into_inner()).0;}
        integrity
    }
    fn stop(&self){lock(&self.running).take();}
    /// Reuses a healthy child or starts one, GPU first; nothing runs before the background pass has succeeded.
    fn start(&self)->Result<(Device,u16,String),String>{
        let _starting=lock(&self.starting);
        if self.closed.load(Acquire){return Err("Language server is shut down".into());}
        {let mut running=lock(&self.running);if let Some(r)=running.as_mut(){if r.healthy&&matches!(r.child.try_wait(),Ok(None)){return Ok((r.device,r.port,r.key.clone()));}}*running=None;}
        let launches=match &*lock(&self.integrity){
            Integrity::Pending=>return Err("Language model files are still being verified".into()),
            Integrity::Failed(e)=>return Err(format!("Language model integrity check failed: {e}")),
            Integrity::Ready(launches)=>launches.clone()};
        let mut errors=Vec::new();
        for launch in &launches{match self.launch(launch){Ok(started)=>return Ok(started),Err(e)=>errors.push(format!("{}: {e}",launch.device.as_str()))}}
        Err(format!("llama-server did not start ({})",errors.join("; ")))
    }
    fn launch(&self,launch:&Launch)->Result<(Device,u16,String),String>{
        if launch.host!=HOST{return Err(format!("llama-server may bind only to {HOST}, not {}",launch.host));}
        let port=match launch.port{Some(port)=>port,None=>free_port()?};let key=session_key()?;
        let mut command=Command::new(&launch.program);
        command.args(launch.args.iter().map(|a|a.replace("{host}",&launch.host).replace("{port}",&port.to_string()))).env("LLAMA_API_KEY",&key).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        // Inherited LLAMA_ARG_* variables would silently change the reviewed arguments, and GGML_* ones (GGML_BACKEND_PATH) load unpinned backends.
        for (name,_) in std::env::vars_os(){let upper=name.to_string_lossy().to_ascii_uppercase();if upper.starts_with("LLAMA_ARG_")||upper.starts_with("GGML_"){command.env_remove(name);}}
        // ggml also searches the working directory for backends, so the child runs in its own pinned folder.
        if let Some(folder)=launch.program.parent().filter(|f|!f.as_os_str().is_empty()){command.current_dir(folder);}
        #[cfg(windows)]{use std::os::windows::process::CommandExt;const CREATE_NO_WINDOW:u32=0x0800_0000;command.creation_flags(CREATE_NO_WINDOW);}
        self.spawns.fetch_add(1,SeqCst);
        let child=command.spawn().map_err(|e|format!("cannot start: {e}"))?;
        *lock(&self.running)=Some(Running{child,device:launch.device,port,key:key.clone(),healthy:false});
        let started=Instant::now();
        loop{
            {let mut running=lock(&self.running);let Some(r)=running.as_mut() else{return Err("stopped while starting".into())};
                if self.closed.load(Acquire){*running=None;return Err("stopped while starting".into());}
                if let Ok(Some(status))=r.child.try_wait(){*running=None;return Err(format!("exited early ({status})"));}}
            if healthy(port){if let Some(r)=lock(&self.running).as_mut(){r.healthy=true;return Ok((launch.device,port,key));}return Err("stopped while starting".into());}
            if started.elapsed()>=launch.health{lock(&self.running).take();return Err(format!("not healthy within {} s",launch.health.as_secs_f32()));}
            thread::sleep(self.timing.tick.min(Duration::from_millis(100)));
        }
    }
    fn complete(&self,port:u16,key:&str,prompt:&str,n_predict:u32,cancel:&AtomicBool,on_token:&mut dyn FnMut(&str)->bool,requested:Instant)->Result<(),String>{
        let body=json!({"prompt":prompt,"n_predict":n_predict,"temperature":0.2,"stream":true,"cache_prompt":true,"timings_per_token":true,"stop":["<|im_end|>"]}).to_string();
        let mut tokens=PromptTokens::default();record_prompt_tokens(tokens);
        let mut socket=TcpStream::connect_timeout(&SocketAddr::from(([127,0,0,1],port)),Duration::from_secs(2)).map_err(|e|{self.stop();format!("llama-server is not reachable: {e}")})?;
        let _=socket.set_nodelay(true);
        socket.write_all(format!("POST /completion HTTP/1.1\r\nHost: {HOST}:{port}\r\nAuthorization: Bearer {key}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes()).map_err(|e|e.to_string())?;
        // A reader thread keeps reads blocking; this thread waits in ticks so cancel and both deadlines are checked even while the server is silent.
        let reader=socket.try_clone().map_err(|e|e.to_string())?;let closer=Close(socket);let (tx,rx)=mpsc::channel::<Result<Vec<u8>,String>>();
        thread::spawn(move||{let mut reader=reader;let mut buffer=[0u8;8192];loop{match reader.read(&mut buffer){
            Ok(0)=>{let _=tx.send(Ok(Vec::new()));return;}Ok(n)=>if tx.send(Ok(buffer[..n].to_vec())).is_err(){return;},Err(e)=>{let _=tx.send(Err(e.to_string()));return;}}}});
        let (timing,mut sse,mut first)=(self.timing,Sse::default(),false);
        loop{
            if cancel.load(Acquire){return Err("cancelled".into());}
            let elapsed=requested.elapsed();
            if !first&&elapsed>=timing.first_token{return Err("first token deadline".into());}
            if elapsed>=timing.total{return Err("answer deadline".into());}
            let limit=if first{timing.total}else{timing.first_token};
            match rx.recv_timeout(timing.tick.min(limit-elapsed)){
                Ok(Ok(bytes)) if bytes.is_empty()=>return Err("llama-server closed the stream early".into()),
                Ok(Ok(bytes))=>for event in sse.feed(&bytes)?{
                    if let Some(error)=event.get("error"){return Err(format!("llama-server error: {error}"));}
                    if tokens.evaluated.is_none()&&event["timings"].is_object(){tokens.evaluated=event["timings"]["prompt_n"].as_u64();tokens.cache_n=event["timings"]["cache_n"].as_u64();record_prompt_tokens(tokens);}
                    if event["stop"]==true{tokens.total=event["tokens_evaluated"].as_u64();record_prompt_tokens(tokens);}
                    let content=event["content"].as_str().unwrap_or("");
                    if !content.is_empty(){
                        if cancel.load(Acquire){return Err("cancelled".into());}
                        first=true;if !on_token(content){drop(closer);return Ok(());}
                    }
                    if event["stop"]==true{return Ok(());}
                },
                Ok(Err(e))=>return Err(format!("llama-server stream failed: {e}")),
                Err(mpsc::RecvTimeoutError::Timeout)=>continue,
                Err(mpsc::RecvTimeoutError::Disconnected)=>return Err("llama-server stream ended".into()),
            }
        }
    }
}
impl Generate for LlamaServer {
    /// The healthy running child's device, else the first pinned launch (manifest/`NETSEC_LLM_DEVICE` order), never the child read alone.
    fn planned_device(&self)->Option<Device>{self.device().or_else(||match &*lock(&self.inner.integrity){Integrity::Ready(launches)=>launches.first().map(|l|l.device),_=>None})}
    fn stream(&self,prompt:&str,n_predict:u32,cancel:&AtomicBool,on_token:&mut dyn FnMut(&str)->bool)->Result<Device,String>{
        let (inner,requested)=(&self.inner,Instant::now());
        inner.active.fetch_add(1,SeqCst);*lock(&inner.last_use)=requested;
        let result=self.ready(cancel,requested).and_then(|(device,port,key)|inner.complete(port,&key,prompt,n_predict,cancel,on_token,requested).map(|_|device));
        *lock(&inner.last_use)=Instant::now();inner.active.fetch_sub(1,SeqCst);
        result
    }
}
impl LlamaServer {
    /// Starts (or reuses) the child on its own thread and waits only until the first-token deadline or a cancel.
    /// A launch that outlives the request keeps going, so the next request finds the server warm.
    fn ready(&self,cancel:&AtomicBool,requested:Instant)->Result<(Device,u16,String),String>{
        let (inner,timing)=(self.inner.clone(),self.inner.timing);let (tx,rx)=mpsc::channel();thread::spawn(move||{let _=tx.send(inner.start());});
        loop{
            if cancel.load(Acquire){return Err("cancelled".into());}
            let elapsed=requested.elapsed();if elapsed>=timing.first_token{return Err("first token deadline".into());}
            match rx.recv_timeout(timing.tick.min(timing.first_token-elapsed)){
                Ok(started)=>return started,
                Err(mpsc::RecvTimeoutError::Timeout)=>continue,
                Err(mpsc::RecvTimeoutError::Disconnected)=>return Err("llama-server start failed".into()),
            }
        }
    }
}
/// Shuts the connection both ways when dropped, which also ends the reader thread and tells llama-server to stop generating.
struct Close(TcpStream);
impl Drop for Close{fn drop(&mut self){let _=self.0.shutdown(Shutdown::Both);}}
/// Incremental HTTP/1.1 response + SSE parser (chunked or plain body); yields each `data: {...}` event.
#[derive(Default)]
struct Sse{raw:Vec<u8>,head:Option<bool>,text:Vec<u8>}
impl Sse {
    fn feed(&mut self,bytes:&[u8])->Result<Vec<Value>,String>{
        self.raw.extend_from_slice(bytes);
        if self.head.is_none(){
            let Some(end)=self.raw.windows(4).position(|w|w==b"\r\n\r\n") else{return if self.raw.len()>65536{Err("llama-server response head too large".into())}else{Ok(Vec::new())}};
            let head=String::from_utf8_lossy(&self.raw[..end]).to_ascii_lowercase();self.raw.drain(..end+4);
            let status=head.lines().next().unwrap_or("").to_string();
            if !status.starts_with("http/1.1 200")&&!status.starts_with("http/1.0 200"){return Err(format!("llama-server answered {}",status.trim()));}
            self.head=Some(head.lines().any(|l|l.starts_with("transfer-encoding:")&&l.contains("chunked")));
        }
        if self.head==Some(true){loop{
            let Some(line)=self.raw.windows(2).position(|w|w==b"\r\n") else{break};
            let size=usize::from_str_radix(String::from_utf8_lossy(&self.raw[..line]).split(';').next().unwrap_or("").trim(),16).map_err(|_|"llama-server sent a malformed chunk".to_string())?;
            if size==0{self.raw.clear();break;}
            if self.raw.len()<line+2+size+2{break;}
            self.text.extend_from_slice(&self.raw[line+2..line+2+size]);self.raw.drain(..line+2+size+2);
        }}else{self.text.append(&mut self.raw);}
        let mut events=Vec::new();
        while let Some(end)=self.text.iter().position(|b|*b==b'\n'){
            let line=String::from_utf8_lossy(&self.text[..end]).trim().to_string();self.text.drain(..=end);
            if let Some(data)=line.strip_prefix("data:"){events.push(serde_json::from_str(data.trim()).map_err(|_|"llama-server sent a malformed event".to_string())?);}
            else if let Some(error)=line.strip_prefix("error:"){return Err(format!("llama-server error: {}",error.trim()));}
        }
        Ok(events)
    }
}
fn free_port()->Result<u16,String>{TcpListener::bind((HOST,0)).and_then(|l|l.local_addr()).map(|a|a.port()).map_err(|e|e.to_string())}
fn session_key()->Result<String,String>{let mut key=[0u8;16];getrandom::fill(&mut key).map_err(|e|e.to_string())?;Ok(key.iter().map(|v|format!("{v:02x}")).collect())}
fn healthy(port:u16)->bool{
    let Ok(mut socket)=TcpStream::connect_timeout(&SocketAddr::from(([127,0,0,1],port)),Duration::from_millis(250)) else{return false};
    let _=socket.set_read_timeout(Some(Duration::from_secs(1)));let _=socket.set_write_timeout(Some(Duration::from_secs(1)));
    let mut status=[0u8;12];
    socket.write_all(format!("GET /health HTTP/1.1\r\nHost: {HOST}:{port}\r\nConnection: close\r\n\r\n").as_bytes()).is_ok()&&socket.read_exact(&mut status).is_ok()&&status.ends_with(b" 200")
}
/// The one background pass: hashes the GGUF (against model.json and any sealed digest) and every server file
/// against its pin. The CPU closure is pinned by the runtime_assets.json seal; the CUDA closure by its sealed
/// "background" digests or, in a dev tree, by release.json. An unpinned closure is unavailable; a pinned file that differs fails the pass.
fn verify_pins(root:&Path)->Result<Vec<Launch>,String>{
    let model=super::installed(root)?;let gguf=model.path(root)?;
    let sealed:BTreeMap<String,String>=read_json(&root.join("runtime_assets.json"),1024*1024).map(|m|m["files"].as_array().into_iter().flatten()
        .filter_map(|e|Some((e["path"].as_str()?.to_string(),e["sha256"].as_str()?.to_string()))).collect()).unwrap_or_default();
    let digest=hash_file(&gguf).map_err(|e|format!("Language model cannot be read: {e}"))?;
    if digest!=model.sha256{return Err("Language model differs from its pinned digest in artifacts/llm/model.json".into());}
    if sealed.get(&format!("artifacts/llm/{}",model.file)).is_some_and(|s|*s!=digest){return Err("Language model differs from its sealed package digest".into());}
    let gguf=gguf.to_string_lossy().into_owned();
    let cache=cache_ram(std::env::var("NETSEC_LLM_CACHE_RAM").ok().as_deref()).to_string();
    let args=|extra:&[&str]|["-m",&gguf,"--host","{host}","--port","{port}","-c","4096","-np","1","--cache-ram",&cache].into_iter().chain(extra.iter().copied()).chain(["--no-webui","--offline"]).map(String::from).collect::<Vec<_>>();
    let mut launches=Vec::new();
    // `--device CUDA0` makes a CUDA build without a usable GPU fail (and fall back) instead of running on CPU labelled as GPU.
    if model.device()==Device::Gpu{if let Some(program)=closure(root,CUDA,&sealed,true)?{launches.push(Launch{device:Device::Gpu,program,args:args(&["-ngl","99","--device","CUDA0"]),host:HOST.into(),port:None,health:Duration::from_secs(8)});}}
    if let Some(program)=closure(root,CPU,&sealed,false)?{launches.push(Launch{device:Device::Cpu,program,args:args(&["-ngl","0","-t","8"]),host:HOST.into(),port:None,health:Duration::from_secs(20)});}
    if launches.is_empty(){return Err("No pinned llama-server runtime is installed".into());}
    Ok(launches)
}
/// Host-RAM cap (MiB) for llama-server's prompt cache; b11564 defaults to 8192 MiB, too much beside training jobs (Task 7 ruling 3).
const CACHE_RAM_MIB:u32=1024;
/// `NETSEC_LLM_CACHE_RAM` may only lower the cap: an integer 0..=1024 (0 turns the cache off); any other value is ignored.
fn cache_ram(value:Option<&str>)->u32{value.and_then(|v|v.trim().parse::<u32>().ok()).filter(|v|*v<=CACHE_RAM_MIB).unwrap_or(CACHE_RAM_MIB)}
/// The other llama.cpp tools' entry DLLs in the b11564 folders: nothing in llama-server imports or enumerates them.
const OTHER_TOOLS:[&str;7]=["llama-batched-bench-impl.dll","llama-bench-impl.dll","llama-cli-impl.dll","llama-completion-impl.dll","llama-fit-params-impl.dll","llama-perplexity-impl.dll","llama-quantize-impl.dll"];
/// Verifies `llama-server.exe` and every DLL beside it except `OTHER_TOOLS`: ggml's backend loader opens every
/// `ggml-<backend>-*.dll` in the folder whatever its suffix, so any other DLL there can run inside the server.
fn closure(root:&Path,folder:&str,sealed:&BTreeMap<String,String>,release:bool)->Result<Option<PathBuf>,String>{
    let directory=root.join(folder);let server=directory.join("llama-server.exe");if !server.is_file(){return Ok(None);}
    let prefix=format!("{folder}/");
    let mut pins:BTreeMap<String,String>=sealed.iter().filter_map(|(p,d)|p.strip_prefix(&prefix).filter(|n|!n.contains('/')).map(|n|(n.to_string(),d.clone()))).collect();
    if pins.is_empty()&&release{if let Ok(manifest)=read_json(&directory.join("release.json"),1024*1024){
        pins=manifest["files"].as_object().into_iter().flatten().filter_map(|(n,d)|Some((n.clone(),d.as_str()?.to_string()))).collect();}}
    // An unpinned server means the closure is not installed for this device; once it is pinned, every file it can load must be too.
    if !pins.contains_key("llama-server.exe"){return Ok(None);}
    let mut names=Vec::new();
    for entry in std::fs::read_dir(&directory).map_err(|e|e.to_string())?{
        let name=entry.map_err(|e|e.to_string())?.file_name().to_string_lossy().into_owned();let lower=name.to_ascii_lowercase();
        // A `.local` file or folder redirects the server's DLL loads away from the pinned files.
        if lower.ends_with(".local"){return Err(format!("Language runtime folder holds a DLL redirection entry: {folder}/{name}"));}
        if lower=="llama-server.exe"||lower.ends_with(".manifest")||(lower.ends_with(".dll")&&!OTHER_TOOLS.contains(&lower.as_str())){names.push(name);}
    }
    for name in &names{
        let Some(pin)=pins.get(name) else{return Err(format!("Language runtime file has no pin: {folder}/{name}"))};
        if hash_file(&directory.join(name))?!=*pin{return Err(format!("Language runtime file differs from its pin: {folder}/{name}"));}
    }
    Ok(Some(server))
}
#[cfg(all(test,windows))]mod tests{
    use super::*;
    use crate::application::hash_file;
    use serde_json::json;
    use std::{io::{Read,Write},net::TcpListener,path::Path,sync::{atomic::Ordering,mpsc},thread,time::Instant};
    const QUICK:Timing=Timing{first_token:Duration::from_secs(5),total:Duration::from_secs(10),idle:Duration::from_secs(600),tick:Duration::from_millis(20)};
    /// Windows timer granularity (about 16 ms) plus thread scheduling, allowed on top of a deadline.
    const SLACK:Duration=Duration::from_millis(150);
    /// A scripted fake `llama-server`: `/health` is 200, `/completion` streams `(delay ms, data JSON)` events as chunked SSE.
    fn fake(events:Vec<(u64,serde_json::Value)>)->(u16,mpsc::Receiver<&'static str>){fake_after(Duration::ZERO,events)}
    /// As `fake`, but `/health` answers 503 until `ready` has passed (a model still loading).
    fn fake_after(ready:Duration,events:Vec<(u64,serde_json::Value)>)->(u16,mpsc::Receiver<&'static str>){
        let listener=TcpListener::bind("127.0.0.1:0").unwrap();let port=listener.local_addr().unwrap().port();let (tx,rx)=mpsc::channel();let born=Instant::now();
        thread::spawn(move||for socket in listener.incoming(){let Ok(mut socket)=socket else{return};let events=events.clone();let tx=tx.clone();thread::spawn(move||{
            let mut head=Vec::new();let mut byte=[0u8;1];while !head.ends_with(b"\r\n\r\n"){if socket.read(&mut byte).unwrap_or(0)==0{return;}head.push(byte[0]);}
            let head=String::from_utf8_lossy(&head).to_string();
            if head.starts_with("GET /health"){let _=socket.write_all(if born.elapsed()>=ready{b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}" as &[u8]}else{b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"});return;}
            let length=head.lines().find_map(|l|l.to_ascii_lowercase().strip_prefix("content-length:").map(|v|v.trim().parse::<usize>().unwrap())).unwrap_or(0);
            let mut body=vec![0u8;length];socket.read_exact(&mut body).unwrap();let body:serde_json::Value=serde_json::from_slice(&body).unwrap();
            assert!(head.starts_with("POST /completion")&&head.contains("Authorization: Bearer ")&&body["stream"]==true&&body["cache_prompt"]==true&&body["temperature"]==0.2&&body["stop"]==json!(["<|im_end|>"])&&body["timings_per_token"]==true,"{head}{body}");
            if socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n").is_err(){let _=tx.send("closed");return;}
            for (delay,event) in events{thread::sleep(Duration::from_millis(delay));let line=format!("data: {event}\n\n");
                if socket.write_all(format!("{:x}\r\n{line}\r\n",line.len()).as_bytes()).and_then(|_|socket.flush()).is_err(){let _=tx.send("closed");return;}}
            let _=socket.write_all(b"0\r\n\r\n");socket.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
            let _=tx.send(match socket.read(&mut byte){Ok(0)=>"closed",Err(e) if !matches!(e.kind(),std::io::ErrorKind::WouldBlock|std::io::ErrorKind::TimedOut)=>"closed",_=>"open"});
        });});
        (port,rx)
    }
    fn token(t:&str)->(u64,serde_json::Value){(60,json!({"content":t,"stop":false}))}
    fn end()->(u64,serde_json::Value){(20,json!({"content":"","stop":true}))}
    /// A harmless long-lived child stands in for the server process; health is answered by the fake on `port`.
    fn idle_child(device:Device,port:u16)->Launch{Launch{device,program:PathBuf::from(std::env::var("SystemRoot").unwrap()).join("System32/PING.EXE"),args:vec!["-n".into(),"120".into(),"127.0.0.1".into()],host:"127.0.0.1".into(),port:Some(port),health:Duration::from_secs(5)}}
    fn failing_gpu()->Launch{Launch{device:Device::Gpu,program:"cmd.exe".into(),args:vec!["/c".into(),"exit".into(),"1".into()],host:"127.0.0.1".into(),port:None,health:Duration::from_secs(5)}}
    fn alive(pid:u32)->bool{let out=std::process::Command::new("tasklist").args(["/FI",&format!("PID eq {pid}"),"/NH","/FO","CSV"]).output().unwrap();String::from_utf8_lossy(&out.stdout).contains(&format!("\"{pid}\""))}
    fn collect(server:&LlamaServer,cancel:&AtomicBool,keep:usize)->(Result<Device,String>,Vec<String>){let mut seen=Vec::new();let r=server.stream("p",8,cancel,&mut |t|{seen.push(t.to_string());seen.len()<keep});(r,seen)}
    #[test]fn stream_delivers_tokens_and_honours_stop(){
        let (port,closed)=fake(vec![token("a"),token("b"),token("c"),end()]);
        let server=LlamaServer::scripted(Integrity::Ready(vec![idle_child(Device::Cpu,port)]),QUICK);
        let (r,seen)=collect(&server,&AtomicBool::new(false),usize::MAX);assert_eq!(r,Ok(Device::Cpu));assert_eq!(seen,["a","b","c"]);assert_eq!(server.device(),Some(Device::Cpu));
        let (port,closed2)=fake(vec![token("a"),(400,json!({"content":"b","stop":false})),(400,json!({"content":"c","stop":false})),end()]);drop(closed);
        let server=LlamaServer::scripted(Integrity::Ready(vec![idle_child(Device::Cpu,port)]),QUICK);
        let (r,seen)=collect(&server,&AtomicBool::new(false),1);assert_eq!(r,Ok(Device::Cpu));assert_eq!(seen,["a"]);
        assert_eq!(closed2.recv_timeout(Duration::from_secs(5)),Ok("closed"),"on_token=false must close the connection");
    }
    /// Task 7 cache-miss row: the first event with `timings` gives prompt_n/cache_n, the final event gives tokens_evaluated.
    #[test]fn stream_reports_prompt_token_counts(){
        let timed=|content:&str,timings:serde_json::Value|(60,json!({"content":content,"stop":false,"timings":timings}));
        let (port,_)=fake(vec![timed("a",json!({"prompt_n":412,"cache_n":388,"predicted_n":1})),timed("b",json!({"prompt_n":999,"cache_n":0,"predicted_n":2})),(20,json!({"content":"","stop":true,"tokens_evaluated":800}))]);
        let server=LlamaServer::scripted(Integrity::Ready(vec![idle_child(Device::Cpu,port)]),QUICK);record_prompt_tokens(PromptTokens{evaluated:Some(1),total:Some(1),cache_n:Some(1)});
        assert_eq!(collect(&server,&AtomicBool::new(false),usize::MAX).0,Ok(Device::Cpu));
        let t=take_prompt_tokens();assert_eq!(t,PromptTokens{evaluated:Some(412),total:Some(800),cache_n:Some(388)});assert_eq!(t.cached(),Some(388));
        assert_eq!(take_prompt_tokens(),PromptTokens::default(),"take clears the counts");
        // Without cache_n the cached count is total - evaluated.
        let (port,_)=fake(vec![timed("a",json!({"prompt_n":300})),(20,json!({"content":"","stop":true,"tokens_evaluated":1000}))]);
        let server=LlamaServer::scripted(Integrity::Ready(vec![idle_child(Device::Cpu,port)]),QUICK);
        assert!(collect(&server,&AtomicBool::new(false),usize::MAX).0.is_ok());let t=take_prompt_tokens();assert_eq!((t.evaluated,t.total,t.cached()),(Some(300),Some(1000),Some(700)));
        // A verifier stop closes the stream before the final event: the total (and so a derived cached count) is unknown.
        let (port,_)=fake(vec![timed("a",json!({"prompt_n":412})),token("b"),end()]);
        let server=LlamaServer::scripted(Integrity::Ready(vec![idle_child(Device::Cpu,port)]),QUICK);
        assert!(collect(&server,&AtomicBool::new(false),1).0.is_ok());let t=take_prompt_tokens();
        assert_eq!(t,PromptTokens{evaluated:Some(412),total:None,cache_n:None});assert_eq!(t.cached(),None);
        assert_eq!(t.json(),json!({"evaluated":412,"total":null,"cached":null}));
    }
    #[test]fn gpu_start_failure_falls_back_to_cpu(){
        let (port,_)=fake(vec![token("ok"),end()]);
        let server=LlamaServer::scripted(Integrity::Ready(vec![failing_gpu(),idle_child(Device::Cpu,port)]),QUICK);
        assert_eq!(server.planned_device(),Some(Device::Gpu),"with no child the first pinned launch is planned");
        let (r,seen)=collect(&server,&AtomicBool::new(false),usize::MAX);assert_eq!(r,Ok(Device::Cpu));assert_eq!(seen,["ok"]);assert_eq!(server.device(),Some(Device::Cpu));assert_eq!(server.spawns(),2);
        assert_eq!(server.planned_device(),Some(Device::Cpu),"after a fallback the running CPU child answers");
        assert_eq!(LlamaServer::scripted(Integrity::Pending,QUICK).planned_device(),None);
    }
    #[test]fn deadline_and_cancel_abort_stream(){
        let (port,_)=fake(vec![token("a"),token("b"),token("c"),token("d"),end()]);
        let server=LlamaServer::scripted(Integrity::Ready(vec![idle_child(Device::Cpu,port)]),QUICK);let cancel=AtomicBool::new(false);
        let mut seen=0;let r=server.stream("p",8,&cancel,&mut |_|{seen+=1;cancel.store(true,Ordering::Release);true});assert!(r.as_ref().is_err_and(|e|e.contains("cancelled")),"{r:?}");assert_eq!(seen,1);
        let (port,_)=fake(vec![(1500,json!({"content":"late","stop":false})),end()]);
        let server=LlamaServer::scripted(Integrity::Ready(vec![idle_child(Device::Cpu,port)]),Timing{first_token:Duration::from_millis(400),..QUICK});
        let started=Instant::now();let (r,seen)=collect(&server,&AtomicBool::new(false),usize::MAX);
        assert!(r.as_ref().is_err_and(|e|e.contains("first token deadline")),"{r:?}");assert!(seen.is_empty()&&started.elapsed()<Duration::from_millis(1400));
        let (port,_)=fake((0..40).map(|i|token(&i.to_string())).collect());
        let server=LlamaServer::scripted(Integrity::Ready(vec![idle_child(Device::Cpu,port)]),Timing{total:Duration::from_millis(700),..QUICK});
        let (r,seen)=collect(&server,&AtomicBool::new(false),usize::MAX);assert!(r.as_ref().is_err_and(|e|e.contains("answer deadline")),"{r:?}");assert!(!seen.is_empty()&&seen.len()<40);
    }
    #[test]fn slow_start_honours_first_token_deadline_and_cancel(){
        // The GPU child fails health slowly and the CPU child is healthy only after the first-token deadline (a cold CPU fallback).
        let (port,_)=fake_after(Duration::from_millis(1300),vec![token("a"),end()]);
        let gpu=Launch{health:Duration::from_millis(700),..idle_child(Device::Gpu,free_port().unwrap())};
        let timing=Timing{first_token:Duration::from_millis(400),..QUICK};
        let server=LlamaServer::scripted(Integrity::Ready(vec![gpu,idle_child(Device::Cpu,port)]),timing);
        let started=Instant::now();let (r,seen)=collect(&server,&AtomicBool::new(false),usize::MAX);let waited=started.elapsed();
        assert!(r.as_ref().is_err_and(|e|e.contains("first token deadline")),"{r:?}");assert!(seen.is_empty());
        assert!(waited<timing.first_token+timing.tick+SLACK,"the request waited {waited:?} for a start that outlived its deadline");
        // The launch keeps going, so the next request finds the CPU server warm.
        while server.device().is_none()&&started.elapsed()<Duration::from_secs(5){thread::sleep(timing.tick);}
        assert_eq!(server.device(),Some(Device::Cpu));assert_eq!(collect(&server,&AtomicBool::new(false),usize::MAX).0,Ok(Device::Cpu));assert_eq!(server.spawns(),2);
        // A cancel (chat clear) during the start returns promptly instead of waiting for health.
        let (port,_)=fake_after(Duration::from_secs(3),vec![token("a"),end()]);
        let server=LlamaServer::scripted(Integrity::Ready(vec![idle_child(Device::Cpu,port)]),QUICK);let cancel=Arc::new(AtomicBool::new(false));
        let flag=cancel.clone();thread::spawn(move||{thread::sleep(Duration::from_millis(200));flag.store(true,Ordering::Release);});
        let started=Instant::now();let (r,_)=collect(&server,&cancel,usize::MAX);let waited=started.elapsed();
        assert!(r.as_ref().is_err_and(|e|e.contains("cancelled")),"{r:?}");assert!(waited<Duration::from_millis(200)+QUICK.tick+SLACK,"cancel took {waited:?}");
    }
    #[test]fn launch_runs_in_server_folder_without_inherited_ggml_env(){
        // ggml also searches the working directory and GGML_BACKEND_PATH for backends, so neither may come from the runtime.
        let root=scratch();std::env::set_var("GGML_BACKEND_PATH",root.join("planted.dll"));
        let program=PathBuf::from(std::env::var("SystemRoot").unwrap()).join("System32").join("cmd.exe");let probe=root.join("probe.txt");let out=probe.to_string_lossy().into_owned();
        let launch=Launch{device:Device::Cpu,program:program.clone(),args:["/c","cd",">",&out,"&","set","GGML",">>",&out,"&","set","SystemRoot",">>",&out].map(String::from).to_vec(),host:"127.0.0.1".into(),port:Some(free_port().unwrap()),health:Duration::from_secs(5)};
        let server=LlamaServer::scripted(Integrity::Ready(vec![launch]),QUICK);let (r,_)=collect(&server,&AtomicBool::new(false),usize::MAX);
        assert!(r.as_ref().is_err_and(|e|e.contains("exited early")),"{r:?}");
        let text=std::fs::read_to_string(&probe).unwrap();std::env::remove_var("GGML_BACKEND_PATH");
        assert!(text.lines().next().unwrap().trim().eq_ignore_ascii_case(&program.parent().unwrap().to_string_lossy()),"child working directory: {text}");
        assert!(!text.to_ascii_uppercase().contains("GGML_BACKEND_PATH="),"GGML_* variables must not reach the child");assert!(text.to_ascii_uppercase().contains("SYSTEMROOT="),"{text}");
        std::fs::remove_dir_all(&root).unwrap();
    }
    #[test]fn rejects_non_loopback_host(){
        let (port,_)=fake(vec![token("a"),end()]);
        for host in ["0.0.0.0","localhost","192.168.1.5","::1"]{let mut launch=idle_child(Device::Cpu,port);launch.host=host.into();
            let server=LlamaServer::scripted(Integrity::Ready(vec![launch]),QUICK);let (r,_)=collect(&server,&AtomicBool::new(false),usize::MAX);
            assert!(r.as_ref().is_err_and(|e|e.contains("127.0.0.1")),"{host}: {r:?}");assert_eq!(server.spawns(),0,"{host}");}
    }
    #[test]fn idle_watcher_stops_server(){
        let (port,_)=fake(vec![token("a"),end()]);
        let server=LlamaServer::scripted(Integrity::Ready(vec![idle_child(Device::Cpu,port)]),Timing{idle:Duration::from_millis(300),..QUICK});
        assert_eq!(collect(&server,&AtomicBool::new(false),usize::MAX).0,Ok(Device::Cpu));let pid=server.pid().unwrap();assert!(alive(pid));
        let started=Instant::now();while server.device().is_some()&&started.elapsed()<Duration::from_secs(5){thread::sleep(Duration::from_millis(20));}
        assert_eq!(server.device(),None);assert!(!alive(pid),"idle unload must kill the child");
        assert_eq!(collect(&server,&AtomicBool::new(false),usize::MAX).0,Ok(Device::Cpu),"an unloaded server restarts lazily");assert_eq!(server.spawns(),2);
    }
    #[test]fn warm_starts_server_before_first_request(){
        let (port,_)=fake(vec![token("a"),end()]);
        let server=LlamaServer::scripted(Integrity::Ready(vec![idle_child(Device::Cpu,port)]),QUICK);server.warm();
        let started=Instant::now();while server.device().is_none()&&started.elapsed()<Duration::from_secs(5){thread::sleep(Duration::from_millis(20));}
        assert_eq!(server.device(),Some(Device::Cpu),"warm must start the child without a request");assert_eq!(server.spawns(),1);
        assert_eq!(collect(&server,&AtomicBool::new(false),usize::MAX).0,Ok(Device::Cpu));assert_eq!(server.spawns(),1,"a warm child is reused");
    }
    /// Task 5 deferred minor: repeat context creates share one warm thread, and nothing starts after `shutdown`.
    #[test]fn warm_is_coalesced_and_ends_at_shutdown(){
        let (port,_)=fake(vec![token("a"),end()]);
        let server=LlamaServer::scripted(Integrity::Pending,QUICK);for _ in 0..5{server.warm();}
        assert_eq!(Arc::strong_count(&server.inner),2,"one warm thread waits for the hash pass, not five");
        server.shutdown();let started=Instant::now();while Arc::strong_count(&server.inner)>1&&started.elapsed()<Duration::from_secs(5){thread::sleep(Duration::from_millis(20));}
        assert_eq!(Arc::strong_count(&server.inner),1,"shutdown releases the waiting warm thread");
        server.set_integrity(Integrity::Ready(vec![idle_child(Device::Cpu,port)]));server.warm();thread::sleep(Duration::from_millis(300));
        assert_eq!(server.spawns(),0,"no child after shutdown");assert!(collect(&server,&AtomicBool::new(false),usize::MAX).0.is_err());assert_eq!(server.spawns(),0);
    }
    #[test]fn drop_and_stop_leave_no_child(){
        let (port,_)=fake(vec![token("a"),end()]);
        let server=LlamaServer::scripted(Integrity::Ready(vec![idle_child(Device::Cpu,port)]),QUICK);
        assert!(collect(&server,&AtomicBool::new(false),usize::MAX).0.is_ok());let pid=server.pid().unwrap();server.stop();assert_eq!(server.device(),None);assert!(!alive(pid),"stop must kill the child");
        assert!(collect(&server,&AtomicBool::new(false),usize::MAX).0.is_ok());let pid=server.pid().unwrap();drop(server);assert!(!alive(pid),"drop must kill the child");
    }
    #[test]fn no_spawn_before_or_after_failed_background_hash(){
        let (port,_)=fake(vec![token("a"),end()]);
        let server=LlamaServer::scripted(Integrity::Pending,QUICK);server.set_integrity(Integrity::Pending);
        let (r,_)=collect(&server,&AtomicBool::new(false),usize::MAX);assert!(r.as_ref().is_err_and(|e|e.contains("being verified")),"{r:?}");
        server.set_integrity(Integrity::Failed("Language model differs from its pinned digest".into()));
        let (r,_)=collect(&server,&AtomicBool::new(false),usize::MAX);assert!(r.as_ref().is_err_and(|e|e.contains("differs")),"{r:?}");assert_eq!(server.spawns(),0);
        server.set_integrity(Integrity::Ready(vec![idle_child(Device::Cpu,port)]));assert!(collect(&server,&AtomicBool::new(false),usize::MAX).0.is_ok());assert_eq!(server.spawns(),1);
    }
    fn scratch()->PathBuf{let mut r=[0u8;8];getrandom::fill(&mut r).unwrap();let p=std::env::temp_dir().join(format!("netsec-pins-{}",r.iter().map(|v|format!("{v:02x}")).collect::<String>()));std::fs::create_dir_all(&p).unwrap();p}
    fn put(root:&Path,relative:&str,bytes:&[u8])->String{let p=root.join(relative);std::fs::create_dir_all(p.parent().unwrap()).unwrap();std::fs::write(&p,bytes).unwrap();hash_file(&p).unwrap()}
    #[test]fn background_pass_checks_every_pin(){
        let root=scratch();let gpu="artifacts/rust_migration/phase1/llm/native-b11564-cuda";let cpu="artifacts/rust_migration/phase1/llm/native-b11564";
        let gguf=put(&root,"artifacts/llm/m.gguf",b"gguf");
        std::fs::write(root.join("artifacts/llm/model.json"),json!({"name":"m","file":"m.gguf","sha256":gguf,"prompt_template":"{system}{user}"}).to_string()).unwrap();
        let server=put(&root,&format!("{gpu}/llama-server.exe"),b"s");let cuda=put(&root,&format!("{gpu}/ggml-cuda.dll"),b"c");put(&root,&format!("{gpu}/llama-cli-impl.dll"),b"never loaded");
        std::fs::write(root.join(gpu).join("release.json"),json!({"files":{"llama-server.exe":server,"ggml-cuda.dll":cuda}}).to_string()).unwrap();
        put(&root,&format!("{cpu}/llama-server.exe"),b"unsealed cpu");
        let launches=verify_pins(&root).unwrap();assert_eq!(launches.iter().map(|l|l.device).collect::<Vec<_>>(),[Device::Gpu],"an unpinned CPU closure is unavailable, not executed");
        assert!(launches[0].args.iter().any(|a|a=="CUDA0")&&launches[0].args.windows(2).any(|w|w[0]=="-ngl"&&w[1]=="99")&&launches[0].host=="127.0.0.1");
        let sealed=|entries:serde_json::Value|std::fs::write(root.join("runtime_assets.json"),json!({"version":1,"files":entries}).to_string()).unwrap();
        let cpu_server=hash_file(&root.join(cpu).join("llama-server.exe")).unwrap();
        sealed(json!([{"path":format!("{cpu}/llama-server.exe"),"sha256":cpu_server,"bytes":12,"verification":"startup"},{"path":"artifacts/llm/m.gguf","sha256":gguf,"bytes":4,"verification":"background"}]));
        let launches=verify_pins(&root).unwrap();assert_eq!(launches.iter().map(|l|l.device).collect::<Vec<_>>(),[Device::Gpu,Device::Cpu]);
        assert!(launches[1].args.windows(2).any(|w|w[0]=="-ngl"&&w[1]=="0")&&launches[1].args.windows(2).any(|w|w[0]=="-t"&&w[1]=="8"));
        let cap=cache_ram(std::env::var("NETSEC_LLM_CACHE_RAM").ok().as_deref()).to_string();
        assert!(launches.iter().all(|l|l.args.windows(2).any(|w|w[0]=="--cache-ram"&&w[1]==cap)),"both launches cap llama-server's host prompt cache");
        put(&root,&format!("{gpu}/ggml-cuda.dll"),b"tampered");assert!(verify_pins(&root).is_err_and(|e|e.contains("ggml-cuda.dll")),"a pinned file that differs is an integrity error");
        put(&root,&format!("{gpu}/ggml-cuda.dll"),b"c");put(&root,&format!("{gpu}/ggml-planted.dll"),b"p");
        assert!(verify_pins(&root).is_err_and(|e|e.contains("ggml-planted.dll")),"ggml loads every ggml-*.dll beside the server, so an unpinned one is an integrity error");std::fs::remove_file(root.join(gpu).join("ggml-planted.dll")).unwrap();
        // ggml's loader opens every ggml-<backend>-*.dll whatever its suffix; only the other tools' known entry DLLs are exempt.
        for planted in ["ggml-cpu-zz-impl.dll","llama-planted-impl.dll","version.dll"]{put(&root,&format!("{gpu}/{planted}"),b"p");
            assert!(verify_pins(&root).is_err_and(|e|e.contains(planted)),"an unpinned {planted} must fail the pass");std::fs::remove_file(root.join(gpu).join(planted)).unwrap();}
        std::fs::create_dir_all(root.join(gpu).join("llama-server.exe.local")).unwrap();
        assert!(verify_pins(&root).is_err_and(|e|e.contains("llama-server.exe.local")),"DLL redirection beside the server must fail the pass");std::fs::remove_dir_all(root.join(gpu).join("llama-server.exe.local")).unwrap();
        // Once any CUDA file is sealed, release.json is ignored: a partial seal fails the whole pass, a full seal offers GPU and CPU.
        let entry=|path:String,sha:&str|json!({"path":path,"sha256":sha,"bytes":1,"verification":"background"});
        sealed(json!([entry(format!("{gpu}/llama-server.exe"),&server),entry(format!("{cpu}/llama-server.exe"),&cpu_server)]));
        assert!(verify_pins(&root).is_err_and(|e|e.contains("no pin")&&e.contains("ggml-cuda.dll")),"a partial CUDA seal is an integrity error");
        sealed(json!([entry(format!("{gpu}/llama-server.exe"),&server),entry(format!("{gpu}/ggml-cuda.dll"),&cuda),entry(format!("{cpu}/llama-server.exe"),&cpu_server)]));
        assert_eq!(verify_pins(&root).unwrap().iter().map(|l|l.device).collect::<Vec<_>>(),[Device::Gpu,Device::Cpu],"a full seal offers both launches");
        std::fs::write(root.join("artifacts/llm/model.json"),json!({"name":"m","file":"m.gguf","sha256":gguf,"prompt_template":"{system}{user}","device":"cpu"}).to_string()).unwrap();
        assert_eq!(verify_pins(&root).unwrap().iter().map(|l|l.device).collect::<Vec<_>>(),[Device::Cpu],"manifest device cpu skips the GPU child");
        std::fs::write(root.join("artifacts/llm/model.json"),json!({"name":"m","file":"m.gguf","sha256":gguf,"prompt_template":"{system}{user}"}).to_string()).unwrap();
        sealed(json!([{"path":"artifacts/llm/m.gguf","sha256":"0".repeat(64),"bytes":4,"verification":"background"}]));
        assert!(verify_pins(&root).is_err(),"the GGUF must match its sealed digest too");
        sealed(json!([]));put(&root,"artifacts/llm/m.gguf",b"changed");assert!(verify_pins(&root).is_err(),"the GGUF must match model.sha256");
        std::fs::remove_dir_all(&root).unwrap();
    }
    /// Task 7 ruling 3: the host prompt cache is capped at 1024 MiB; NETSEC_LLM_CACHE_RAM may only lower it (0..=1024, anything else ignored).
    #[test]fn cache_ram_is_capped_and_env_only_lowers(){
        assert_eq!(CACHE_RAM_MIB,1024);assert_eq!(cache_ram(None),1024);
        for (value,mib) in [("0",0),("512",512),("1024",1024),(" 256 ",256),("1025",1024),("8192",1024),("-1",1024),("1e3",1024),("",1024),("off",1024)]{assert_eq!(cache_ram(Some(value)),mib,"{value:?}");}
    }
    /// Task 7 check after `stage_native_app.py --seal-only`: the real dev tree pins both closures. Hashes about 2.5 GB; starts nothing, uses no GPU.
    /// `NETSEC_PINS_ROOT=<package root>` repeats it against the staged package (plan Task 7 Step 3).
    #[test]#[ignore="Hashes the real GGUF and runtime folders; run explicitly after a reseal"]fn real_tree_pins_offer_gpu_and_cpu(){
        let root=std::env::var_os("NETSEC_PINS_ROOT").map(PathBuf::from).unwrap_or_else(||Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(4).unwrap().to_path_buf());
        let launches=verify_pins(&root);println!("{:?}",launches.as_ref().map(|l|l.iter().map(|l|l.device).collect::<Vec<_>>()));
        assert_eq!(launches.unwrap().iter().map(|l|l.device).collect::<Vec<_>>(),[Device::Gpu,Device::Cpu]);
    }
    /// Manual GPU check against the real pinned model (plan Task 3 Step 5).
    #[test]#[ignore="Starts the real pinned llama-server and model; run explicitly"]fn real_server_answers_once(){
        let root=Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(4).unwrap().to_path_buf();let hashing=Instant::now();
        let server=LlamaServer::new(root.clone());assert!(server.wait_verified(Duration::from_secs(300)),"background hash pass did not finish");let hashed=hashing.elapsed();
        let model=crate::language::installed(&root).unwrap();let prompt=model.prompt_template.replacen("{system}","Answer in one word.",1).replacen("{user}","Say OK",1);
        let started=Instant::now();let mut first=None;let mut text=String::new();
        let device=server.stream(&prompt,8,&AtomicBool::new(false),&mut |t|{first.get_or_insert(started.elapsed());text.push_str(t);true}).unwrap();
        let total=started.elapsed();println!("{device:?} hash_pass={hashed:?} first_token={first:?} total={total:?} text={text:?}");
        assert_eq!(device,Device::Gpu);assert!(total<Duration::from_secs(10),"{total:?}");assert!(text.to_ascii_lowercase().contains("ok"),"{text}");server.stop();
    }
}
