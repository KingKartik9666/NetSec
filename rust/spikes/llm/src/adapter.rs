//! Pinned b11564 C ABI adapter and owned local llama-completion process.
//! Unsafe calls stay here. This is not a general-purpose ABI or production port.
use crate::{Request, SpikeError};
use libloading::Library;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    ffi::{c_char, c_void, CString},
    fs::{self, File},
    io::Read,
    path::Path,
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

const WEIGHTS_HASH: &str = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db";

struct OwnedChild(Child);

impl Drop for OwnedChild {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

#[repr(C)]
struct ModelParams {
    devices: *mut c_void,
    tensor_buft_overrides: *mut c_void,
    n_gpu_layers: i32,
    split_mode: i32,
    load_mode: i32,
    lazy_mode: i32,
    main_gpu: i32,
    tensor_split: *const f32,
    progress_callback: *mut c_void,
    progress_callback_user_data: *mut c_void,
    kv_overrides: *const c_void,
    vocab_only: bool,
    check_tensors: bool,
    use_extra_bufts: bool,
    no_host: bool,
    no_alloc: bool,
    load_mtp: bool,
}

#[repr(C)]
struct ContextParams {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_rs_seq: u32,
    n_outputs_max: u32,
    n_outputs_max_per_seq: u32,
    n_threads: i32,
    n_threads_batch: i32,
    ctx_type: i32,
    rope_scaling_type: i32,
    pooling_type: i32,
    attention_type: i32,
    flash_attn_type: i32,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    defrag_thold: f32,
    cb_eval: *mut c_void,
    cb_eval_user_data: *mut c_void,
    type_k: i32,
    type_v: i32,
    moe_cache_size: usize,
    abort_callback: *mut c_void,
    abort_callback_data: *mut c_void,
    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    samplers: *mut c_void,
    n_samplers: usize,
    ctx_other: *mut c_void,
}

#[repr(C)]
struct Batch {
    n_tokens: i32,
    token: *mut i32,
    embd: *mut f32,
    pos: *mut i32,
    n_seq_id: *mut i32,
    seq_id: *mut *mut i32,
    logits: *mut i8,
}

fn file_hash(path: &Path) -> Result<String, SpikeError> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn verify_weights(path: &Path) -> Result<(), SpikeError> {
    if file_hash(path)? != WEIGHTS_HASH {
        return Err(SpikeError::Invalid(
            "Local Qwen weights differ from the verified official model".into(),
        ));
    }
    Ok(())
}

fn verify_native(path: &Path) -> Result<(), SpikeError> {
    if file_hash(&path.join("llama.dll"))?
        != "61e98ea9589e40e3bd45c3d98767071e31afbc392cf234d40ac336c6cbe9e76e"
    {
        return Err(SpikeError::Invalid(
            "Native library differs from pinned b11564 ABI".into(),
        ));
    }
    Ok(())
}

fn cstring(s: &str) -> Result<CString, SpikeError> {
    CString::new(s).map_err(|_| SpikeError::Invalid("NUL in native parameter".into()))
}

pub fn native(request: &Request) -> Result<Value, SpikeError> {
    let start = Instant::now();
    verify_weights(&request.model)?;
    verify_native(&request.native_dir)?;
    let prompt = fs::read_to_string(&request.prompt_file)?;
    let model_name = cstring(&request.model.to_string_lossy())?;
    let directory = cstring(&request.native_dir.to_string_lossy())?;
    // SAFETY: library filenames and struct definitions are pinned to b11564.
    // All symbols are used while their owning libraries are alive. Context,
    // sampler and model are freed in that order on both success and error.
    // Native faults cannot be caught by Rust; the experiment runs in its own process.
    unsafe {
        let _omp = Library::new(request.native_dir.join("libomp.dll"))?;
        let _ggml_base = Library::new(request.native_dir.join("ggml-base.dll"))?;
        let ggml = Library::new(request.native_dir.join("ggml.dll"))?;
        let llama = Library::new(request.native_dir.join("llama.dll"))?;
        let backend_load =
            ggml.get::<unsafe extern "C" fn(*const c_char)>(b"ggml_backend_load_all_from_path\0")?;
        backend_load(directory.as_ptr());
        let backend_init = llama.get::<unsafe extern "C" fn()>(b"llama_backend_init\0")?;
        let backend_free = llama.get::<unsafe extern "C" fn()>(b"llama_backend_free\0")?;
        let model_defaults =
            llama.get::<unsafe extern "C" fn() -> ModelParams>(b"llama_model_default_params\0")?;
        let context_defaults = llama
            .get::<unsafe extern "C" fn() -> ContextParams>(b"llama_context_default_params\0")?;
        let load = llama.get::<unsafe extern "C" fn(*const c_char, ModelParams) -> *mut c_void>(
            b"llama_model_load_from_file\0",
        )?;
        let init = llama.get::<unsafe extern "C" fn(*mut c_void, ContextParams) -> *mut c_void>(
            b"llama_init_from_model\0",
        )?;
        let free_model = llama.get::<unsafe extern "C" fn(*mut c_void)>(b"llama_model_free\0")?;
        let free_ctx = llama.get::<unsafe extern "C" fn(*mut c_void)>(b"llama_free\0")?;
        let free_sampler =
            llama.get::<unsafe extern "C" fn(*mut c_void)>(b"llama_sampler_free\0")?;
        let vocab_fn = llama.get::<unsafe extern "C" fn(*mut c_void) -> *const c_void>(
            b"llama_model_get_vocab\0",
        )?;
        let tokenize = llama.get::<unsafe extern "C" fn(
            *const c_void,
            *const c_char,
            i32,
            *mut i32,
            i32,
            bool,
            bool,
        ) -> i32>(b"llama_tokenize\0")?;
        let one =
            llama.get::<unsafe extern "C" fn(*mut i32, i32) -> Batch>(b"llama_batch_get_one\0")?;
        let decode =
            llama.get::<unsafe extern "C" fn(*mut c_void, Batch) -> i32>(b"llama_decode\0")?;
        let greedy =
            llama.get::<unsafe extern "C" fn() -> *mut c_void>(b"llama_sampler_init_greedy\0")?;
        let sample = llama.get::<unsafe extern "C" fn(*mut c_void, *mut c_void, i32) -> i32>(
            b"llama_sampler_sample\0",
        )?;
        let is_eog = llama
            .get::<unsafe extern "C" fn(*const c_void, i32) -> bool>(b"llama_vocab_is_eog\0")?;
        let piece = llama.get::<unsafe extern "C" fn(
            *const c_void,
            i32,
            *mut c_char,
            i32,
            i32,
            bool,
        ) -> i32>(b"llama_token_to_piece\0")?;
        backend_init();
        let mut model = std::ptr::null_mut();
        let mut context = std::ptr::null_mut();
        let mut sampler = std::ptr::null_mut();
        let result = (|| -> Result<Value, SpikeError> {
            let mut model_params = model_defaults();
            model_params.n_gpu_layers = 0;
            model = load(model_name.as_ptr(), model_params);
            if model.is_null() {
                return Err(SpikeError::Invalid("Native Qwen load failed".into()));
            }
            let mut params = context_defaults();
            params.n_ctx = 2048;
            params.n_batch = 128;
            params.n_ubatch = 128;
            params.n_threads = 2;
            params.n_threads_batch = 2;
            params.flash_attn_type = 0;
            context = init(model, params);
            if context.is_null() {
                return Err(SpikeError::Invalid("Native Qwen context failed".into()));
            }
            let ready = start.elapsed().as_secs_f64();
            let vocab = vocab_fn(model);
            let mut tokens = vec![0_i32; 2048];
            let count = tokenize(
                vocab,
                prompt.as_ptr().cast(),
                i32::try_from(prompt.len())
                    .map_err(|_| SpikeError::Invalid("Oversize prompt".into()))?,
                tokens.as_mut_ptr(),
                2048,
                true,
                true,
            );
            if !(1..=1748).contains(&count) {
                return Err(SpikeError::Invalid(
                    "Evidence exceeds the local model's bounded context".into(),
                ));
            }
            tokens.truncate(count as usize);
            for chunk in tokens.chunks_mut(128) {
                if decode(context, one(chunk.as_mut_ptr(), chunk.len() as i32)) != 0 {
                    return Err(SpikeError::Invalid("Native prompt decode failed".into()));
                }
            }
            sampler = greedy();
            if sampler.is_null() {
                return Err(SpikeError::Invalid(
                    "Native sampler allocation failed".into(),
                ));
            }
            let mut bytes = Vec::new();
            let mut finish = "length";
            let mut first_token = None;
            let mut generated = 0;
            for _ in 0..200 {
                if start.elapsed().as_secs() >= request.timeout_seconds {
                    return Err(SpikeError::Invalid(
                        "Native generation deadline exceeded".into(),
                    ));
                }
                let mut token = sample(sampler, context, -1);
                if is_eog(vocab, token) {
                    finish = "stop";
                    break;
                }
                first_token.get_or_insert_with(|| start.elapsed().as_secs_f64());
                let mut buffer = vec![0_u8; 256];
                let mut len = piece(
                    vocab,
                    token,
                    buffer.as_mut_ptr().cast(),
                    buffer.len() as i32,
                    0,
                    false,
                );
                if len < 0 {
                    buffer.resize((-len) as usize, 0);
                    len = piece(
                        vocab,
                        token,
                        buffer.as_mut_ptr().cast(),
                        buffer.len() as i32,
                        0,
                        false,
                    );
                }
                if len < 0 {
                    return Err(SpikeError::Invalid("Native detokenization failed".into()));
                }
                bytes.extend_from_slice(&buffer[..len as usize]);
                generated += 1;
                if decode(context, one(&mut token, 1)) != 0 {
                    return Err(SpikeError::Invalid(
                        "Native generation decode failed".into(),
                    ));
                }
            }
            let text = String::from_utf8(bytes)
                .map_err(|_| SpikeError::Invalid("Invalid generated UTF-8".into()))?
                .trim()
                .to_string();
            Ok(
                json!({"text": text, "finish_reason": finish, "usage": {"prompt_tokens": count,"completion_tokens":generated}, "load_seconds":ready,"first_token_seconds":first_token,"generation_total_seconds":start.elapsed().as_secs_f64(),"native_release":"b11564","llm_executed":true}),
            )
        })();
        if !sampler.is_null() {
            free_sampler(sampler);
        }
        if !context.is_null() {
            free_ctx(context);
        }
        if !model.is_null() {
            free_model(model);
        }
        backend_free();
        result
    }
}

pub fn child(request: &Request) -> Result<Value, SpikeError> {
    let start = Instant::now();
    verify_weights(&request.model)?;
    verify_native(&request.native_dir)?;
    let stdout_path = request.prompt_file.with_extension("native.stdout");
    let stderr_path = request.prompt_file.with_extension("native.stderr");
    let mut process = OwnedChild(
        Command::new(request.native_dir.join("llama-completion.exe"))
            .args(["-m"])
            .arg(&request.model)
            .args(["-f"])
            .arg(&request.prompt_file)
            .args([
                "-c",
                "2048",
                "-t",
                "2",
                "-tb",
                "2",
                "-b",
                "128",
                "-ub",
                "128",
                "-ngl",
                "0",
                "--seed",
                "42",
                "-n",
                "200",
                "--temp",
                "0",
                "--top-p",
                "1",
                "--repeat-penalty",
                "1",
                "-fa",
                "off",
                "--no-conversation",
                "--no-warmup",
                "--no-display-prompt",
                "--simple-io",
                "--offline",
                "--fit",
                "off",
            ])
            .stdin(Stdio::null())
            .stdout(File::create(&stdout_path)?)
            .stderr(File::create(&stderr_path)?)
            .spawn()?,
    );
    let child_id = process.0.id();
    let mut first_token = None;
    let status = loop {
        if first_token.is_none() && fs::metadata(&stdout_path)?.len() > 0 {
            first_token = Some(start.elapsed().as_secs_f64());
        }
        if let Some(status) = process.0.try_wait()? {
            break status;
        }
        if start.elapsed().as_secs() >= request.timeout_seconds {
            process.0.kill()?;
            process.0.wait()?;
            return Err(SpikeError::Invalid(
                "Owned llama.cpp process deadline exceeded".into(),
            ));
        }
        thread::sleep(Duration::from_millis(10));
    };
    if !status.success() {
        return Err(SpikeError::Invalid(format!(
            "Owned llama.cpp process exited {status}"
        )));
    }
    let stdout = fs::read_to_string(&stdout_path)?.replace("\r\n", "\n");
    let generated = stdout.trim();
    let diagnostics = fs::read_to_string(&stderr_path)?;
    // Native CLI has no stable JSON response contract. This spike parses only
    // its known b11564 diagnostics and does not claim a production protocol.
    let (text, finish) = match generated.strip_suffix("[end of text]") {
        Some(text) => (text.trim().to_string(), "stop"),
        None => (generated.to_string(), "length"),
    };
    let load = regex::Regex::new(r"load time\s*=\s*([0-9.]+) ms")?
        .captures(&diagnostics)
        .and_then(|c| c.get(1))
        .and_then(|s| s.as_str().parse::<f64>().ok())
        .map(|ms| ms / 1000.0);
    Ok(
        json!({"text":text,"finish_reason":finish,"first_token_seconds":first_token,"llama_reported_load_seconds":load,"generation_total_seconds":start.elapsed().as_secs_f64(),"native_release":"b11564","llm_executed":true,"child_id":child_id,"child_reaped":true}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_windows_abi_sizes() {
        assert_eq!(std::mem::size_of::<ModelParams>(), 80);
        assert_eq!(std::mem::size_of::<ContextParams>(), 168);
        assert_eq!(std::mem::size_of::<Batch>(), 56);
    }
}
