//! Actual bounded local generation. Evidence facts are input, never a fallback response.
use crate::application::{Application,hash_file,read_json};
use netsec_core::jobs::Job;
use serde_json::{json,Value};
use std::{collections::BTreeSet,fs::{self,File},path::Path,process::{Command,Stdio},sync::atomic::Ordering,time::{Duration,Instant}};

const MODEL:&str="qwen2.5-0.5b-instruct-q4_k_m.gguf";
const MODEL_HASH:&str="74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db";
const LIBRARY_HASH:&str="61e98ea9589e40e3bd45c3d98767071e31afbc392cf234d40ac336c6cbe9e76e";
pub fn status(root:&Path)->Value {
    let available=root.join("artifacts/llm/qwen").join(MODEL).is_file()
        &&root.join("artifacts/rust_migration/phase1/llm/native-b11564/llama-completion.exe").is_file();
    json!({"status":if available{"ready"}else{"unavailable"},"model":"Qwen2.5-0.5B-Instruct Q4_K_M","provider":"Local llama.cpp b11564",
        "cpu_threads":2,"offline":true,"automatic_generation":false,"reason":if available{"Generation starts only when requested"}else{"Verified local GGUF/native runtime is missing"}})
}
fn normalized(text:&str)->String{text.split_whitespace().collect::<Vec<_>>().join(" ")}
fn facts(prediction:&Value,label:&str,explanation:Option<&Value>)->Vec<String>{
    let name=prediction["display_label"].as_str().or_else(||prediction["label"].as_str()).unwrap_or("an unavailable category");
    let mut facts=vec![format!("{label} predicts {name} for this observation.")];
    facts.push(if prediction["score_kind"]=="vote_correctness" {
        "Its confidence estimates the correctness of a hard class vote, not an ensemble class probability.".into()
    }else{"Its model probability is an estimate, not independent evidence that the category is correct.".into()});
    if let Some(explanation)=explanation {
        let method=explanation["method"].as_str().unwrap_or("");
        match method {
            "anchors"=>facts.push(if explanation["details"]["state"]=="supported_anchor" {
                "The anchor has measured reference support, with the recorded sampling limits.".into()
            }else{"The reference does not provide enough support to certify an anchor rule.".into()}),
            "counterfactual"=>facts.push(if explanation["details"]["decision_change_verified"]==true {
                "An observed reference with the same measured-feature mask changes the model decision; this is not a causal remedy.".into()
            }else{"No feasible observed-reference decision change was verified within the search budget.".into()}),
            _=>{
                let strongest=explanation["features"].as_array().into_iter().flatten().filter_map(|f|f["contribution"].as_f64().map(|v|(v,f)))
                    .max_by(|(a,_),(b,_)|a.abs().total_cmp(&b.abs()));
                if let Some((value,feature))=strongest{
                    let name=feature["feature"].as_str().unwrap_or("the recorded feature group");
                    let direction=if value>0.{"positive"}else if value<0.{"negative"}else{"zero"};
                    if method=="permutation"{facts.push(format!("On the training reference, {name} has the largest displayed {direction} Brier-loss change; this is global importance."));}
                    else if method=="ale"{facts.push(format!("The recorded accumulated local effect for {name} is {direction}; it describes a model response, not a causal effect."));}
                    else{let units=explanation["units"].as_str().unwrap_or("model score");facts.push(format!("Relative to its reference, {name} has the largest displayed {direction} contribution in {units} units."));}
                }else{facts.push("The explanation has insufficient feasible reference evidence for a feature attribution.".into());}
            }
        }
    }
    facts.push("This model prediction needs human verification and does not confirm an attack or guarantee safety.".into());facts
}
fn terms(text:&str)->BTreeSet<String>{text.to_lowercase().split(|c:char|!c.is_alphanumeric()).filter(|w|w.len()>2).map(str::to_owned).collect()}
fn retrieve(root:&Path,query:&str)->Result<(String,Value),String>{
    let path=root.join("framework/rag/knowledge.json");let corpus=read_json(&path,1024*1024)?;
    let chunks=corpus["chunks"].as_array().ok_or("Curated local knowledge has no passages")?;let query=terms(query);
    let mut scored=chunks.iter().map(|chunk|{
        let words=terms(&format!("{} {}",chunk["title"].as_str().unwrap_or(""),chunk["text"].as_str().unwrap_or("")));
        let score=query.intersection(&words).map(|word|{
            let count=chunks.iter().filter(|c|terms(c["text"].as_str().unwrap_or("")).contains(word)).count();
            (((chunks.len()+1) as f64/(count+1) as f64).ln()+1.)/(words.len().max(1) as f64).sqrt()
        }).sum::<f64>();(score,chunk)
    }).collect::<Vec<_>>();scored.sort_by(|a,b|b.0.total_cmp(&a.0).then(a.1["id"].as_str().cmp(&b.1["id"].as_str())));
    let mut context=String::new();let mut audit=Vec::new();
    for (score,chunk) in scored.into_iter().filter(|(s,_)|*s>0.).take(2){
        let text=chunk["text"].as_str().unwrap_or("");if context.len()+text.len()>700{continue;}
        context.push_str(text);context.push('\n');audit.push(json!({"id":chunk["id"],"source":chunk["source"],"score":score}));
    }
    Ok((context,json!({"method":"Bounded local inverse-document-frequency term retrieval","corpus_sha256":hash_file(&path)?,"passages":audit,"network_used":false})))
}
struct Scratch(std::path::PathBuf);
impl Drop for Scratch {fn drop(&mut self){for name in ["prompt.txt","stdout.txt","stderr.txt"]{let _=fs::remove_file(self.0.join(name));}let _=fs::remove_dir(&self.0);}}
struct OwnedChild(std::process::Child);
impl Drop for OwnedChild {fn drop(&mut self){if self.0.try_wait().ok().flatten().is_none(){let _=self.0.kill();let _=self.0.wait();}}}

pub fn generate(app:&Application,job:&Job,model:&str,row:usize,explanation:Option<&Value>)->Result<Value,String>{
    let (_,prediction)=job.observation(model,row)?;let info=app.catalog.info(model).map_err(|e|e.to_string())?;
    generate_evidence(app,&prediction,&info.label,explanation,&job.cancelled,json!({"job_id":job.view().id,"input_id":job.input.id,"model_id":model,"row_index":row}))
}

pub fn generate_evidence(app:&Application,prediction:&Value,label:&str,explanation:Option<&Value>,cancel:&std::sync::atomic::AtomicBool,binding:Value)->Result<Value,String>{
    if app.stopping.load(Ordering::Acquire)||cancel.load(Ordering::Acquire){return Err("English generation cancelled".into());}
    let facts=facts(prediction,label,explanation);let expected=normalized(&facts.join(" "));
    let (background,retrieval)=retrieve(&app.root,&facts.join(" "))?;
    let model_path=app.root.join("artifacts/llm/qwen").join(MODEL);
    let native=app.root.join("artifacts/rust_migration/phase1/llm/native-b11564");
    if hash_file(&model_path)?!=MODEL_HASH||hash_file(&native.join("llama.dll"))?!=LIBRARY_HASH{return Err("Local language weights or native library differ from the pinned verified release".into());}
    let mut random=[0u8;16];getrandom::fill(&mut random).map_err(|e|e.to_string())?;
    let folder=std::env::temp_dir().join(format!("netsec-{}",random.iter().map(|v|format!("{v:02x}")).collect::<String>()));
    fs::create_dir(&folder).map_err(|e|e.to_string())?;let scratch=Scratch(folder);
    let started=Instant::now();let mut attempts=Vec::new();
    for attempt in 0..2 {
        let instruction="Join every FACT sentence into one paragraph. Copy every FACT sentence exactly, in order. Do not add or omit words. Reference notes are background only; do not copy them. Output only the joined FACT sentences.";
        let prompt=format!("<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\nREFERENCE NOTES:\n{background}\nFACTS:\n{}\n{}<|im_end|>\n<|im_start|>assistant\n",facts.iter().map(|v|format!("FACT: {v}")).collect::<Vec<_>>().join("\n"),if attempt==0{""}else{"Copy the FACT sentences exactly. Do not describe the instructions.\n"});
        fs::write(scratch.0.join("prompt.txt"),prompt).map_err(|e|e.to_string())?;
        let mut command=Command::new(native.join("llama-completion.exe"));
        command.arg("-m").arg(&model_path).arg("-f").arg(scratch.0.join("prompt.txt"))
            .args(["-c","2048","-t","2","-tb","2","-b","128","-ub","128","-ngl","0","--seed","42","-n","200","--temp","0","--top-p","1","--repeat-penalty","1","-fa","off","--no-conversation","--no-warmup","--no-display-prompt","--simple-io","--offline","--fit","off"])
            .stdin(Stdio::null()).stdout(File::create(scratch.0.join("stdout.txt")).map_err(|e|e.to_string())?)
            .stderr(File::create(scratch.0.join("stderr.txt")).map_err(|e|e.to_string())?);
        #[cfg(windows)] {use std::os::windows::process::CommandExt;command.creation_flags(0x08000000);}
        let mut child=OwnedChild(command.spawn().map_err(|e|e.to_string())?);let clock=Instant::now();
        let status=loop{
            if app.stopping.load(Ordering::Acquire)||cancel.load(Ordering::Acquire){return Err("English generation cancelled".into());}
            if clock.elapsed()>Duration::from_secs(60){return Err("Owned local language generation exceeded its deadline".into());}
            if fs::metadata(scratch.0.join("stdout.txt")).map_err(|e|e.to_string())?.len()>128*1024{return Err("Language output exceeded its bound".into());}
            if let Some(status)=child.0.try_wait().map_err(|e|e.to_string())?{break status;}
            std::thread::sleep(Duration::from_millis(15));
        };
        if !status.success(){return Err(format!("Local llama.cpp exited {status}; predictions and XAI remain available"));}
        let generated=fs::read_to_string(scratch.0.join("stdout.txt")).map_err(|e|e.to_string())?;
        let generated=generated.trim().strip_suffix("[end of text]").unwrap_or(generated.trim()).trim();
        let passed=normalized(generated)==expected;
        attempts.push(json!({"attempt":attempt+1,"passed_exact_fact_check":passed,"seconds":clock.elapsed().as_secs_f64()}));
        if passed{let mut result=json!({"status":"completed","text":generated,"model":"Qwen2.5-0.5B-Instruct Q4_K_M","provider":"Local llama.cpp b11564",
            "llm_executed":true,"inference_device":"cpu","grounding":{"passed":true,"method":"Every generated fact must exactly match supplied model/evidence sentences after whitespace normalization","limitations":"Conservative lexical fidelity, not proof that detector or explanation is correct"},
            "retrieval":retrieval,"attempts":attempts,"elapsed_seconds":started.elapsed().as_secs_f64(),
            "warnings":["The language model converts supplied evidence to English; it does not independently detect attacks or validate research accuracy."]});
            if let Some(fields)=binding.as_object(){for (key,value) in fields{result[key]=value.clone();}}
            return Ok(result);
        }
    }
    Err("The local language model did execute, but its output failed evidence-fidelity checks; no substitute text was returned".into())
}
