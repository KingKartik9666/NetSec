use crate::application::{Application,display_label};
use netsec_core::jobs::Job;
use netsec_models::{ModelCatalog,ModelInfo,ModelOutput};
use serde_json::{json,Value};
use std::{env,io::{Read,Write},process::{Command,Stdio},sync::{Arc,atomic::Ordering},time::{Duration,Instant}};

pub fn prediction(info:&ModelInfo,value:&ModelOutput)->Value{
    let mut ordered=(0..info.classes.len()).collect::<Vec<_>>();
    if let Some(p)=&value.probabilities{ordered.sort_by(|&a,&b|p[b].total_cmp(&p[a]).then(a.cmp(&b)));}
    let classes=value.probabilities.as_ref().map(|p|ordered.iter().take(3).map(|&i|json!({"label":info.classes[i],"display_label":display_label(&info.classes[i]),"score":p[i]})).collect::<Vec<_>>()).unwrap_or_default();
    json!({"class_index":value.predicted,"label":info.classes[value.predicted],"display_label":display_label(&info.classes[value.predicted]),
        "score":value.confidence,"score_kind":value.confidence_kind,"top_classes":classes,"probabilities":value.probabilities,
        "attack_probability":value.attack_probability,"novelty":"unvalidated","is_attack":value.predicted!=0})
}
pub fn worker(args:&[String])->Result<(),String>{
    let models=args.windows(2).find(|v|v[0]=="--models").map(|v|&v[1]).ok_or("Missing owned worker models directory")?;
    let model=args.windows(2).find(|v|v[0]=="--model").map(|v|&v[1]).ok_or("Missing worker model")?;
    let mut bytes=Vec::new();std::io::stdin().take(10*1024*1024+1).read_to_end(&mut bytes).map_err(|e|e.to_string())?;
    if bytes.len()>10*1024*1024{return Err("Inference message exceeds size bound".into());}
    let rows:Vec<Vec<Option<f64>>>=serde_json::from_slice(&bytes).map_err(|e|e.to_string())?;
    let catalog=ModelCatalog::load(std::path::Path::new(models)).map_err(|e|e.to_string())?;
    let selected=catalog.load_model(model).map_err(|e|e.to_string())?;
    if rows.is_empty()||rows.len()>500{return Err("Inference worker observation bound exceeded".into());}
    let mut results=Vec::new();
    for row in rows{
        let row=row.iter().map(|v|v.unwrap_or(f64::NAN)).collect::<Vec<_>>();
        let output=selected.predict(&row).map_err(|e|e.to_string())?;results.push(prediction(selected.info(),&output));
    }
    serde_json::to_writer(std::io::stdout(),&results).map_err(|e|e.to_string())?;Ok(())
}
struct OwnedChild(std::process::Child);
impl Drop for OwnedChild{fn drop(&mut self){if self.0.try_wait().ok().flatten().is_none(){let _=self.0.kill();let _=self.0.wait();}}}

fn predict_isolated(app:&Application,job:&Job,id:&str)->Result<Vec<Value>,String>{
    let executable=env::current_exe().map_err(|e|e.to_string())?;
    let mut command=Command::new(executable);
    command.arg("--predict-worker").arg("--models").arg(app.catalog.root()).arg("--model").arg(id)
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)] {use std::os::windows::process::CommandExt;command.creation_flags(0x08000000);}
    let mut child=OwnedChild(command.spawn().map_err(|e|e.to_string())?);
    let mut stdin=child.0.stdin.take().ok_or("Worker input pipe missing")?;
    let input=serde_json::to_vec(&job.input.rows).map_err(|e|e.to_string())?;
    let input_thread=std::thread::spawn(move||stdin.write_all(&input));
    let mut stdout=child.0.stdout.take().ok_or("Worker output pipe missing")?;
    let output_thread=std::thread::spawn(move||{let mut bytes=Vec::new();stdout.by_ref().take(16*1024*1024+1).read_to_end(&mut bytes).map(|_|bytes)});
    let stderr=child.0.stderr.take().ok_or("Worker diagnostics pipe missing")?;
    let error_thread=std::thread::spawn(move||{let mut bytes=Vec::new();stderr.take(256*1024).read_to_end(&mut bytes).map(|_|bytes)});
    let started=Instant::now();
    let status=loop{
        if job.cancelled.load(Ordering::Acquire)||app.stopping.load(Ordering::Acquire){return Err("Analysis cancelled".into());}
        if started.elapsed()>Duration::from_secs(300){return Err("Owned native inference worker exceeded its five-minute deadline".into());}
        if let Some(status)=child.0.try_wait().map_err(|e|e.to_string())?{break status;}
        std::thread::sleep(Duration::from_millis(15));
    };
    input_thread.join().map_err(|_|"Worker input writer failed")?.map_err(|e|e.to_string())?;
    let output=output_thread.join().map_err(|_|"Worker output reader failed")?.map_err(|e|e.to_string())?;
    let error=error_thread.join().map_err(|_|"Worker diagnostic reader failed")?.map_err(|e|e.to_string())?;
    if !status.success(){return Err(format!("Native model worker failed: {}",String::from_utf8_lossy(&error)));}
    if output.len()>16*1024*1024{return Err("Model worker response exceeded output bound".into());}
    serde_json::from_slice(&output).map_err(|e|e.to_string())
}
pub fn enqueue(app:Arc<Application>,job:Arc<Job>){
    tokio::spawn(async move{
        let permit=tokio::select!{
            permit=app.inference_slots.clone().acquire_owned()=>permit.ok(),
            _=app.shutdown.notified()=>None,
        };
        let Some(_permit)=permit else{job.cancel();return;};
        let app2=app.clone();let job2=job.clone();
        let _=tokio::task::spawn_blocking(move||{
            let models=job2.view().models;
            for (index,model) in models.iter().enumerate(){
                if !job2.begin(index)||app2.stopping.load(Ordering::Acquire){break;}
                let started=Instant::now();let result=predict_isolated(&app2,&job2,&model.model_id);
                job2.record(index,result,started.elapsed().as_millis() as u64);
            }
        }).await;
    });
}
