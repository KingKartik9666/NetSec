//! Authenticated loopback HTTP adapter. Domain and inference do not depend on axum.
use crate::{application::Application,analysis};
use axum::{Router,Json,body::Body,extract::{State,Path,Query,DefaultBodyLimit,rejection::JsonRejection},http::{StatusCode,Request,HeaderMap,header},response::{IntoResponse,Response},routing::{get,post},middleware::{self,Next}};
use netsec_core::input;
use netsec_explain::{Method,Settings,Evaluation};
use serde::Deserialize;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{path::PathBuf,sync::{Arc,atomic::Ordering},io::{BufRead,Write}};

#[derive(Clone)] struct Server {app:Arc<Application>,token:String,port:u16}
struct ApiError(StatusCode,String);
impl IntoResponse for ApiError{fn into_response(self)->Response{(self.0,Json(json!({"error":self.1,"status":"failed"}))).into_response()}}
impl From<String> for ApiError{fn from(v:String)->Self{Self(StatusCode::BAD_REQUEST,v)}}
fn payload<T>(body:Result<Json<T>,JsonRejection>)->Result<T,ApiError>{body.map(|v|v.0).map_err(|e|ApiError(e.status(),e.body_text()))}
fn equal(a:&str,b:&str)->bool{a.len()==b.len()&&a.as_bytes().iter().zip(b.as_bytes()).fold(0u8,|all,(a,b)|all|(a^b))==0}
fn cookie(headers:&HeaderMap)->Option<&str>{headers.get(header::COOKIE)?.to_str().ok()?.split(';').find_map(|v|v.trim().strip_prefix("netsec_session="))}
async fn protect(State(state):State<Server>,request:Request<Body>,next:Next)->Response{
    let host=request.headers().get(header::HOST).and_then(|v|v.to_str().ok()).unwrap_or("");
    if host!=format!("127.0.0.1:{}",state.port)&&host!=format!("localhost:{}",state.port){return ApiError(StatusCode::FORBIDDEN,"Only the owned loopback host is permitted".into()).into_response();}
    if let Some(origin)=request.headers().get(header::ORIGIN).and_then(|v|v.to_str().ok()){
        if origin!=format!("http://{host}"){return ApiError(StatusCode::FORBIDDEN,"Only requests from this local app are allowed".into()).into_response();}
    }
    if request.headers().get("sec-fetch-site").is_some_and(|v|v=="cross-site") {return ApiError(StatusCode::FORBIDDEN,"Cross-site local requests are refused".into()).into_response();}
    if request.uri().path()!="/bootstrap" {
        let supplied=request.headers().get("x-imdai-session").and_then(|v|v.to_str().ok()).or_else(||cookie(request.headers())).unwrap_or("");
        if !equal(supplied,&state.token){return ApiError(StatusCode::FORBIDDEN,"This analysis service belongs to the current desktop session".into()).into_response();}
    }
    // Consume a bounded rejected body so Windows clients receive the JSON 415
    // response instead of a reset caused by closing a connection with unread data.
    if request.method()==axum::http::Method::POST {
        let kind=request.headers().get(header::CONTENT_TYPE).and_then(|v|v.to_str().ok()).unwrap_or("");
        let mime=kind.split(';').next().unwrap_or("").trim();
        if mime!="application/json" && !(mime.starts_with("application/")&&mime.ends_with("+json")) {
            let status=if axum::body::to_bytes(request.into_body(),12*1024*1024).await.is_ok(){StatusCode::UNSUPPORTED_MEDIA_TYPE}else{StatusCode::PAYLOAD_TOO_LARGE};
            return ApiError(status,"Requests must contain bounded application/json data".into()).into_response();
        }
    }
    let mut response=next.run(request).await;
    let headers=response.headers_mut();
    headers.insert("x-content-type-options","nosniff".parse().unwrap());headers.insert(header::CACHE_CONTROL,"no-store".parse().unwrap());
    headers.insert("content-security-policy","default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; font-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'".parse().unwrap());
    response
}
#[derive(Deserialize)]struct Bootstrap{session:String}
async fn bootstrap(State(state):State<Server>,Query(query):Query<Bootstrap>)->Response{
    if !equal(&query.session,&state.token){return ApiError(StatusCode::FORBIDDEN,"Invalid desktop bootstrap".into()).into_response();}
    let mut response=StatusCode::SEE_OTHER.into_response();response.headers_mut().insert(header::LOCATION,"/".parse().unwrap());
    response.headers_mut().insert(header::SET_COOKIE,format!("netsec_session={}; HttpOnly; SameSite=Strict; Path=/",state.token).parse().unwrap());response
}
async fn page(State(state):State<Server>)->Result<Response,ApiError>{
    let path=state.app.root.join("rust/netsec/web/dist/index.html");
    let html=std::fs::read_to_string(path).map_err(|e|ApiError(StatusCode::SERVICE_UNAVAILABLE,e.to_string()))?;
    let html=if std::env::var_os("NETSEC_DESKTOP").is_some(){html.replace("class=\"native-app\"","class=\"native-app\" data-desktop=\"true\"")}else{html};
    Ok(([(header::CONTENT_TYPE,"text/html; charset=utf-8")],html).into_response())
}
async fn asset(State(state):State<Server>,Path(name):Path<String>)->Result<Response,ApiError>{
    let path=std::path::Path::new(&name);
    if path.components().any(|v|!matches!(v,std::path::Component::Normal(_))){return Err(ApiError(StatusCode::NOT_FOUND,"Asset not found".into()));}
    let base=state.app.root.join("rust/netsec/web/dist/static").canonicalize().map_err(|e|e.to_string())?;
    let selected=base.join(path).canonicalize().map_err(|_|ApiError(StatusCode::NOT_FOUND,"Asset not found".into()))?;
    if !selected.starts_with(&base){return Err(ApiError(StatusCode::NOT_FOUND,"Asset not found".into()));}
    let mime=match selected.extension().and_then(|v|v.to_str()).unwrap_or(""){
        "css"=>"text/css; charset=utf-8","mjs"|"js"=>"application/javascript; charset=utf-8","svg"=>"image/svg+xml","woff2"=>"font/woff2","png"=>"image/png",_=>return Err(ApiError(StatusCode::NOT_FOUND,"Asset type not served".into()))};
    let bytes=std::fs::read(selected).map_err(|e|e.to_string())?;Ok(([(header::CONTENT_TYPE,mime)],bytes).into_response())
}
#[derive(Deserialize)]struct JobRequest{#[serde(default)]model_ids:Vec<String>,#[serde(default)]mode:String,sample_id:Option<String>,rows:Option<Vec<Vec<Option<f64>>>>,csv:Option<String>}
async fn create_job(State(state):State<Server>,body:Result<Json<JobRequest>,JsonRejection>)->Result<(StatusCode,Json<Value>),ApiError>{
    let request=payload(body)?;let info=state.app.catalog.summaries();let schema=&info.first().ok_or("No compatible models".to_string())?.feature_names;
    if usize::from(request.sample_id.is_some())+usize::from(request.rows.is_some())+usize::from(request.csv.is_some())!=1{return Err("Supply exactly one sample, prepared row batch or CSV".to_string().into());}
    let snapshot=if let Some(id)=request.sample_id {
        let sample=state.app.samples["rows"].as_array().ok_or("Samples unavailable".to_string())?.iter().find(|v|v["id"].as_str()==Some(&id)).ok_or("Sample does not belong to this schema".to_string())?;
        let row:Vec<Option<f64>>=serde_json::from_value(sample["values"].clone()).map_err(|e|e.to_string())?;
        input::snapshot(vec![row],schema,&format!("validation_sample:{id}"))?
    }else if let Some(csv)=request.csv {input::parse_csv(&csv,schema)?}else{input::snapshot(request.rows.unwrap_or_default(),schema,"prepared_rows")?};
    let ids=if request.mode=="extensive"{info.iter().map(|v|v.id.clone()).collect::<Vec<_>>()}else if request.mode=="selected"||request.mode.is_empty(){request.model_ids}else{return Err("Unknown analysis mode".to_string().into());};
    let mut unique=std::collections::BTreeSet::new();let mut models=Vec::new();
    for id in ids{if !unique.insert(id.clone()){continue;}let model=info.iter().find(|m|m.id==id).ok_or(format!("Unknown visible model {id}"))?;
        models.push((id,model.label.clone(),model.artifact_sha256.clone()));}
    let (job,new)=state.app.jobs.create(snapshot,models)?;
    if new{analysis::enqueue(state.app.clone(),job.clone());}
    Ok((StatusCode::ACCEPTED,Json(serde_json::to_value(job.view()).map_err(|e|e.to_string())?)))
}
async fn get_job(State(state):State<Server>,Path(id):Path<String>)->Result<Json<Value>,ApiError>{let job=state.app.jobs.get(&id)?;Ok(Json(serde_json::to_value(job.view()).map_err(|e|e.to_string())?))}
async fn cancel_job(State(state):State<Server>,Path(id):Path<String>)->Result<Json<Value>,ApiError>{let job=state.app.jobs.get(&id)?;job.cancel();Ok(Json(serde_json::to_value(job.view()).map_err(|e|e.to_string())?))}
#[derive(Deserialize)]struct EvidenceRequest{job_id:String,model_id:String,#[serde(default)]row_index:usize,method:Option<String>,explanation:Option<Value>}
async fn explain(State(state):State<Server>,body:Result<Json<EvidenceRequest>,JsonRejection>)->Result<Json<Value>,ApiError>{
    let request=payload(body)?;let method=Method::parse(request.method.as_deref().unwrap_or("shap"))?;
    let job=state.app.jobs.get(&request.job_id)?;let (row,prediction)=job.observation(&request.model_id,request.row_index)?;
    let permit=state.app.explanation_slots.clone().try_acquire_owned().map_err(|_|ApiError(StatusCode::TOO_MANY_REQUESTS,"Another explanation is running; predictions remain available".into()))?;
    let app=state.app.clone();
    let output=tokio::task::spawn_blocking(move||->Result<Value,String>{
        let _permit=permit;let model=app.catalog.load_model(&request.model_id).map_err(|e|e.to_string())?;
        let target=prediction["class_index"].as_u64().ok_or("Prediction has no target class")? as usize;
        let mut output=if method==Method::Shap {if let Some(tree)=model.tree_explanation(&row,target).map_err(|e|e.to_string())?{
            json!({"method":"shap","method_label":"Exact native TreeSHAP","target_label":model.info().classes[target],"scope":"selected_model_raw_class_margin","units":tree.units,
                "features":tree.contributions.iter().enumerate().map(|(i,v)|json!({"feature":model.info().feature_names[i],"value":row[i].is_finite().then_some(row[i]),"contribution":v})).collect::<Vec<_>>(),
                "quality":{"reconstruction_error":tree.reconstruction_error},"details":{"bias":tree.bias,"margin":tree.margin},
                "warnings":["Contributions reconstruct the selected model's raw class margin, not calibrated probability or causal effects."]})
        }else{generic_explanation(&app,&model,method,&row,target,&job.cancelled)?}}else{generic_explanation(&app,&model,method,&row,target,&job.cancelled)?};
        if job.cancelled.load(Ordering::Acquire){return Err("Explanation cancelled".into());}
        output["job_id"]=json!(request.job_id);output["input_id"]=json!(job.input.id);output["model_id"]=json!(request.model_id);output["row_index"]=json!(request.row_index);output["artifact_sha256"]=json!(model.info().artifact_sha256);
        let identity=format!("{:x}",Sha256::digest(serde_json::to_vec(&output).map_err(|e|e.to_string())?));output["explanation_id"]=json!(identity);
        let mut saved=app.explanations.lock().map_err(|_|"Explanation store unavailable")?;
        if saved.len()>=64{if let Some(key)=saved.keys().next().cloned(){saved.remove(&key);}}
        saved.insert(identity,output.clone());Ok(output)
    }).await.map_err(|e|ApiError(StatusCode::SERVICE_UNAVAILABLE,e.to_string()))?.map_err(|e|ApiError(StatusCode::SERVICE_UNAVAILABLE,e))?;
    Ok(Json(output))
}
fn generic_explanation(app:&Application,model:&netsec_models::LoadedModel,method:Method,row:&[f64],target:usize,cancel:&std::sync::atomic::AtomicBool)->Result<Value,String>{
    let settings=Settings{budget:64,seed:42,target_index:target,target_label:model.info().classes[target].clone(),units:if model.info().kind=="hem"{"class_decision_indicator"}else{"calibrated_class_probability"}.into()};
    let evaluate=|row:&[f64]|{let result=model.predict(row).map_err(|e|e.to_string())?;let score=result.probabilities.as_ref().map(|p|p[target]).unwrap_or_else(||usize::from(result.predicted==target) as f64);Ok(Evaluation{score,class_index:result.predicted})};
    netsec_explain::explain_prediction(method,row,&app.reference,&settings,&evaluate,cancel)
}
async fn english(State(state):State<Server>,body:Result<Json<EvidenceRequest>,JsonRejection>)->Result<Json<Value>,ApiError>{
    let request=payload(body)?;let job=state.app.jobs.get(&request.job_id)?;
    let explanation=if let Some(value)=request.explanation{
        let id=value["explanation_id"].as_str().ok_or("Use an explanation generated by this app session".to_string())?;
        let saved=state.app.explanations.lock().map_err(|_|"Explanation store unavailable".to_string())?.get(id).cloned().ok_or("Explanation expired; request it again".to_string())?;
        if saved["job_id"]!=request.job_id||saved["model_id"]!=request.model_id||saved["row_index"]!=request.row_index{return Err("Explanation does not belong to this input/model/row".to_string().into());}Some(saved)
    }else{None};
    let permit=state.app.language_slots.clone().try_acquire_owned().map_err(|_|ApiError(StatusCode::TOO_MANY_REQUESTS,"Local language model is busy".into()))?;
    let app=state.app.clone();let output=tokio::task::spawn_blocking(move||{let _permit=permit;crate::language::generate(&app,&job,&request.model_id,request.row_index,explanation.as_ref())}).await.map_err(|e|e.to_string())?
        .map_err(|e|ApiError(StatusCode::SERVICE_UNAVAILABLE,e))?;Ok(Json(output))
}
#[derive(Deserialize)]struct LiveStart{adapter_id:String,mode:String}
async fn live_start(State(state):State<Server>,body:Result<Json<LiveStart>,JsonRejection>)->Result<Json<Value>,ApiError>{let v=payload(body)?;Ok(Json(state.app.live.start(&v.adapter_id,&v.mode)?))}
#[derive(Deserialize)]struct LiveEnglish{observation_id:String,model_id:String}
async fn live_english(State(state):State<Server>,body:Result<Json<LiveEnglish>,JsonRejection>)->Result<Json<Value>,ApiError>{
    let request=payload(body)?;let prediction=state.app.live.evidence(&request.observation_id,&request.model_id)?;
    let permit=state.app.language_slots.clone().try_acquire_owned().map_err(|_|ApiError(StatusCode::TOO_MANY_REQUESTS,"Local language model is busy".into()))?;
    let app=state.app.clone();
    let output=tokio::task::spawn_blocking(move||->Result<Value,String>{
        let _permit=permit;let label=if request.model_id=="live_partial_xgboost"{"Partial-input XGBoost"}else{"Experimental packet-only XGBoost"};
        let output=crate::language::generate_evidence(&app,&prediction,label,prediction.get("explanation"),app.live.cancellation(),
            json!({"observation_id":request.observation_id,"model_id":request.model_id}))?;
        app.live.evidence(&request.observation_id,&request.model_id).map_err(|_|"The live observation changed during generation; request English for the current window".to_owned())?;Ok(output)
    }).await.map_err(|e|e.to_string())?.map_err(|e|ApiError(StatusCode::SERVICE_UNAVAILABLE,e))?;Ok(Json(output))
}
async fn window_action(Path(action):Path<String>)->Result<Json<Value>,ApiError>{
    if std::env::var_os("NETSEC_DESKTOP").is_none(){return Err("Native window controls require the desktop shell".to_string().into());}
    if !["minimize","toggle_maximize","request_close","cancel_close","confirm_close","start_drag"].contains(&action.as_str()){return Err("Unsupported fixed window action".to_string().into());}
    println!("{}",json!({"type":"window","action":action}));let _=std::io::stdout().flush();
    Ok(Json(json!({"accepted":true})))
}
async fn not_found()->impl IntoResponse{ApiError(StatusCode::NOT_FOUND,"Route not found".into())}

pub async fn serve(root:PathBuf,port:u16,parent_stdin:bool,browser:bool)->Result<(),String>{
    let app=Application::load(root)?;let listener=tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST,port)).await.map_err(|e|e.to_string())?;
    let port=listener.local_addr().map_err(|e|e.to_string())?.port();
    let token=if let Ok(token)=std::env::var("IMDAI_SESSION_TOKEN"){
        if token.len()!=64||!token.bytes().all(|v|v.is_ascii_hexdigit()){return Err("Desktop session token must be 64 hexadecimal characters".into());}token
    }else{let mut bytes=[0u8;32];getrandom::fill(&mut bytes).map_err(|e|e.to_string())?;bytes.iter().map(|v|format!("{v:02x}")).collect()};
    let browser_url=browser.then(||format!("http://127.0.0.1:{port}/bootstrap?session={token}"));
    let state=Server{app:app.clone(),token,port};
    let mut router=Router::new().route("/bootstrap",get(bootstrap)).route("/static/{*path}",get(asset))
        .route("/api/v1/catalog",get(|State(s):State<Server>|async move{Json(s.app.catalog_view())}))
        .route("/api/v1/samples",get(|State(s):State<Server>|async move{Json(s.app.samples_view())}))
        .route("/api/v1/jobs",post(create_job)).route("/api/v1/jobs/{id}",get(get_job)).route("/api/v1/jobs/{id}/cancel",post(cancel_job))
        .route("/api/v1/explain",post(explain)).route("/api/v1/english",post(english))
        .route("/api/v1/live/status",get(|State(s):State<Server>|async move{Json(s.app.live.status())}))
        .route("/api/v1/live/adapters",get(|State(s):State<Server>|async move{match s.app.live.adapters(){Ok(v)=>Json(v).into_response(),Err(e)=>ApiError(StatusCode::SERVICE_UNAVAILABLE,e).into_response()}}))
        .route("/api/v1/live/start",post(live_start)).route("/api/v1/live/stop",post(|State(s):State<Server>|async move{Json(s.app.live.stop())}))
        .route("/api/v1/live/english",post(live_english))
        .route("/api/v1/window",get(|State(s):State<Server>|async move{Json(s.app.window.lock().unwrap_or_else(|e|e.into_inner()).clone())}))
        .route("/api/v1/window/{action}",post(window_action))
        .route("/api/v1/settings",get(|State(s):State<Server>|async move{Json(json!({"application":"NetSec","version":"0.4.0","backend":"Rust","python_inference":false,"offline":true,"cpu_only":true,"retained_jobs":16,"max_rows":500,"max_csv_bytes":10485760,"language":crate::language::status(&s.app.root),"models":s.app.catalog.summaries().len(),"data_retention":"Session memory only; export is explicit","test_loaded":false}))}));
    for route in ["/","/models","/samples","/csv","/live","/settings","/help"]{router=router.route(route,get(page));}
    let router=router.fallback(not_found).layer(DefaultBodyLimit::max(12*1024*1024)).layer(middleware::from_fn_with_state(state.clone(),protect)).with_state(state);
    if parent_stdin{let app=app.clone();std::thread::spawn(move||{
        for line in std::io::stdin().lock().lines(){let Ok(line)=line else{break;};if line=="shutdown"{break;}
            if line.len()<4096{if let Ok(v)=serde_json::from_str::<Value>(&line){if v["type"]=="window_state"{*app.window.lock().unwrap_or_else(|e|e.into_inner())=v["state"].clone();}}}
        }
        app.stopping.store(true,Ordering::Release);app.jobs.cancel_all();app.live.stop();app.shutdown.notify_waiters();
    });}
    let closing=app.clone();tokio::spawn(async move{let _=tokio::signal::ctrl_c().await;closing.stopping.store(true,Ordering::Release);closing.jobs.cancel_all();closing.live.stop();closing.shutdown.notify_waiters();});
    println!("{}",json!({"type":"ready","port":port,"pid":std::process::id(),"backend":"rust"}));let _=std::io::stdout().flush();
    if let Some(url)=browser_url{open_browser(&url)?;}
    let shutdown = app.clone();
    axum::serve(listener,router).with_graceful_shutdown(async move {
        let signal = shutdown.shutdown.notified();
        tokio::pin!(signal);
        signal.as_mut().enable();
        // Registration precedes the flag check, covering both early EOF and a
        // notification concurrent with checking the retained stopping state.
        if !shutdown.stopping.load(Ordering::Acquire) { signal.await; }
    }).await.map_err(|e|e.to_string())?;
    app.stopping.store(true,Ordering::Release);app.jobs.cancel_all();app.live.stop();Ok(())
}

#[cfg(windows)]
fn open_browser(url:&str)->Result<(),String>{
    use windows_sys::Win32::UI::{Shell::ShellExecuteW,WindowsAndMessaging::SW_SHOWNORMAL};
    let action="open\0".encode_utf16().collect::<Vec<_>>();let url=url.encode_utf16().chain(Some(0)).collect::<Vec<_>>();
    // SAFETY: both UTF-16 strings are terminated and live through the call; the
    // operating system owns the explicitly requested default browser launch.
    let result=unsafe{ShellExecuteW(std::ptr::null_mut(),action.as_ptr(),url.as_ptr(),std::ptr::null(),std::ptr::null(),SW_SHOWNORMAL)};
    if result as isize<=32{Err("Could not open the default browser".into())}else{Ok(())}
}
#[cfg(not(windows))]
fn open_browser(_url:&str)->Result<(),String>{Err("Automatic browser opening is currently supported on Windows".into())}
