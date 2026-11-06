//! TEST ONLY: the base-vs-new verifier harness (replaces the review's out-of-tree scratch copy). `base_check` runs the frozen base verifier
//! (`basecopy`) on a real matrix sample and method; `probe_harness` (ignored) reads PROBES=<json list of {sample, method, text}> or of
//! [sample, method, text] and writes <PROBES>.out.json with both verdicts and a NEW-ACCEPT tag for every base-rejected sentence now accepted.
use serde_json::{json,Value};
use std::path::Path;
fn catalog()->(Vec<String>,Vec<String>){
    let dir=Path::new(env!("CARGO_MANIFEST_DIR"));
    let catalog:Value=serde_json::from_slice(&std::fs::read(dir.join("../../../../artifacts/application/native_models_v1/catalog.json")).unwrap()).unwrap();
    let strings=|key:&str|catalog[key].as_array().unwrap().iter().map(|v|v.as_str().unwrap().to_string()).collect::<Vec<_>>();
    (strings("classes"),strings("feature_names"))
}
/// The base verifier's verdict on a sentence for a real matrix sample with one of its real XAI explanations.
pub(crate) fn base_check(sample:&str,method:&str,sentence:&str)->Result<(),String>{
    let dir=Path::new(env!("CARGO_MANIFEST_DIR"));
    let data:Value=serde_json::from_slice(&std::fs::read(dir.join("src/language/testdata/matrix_methods.json")).unwrap()).unwrap();
    let (classes,features)=catalog();let v=&data[sample];
    let p=super::basecopy::packet::build(&v["prediction"],"XGBoost",classes.len(),Some(&v["explanations"][method]));
    let r=super::basecopy::verify::rules(&p,&features,&super::basecopy::packet::lexicon(&classes));
    super::basecopy::verify::check(sentence,&r)
}
/// A probe as `{sample, method, text}` or `[sample, method, text]`.
pub(crate) fn probe(p:&Value)->(String,String,String){
    let s=|k:&str,i:usize|p.get(k).or_else(||p.get(i)).and_then(Value::as_str).unwrap_or_default().to_string();
    (s("sample",0),s("method",1),s("text",2))
}
#[test]#[ignore="reads PROBES=<json> and writes <PROBES>.out.json; run with -- --ignored"]fn probe_harness(){
    let path=std::env::var("PROBES").expect("PROBES");
    let probes:Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let (mut out,mut accepted)=(Vec::new(),0);
    for p in probes.as_array().unwrap(){
        let (s,m,t)=probe(p);let ev=super::matrix_method_evidence(&s,&m);
        let (b,n)=(base_check(&s,&m,&t),super::verify::check(&t,&ev.rules));
        let tag=match(&b,&n){(Err(_),Ok(()))=>"NEW-ACCEPT",(Ok(()),Ok(()))=>"both-accept",(Err(_),Err(_))=>"both-reject",(Ok(()),Err(_))=>"new-reject"};
        accepted+=usize::from(tag=="NEW-ACCEPT");
        out.push(json!({"tag":tag,"sample":s,"method":m,"text":t,"base":format!("{b:?}"),"new":format!("{n:?}"),"names_result":super::verify::names_result(&t,&ev.rules),"defines_method":super::verify::defines_method(&t,&ev.rules)}));
    }
    println!("{} probes, NEW-ACCEPT {accepted}",out.len());
    std::fs::write(format!("{path}.out.json"),serde_json::to_vec_pretty(&out).unwrap()).unwrap();
}
