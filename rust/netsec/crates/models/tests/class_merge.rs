//! The catalog class merge (ping sweep into recon scan) matches the training chat's reference
//! (artifacts/training/class_merge_20261101) and is applied by every model the catalog loads.
use netsec_models::{argmax,ClassMerge,ModelCatalog};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{fs,path::{Path,PathBuf},time::{SystemTime,UNIX_EPOCH}};

fn reference(name:&str)->Value{
    let path=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../artifacts/training/class_merge_20261101").join(name);
    serde_json::from_slice(&fs::read(&path).unwrap_or_else(|e|panic!("{}: {e}",path.display()))).unwrap()
}
fn vector(value:&Value)->Vec<f64>{value.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()}

#[test]
fn merge_matches_golden_vectors_and_decisions(){
    let spec=reference("spec.json");let golden=reference("golden.json");
    let classes:Vec<String>=serde_json::from_value(spec["classes"].clone()).unwrap();
    let merge=ClassMerge::parse(&json!({"source_class":spec["source_class"],"target_class":spec["target_class"]}),&classes).unwrap().unwrap();
    assert_eq!((merge.source as u64,merge.target as u64),(golden["source_index"].as_u64().unwrap(),golden["target_index"].as_u64().unwrap()));
    let tolerance=golden["tolerance_abs"].as_f64().unwrap();
    let (mut checked,mut changed)=(0,0);
    for row in golden["rows"].as_array().unwrap(){
        for model in ["tabm_ne","default_blend_tabm_xgboost_kan"]{
            let case=&row[model];let before=vector(&case["before"]);let after=vector(&case["after"]);
            assert_eq!(classes[argmax(&before)],case["decision_before"].as_str().unwrap(),"row {} {model}",row["validation_row"]);
            let mut ours=before.clone();merge.apply(&mut ours);
            for (a,b) in ours.iter().zip(&after){assert!((a-b).abs()<=tolerance,"row {} {model}: {a} vs {b}",row["validation_row"]);}
            assert_eq!(classes[argmax(&ours)],case["decision_after"].as_str().unwrap(),"row {} {model}",row["validation_row"]);
            if case["decision_before"]!=case["decision_after"]{changed+=1}
            checked+=1;
        }
    }
    assert_eq!(checked,golden["rows"].as_array().unwrap().len()*2);
    assert!(changed>0,"the reference must include decisions the merge changes");
}

#[test]
fn merge_rules(){
    let classes:Vec<String>=["benign","recon_scan","recon_ping-sweep"].map(String::from).to_vec();
    assert_eq!(ClassMerge::parse(&Value::Null,&classes).unwrap(),None);
    let merge=ClassMerge::parse(&json!({"source_class":"recon_ping-sweep","target_class":"recon_scan"}),&classes).unwrap().unwrap();
    let mut p=vec![0.4,0.25,0.35];merge.apply(&mut p);
    assert!((p[1]-0.6).abs()<1e-12&&p[2]==0.0&&p[0]==0.4,"{p:?}");assert_eq!(argmax(&p),1);
    let mut tie=vec![0.5,0.25,0.25];merge.apply(&mut tie);assert_eq!(argmax(&tie),0,"ties keep the lowest index");
    assert_eq!((merge.vote(2),merge.vote(1),merge.vote(0)),(1,1,0));
    for bad in [json!({"source_class":"recon_scan","target_class":"recon_scan"}),json!({"source_class":"unknown","target_class":"recon_scan"}),json!({"target_class":"recon_scan"})]{
        assert!(ClassMerge::parse(&bad,&classes).is_err(),"{bad}");
    }
}

struct Fixture { root:PathBuf }
impl Fixture {
    fn new()->Self{
        let unique=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root=std::env::temp_dir().join(format!("netsec-class-merge-{}-{unique}",std::process::id()));
        fs::create_dir_all(&root).unwrap();Self{root}
    }
    fn model(&self,id:&str)->Value{
        let directory=self.root.join(id);fs::create_dir_all(&directory).unwrap();
        let numbers=[1.0f32,-1.0,-1.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,-1.0,1.0,-1.0,1.0];
        let blob:Vec<u8>=numbers.iter().flat_map(|n|n.to_le_bytes()).collect();fs::write(directory.join("weights.bin"),&blob).unwrap();
        let manifest=json!({"version":1,"id":id,"kind":"kan","label":id,"feature_names":["a","b"],"classes":["benign","attack"],
          "benign_index":0,"supports_partial":false,"calibration":{"temperature":1.0},"config":{"hidden":[],"spline_order":0,"grid_size":1},
          "preprocessor":{"kind":"identity"},"files":[{"path":"weights.bin","bytes":blob.len(),"sha256":format!("{:x}",Sha256::digest(&blob))}],
          "tensors":{
           "kan.layers.0.base_weight":{"dtype":"f32","shape":[2,2],"offset_bytes":0,"length_bytes":16},
           "kan.layers.0.spline_weight":{"dtype":"f32","shape":[2,2,1],"offset_bytes":16,"length_bytes":16},
           "kan.layers.0.spline_scaler":{"dtype":"f32","shape":[2,2],"offset_bytes":32,"length_bytes":16},
           "kan.layers.0.grid":{"dtype":"f32","shape":[2,2],"offset_bytes":48,"length_bytes":16}}});
        let bytes=serde_json::to_vec(&manifest).unwrap();fs::write(directory.join("model.json"),&bytes).unwrap();
        json!({"id":id,"kind":"kan","label":id,"manifest":format!("{id}/model.json"),"sha256":format!("{:x}",Sha256::digest(&bytes)),"auxiliary":false})
    }
    fn catalog(&self,entries:Vec<Value>,merge:Value){
        let mut catalog=json!({"version":1,"feature_names":["a","b"],"classes":["benign","attack"],"benign_index":0,"models":entries});
        if !merge.is_null(){catalog["class_merge"]=merge;}
        fs::write(self.root.join("catalog.json"),serde_json::to_vec(&catalog).unwrap()).unwrap();
    }
}
impl Drop for Fixture{fn drop(&mut self){let _=fs::remove_dir_all(&self.root);}}

#[test]
fn catalog_merge_reaches_loaded_model_predictions(){
    let row=[0.0,1.0];
    let plain=Fixture::new();let entry=plain.model("m");plain.catalog(vec![entry],Value::Null);
    let catalog=ModelCatalog::load(&plain.root).unwrap();assert!(catalog.class_merge().is_none());
    let before=catalog.load_model("m").unwrap().predict(&row).unwrap();
    let merged=Fixture::new();let entry=merged.model("m");merged.catalog(vec![entry],json!({"source_class":"attack","target_class":"benign"}));
    let catalog=ModelCatalog::load(&merged.root).unwrap();assert_eq!(catalog.class_merge(),Some(&ClassMerge{source:1,target:0}));
    let after=catalog.load_model("m").unwrap().predict(&row).unwrap();
    let (b,a)=(before.probabilities.unwrap(),after.probabilities.unwrap());
    assert!((a[0]-(b[0]+b[1])).abs()<1e-12&&a[1]==0.0,"{b:?} -> {a:?}");
    assert_eq!((after.predicted,after.confidence),(0,Some(a[0])));
    assert_eq!(after.raw_probabilities,before.raw_probabilities,"raw model output stays unmerged");
    assert_eq!(after.attack_probability,before.attack_probability);
}

#[test]
fn invalid_catalog_merge_is_refused(){
    let fixture=Fixture::new();let entry=fixture.model("m");
    fixture.catalog(vec![entry],json!({"source_class":"attack","target_class":"missing"}));
    assert!(ModelCatalog::load(&fixture.root).is_err());
}
