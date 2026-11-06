//! A single damaged or incompatible model must not disable independent models.
use netsec_models::ModelCatalog;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{fs,path::PathBuf,time::{SystemTime,UNIX_EPOCH}};

struct Fixture { root:PathBuf }
impl Fixture {
    fn new()->Self{
        let unique=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root=std::env::temp_dir().join(format!("netsec-model-isolation-{}-{unique}",std::process::id()));
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
        self.write_manifest(id,manifest)
    }
    fn write_manifest(&self,id:&str,manifest:Value)->Value{
        let directory=self.root.join(id);fs::create_dir_all(&directory).unwrap();
        let bytes=serde_json::to_vec(&manifest).unwrap();fs::write(directory.join("model.json"),&bytes).unwrap();
        json!({"id":id,"kind":manifest["kind"],"label":id,"manifest":format!("{id}/model.json"),"sha256":format!("{:x}",Sha256::digest(&bytes)),"auxiliary":false})
    }
    fn catalog(&self,entries:Vec<Value>){fs::write(self.root.join("catalog.json"),serde_json::to_vec(&json!({"version":1,"feature_names":["a","b"],"classes":["benign","attack"],"benign_index":0,"models":entries})).unwrap()).unwrap()}
    fn hybrid(&self,id:&str,branches:Vec<&str>)->Value{
        let weight=1.0/branches.len() as f64;
        self.write_manifest(id,json!({"version":1,"id":id,"kind":"hybrid","label":id,"feature_names":["a","b"],"classes":["benign","attack"],"benign_index":0,
          "files":[],"tensors":{},"branches":branches,"calibration":{"temperature":1.0},
          "combiner":{"kind":"global","names":branches,"n_classes":2,"weights":vec![weight;branches.len()]}}))
    }
}
impl Drop for Fixture{fn drop(&mut self){let _=fs::remove_dir_all(&self.root);}}

#[test]
fn missing_manifest_preserves_other_model_predictions_and_metadata(){
    let fixture=Fixture::new();let good=fixture.model("good");let broken=fixture.model("broken");
    fs::remove_file(fixture.root.join("broken/model.json")).unwrap();fixture.catalog(vec![good,broken]);
    let catalog=ModelCatalog::load(&fixture.root).expect("independent model must remain available");
    assert_eq!(catalog.summaries().len(),2);assert_eq!(catalog.info("broken").unwrap().label,"broken");
    assert_eq!(catalog.availability("broken").unwrap().status,"unavailable");
    assert_eq!(catalog.availability("good").unwrap().status,"ready");
    assert!(catalog.load_model("broken").is_err());assert!(catalog.load_model("good").unwrap().predict(&[1.0,0.0]).is_ok());
}

#[test]
fn checksum_failure_is_scoped_and_dependent_hybrid_is_unavailable(){
    let fixture=Fixture::new();let good=fixture.model("good");let damaged=fixture.model("damaged");
    let hybrid=fixture.hybrid("hybrid",vec!["good","damaged"]);
    fs::write(fixture.root.join("damaged/model.json"),"{}").unwrap();fixture.catalog(vec![good,damaged,hybrid]);
    let catalog=ModelCatalog::load(&fixture.root).expect("damage must remain scoped");
    assert_eq!(catalog.summaries().len(),3);assert_ne!(catalog.availability("damaged").unwrap().status,"ready");
    assert_ne!(catalog.availability("hybrid").unwrap().status,"ready");
    assert_eq!(catalog.availability("good").unwrap().status,"ready");
}

#[test]
fn hybrid_cycles_and_invalid_weights_fail_without_recursing(){
    let fixture=Fixture::new();let good=fixture.model("good");let first=fixture.hybrid("first",vec!["second"]);let second=fixture.hybrid("second",vec!["first"]);
    fixture.catalog(vec![good,first,second]);let catalog=ModelCatalog::load(&fixture.root).unwrap();
    assert!(catalog.availability("first").unwrap().reason.unwrap().contains("cycle"));assert!(catalog.load_model("first").is_err());
    assert!(catalog.load_model("good").unwrap().predict(&[1.0,0.0]).is_ok());
}

#[test]
fn invalid_hybrid_weights_are_scoped_to_the_hybrid(){
    let fixture=Fixture::new();let good=fixture.model("good");fixture.hybrid("bad_weights",vec!["good"]);
    let mut value:Value=serde_json::from_slice(&fs::read(fixture.root.join("bad_weights/model.json")).unwrap()).unwrap();
    value["combiner"]["weights"]=json!([0.5]);let hybrid=fixture.write_manifest("bad_weights",value);
    fixture.catalog(vec![good,hybrid]);let catalog=ModelCatalog::load(&fixture.root).unwrap();
    assert!(catalog.availability("bad_weights").unwrap().reason.unwrap().contains("sum to one"));
    assert_eq!(catalog.availability("good").unwrap().status,"ready");
}
