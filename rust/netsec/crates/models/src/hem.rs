use crate::{invalid,ModelError,tensor::Tensors,neural};
use serde_json::Value;
use serde::Serialize;

#[derive(Clone,Debug,Serialize)]
pub struct HemDetails {
    pub predicted:usize,pub confidence:f64,pub multiclass_probabilities:Vec<f64>,pub binary_probability:f64,
    pub knn_prediction:usize,pub neighbor_indices:Vec<usize>,pub neighbor_distances:Vec<f32>,pub transformed:Vec<f32>,
}

fn widths(manifest:&Value,key:&str,last:usize)->Result<Vec<usize>,ModelError>{
    let mut sizes:Vec<usize>=manifest["config"][key].as_array().ok_or_else(||invalid("Missing HEM dimensions"))?.iter()
        .map(|v|v.as_u64().filter(|v|*v>0&&*v<=4096).map(|v|v as usize).ok_or_else(||invalid("Invalid HEM dimension"))).collect::<Result<_,_>>()?;
    if sizes.len()>16{return Err(invalid("HEM depth exceeds bound"))}sizes.push(last);Ok(sizes)
}

pub fn predict(manifest:&Value,tensors:&Tensors,x:Vec<f32>)->Result<HemDetails,ModelError>{
    let classes=manifest["classes"].as_array().ok_or_else(||invalid("Missing HEM classes"))?.len();
    let multiclass=neural::softmax(&neural::mlp(tensors,"multiclass",&x,&widths(manifest,"hidden",classes)?)?);
    let binary=neural::sigmoid(neural::mlp(tensors,"binary",&x,&widths(manifest,"binary_hidden",1)?)?[0] as f64) as f32;
    let multi=multiclass.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1).then_with(||b.0.cmp(&a.0))).map(|(i,_)|i).ok_or_else(||invalid("Empty HEM probabilities"))?;
    let k=manifest["config"]["knn_k"].as_u64().filter(|v|*v>0&&*v<=32).ok_or_else(||invalid("Invalid HEM neighbors"))? as usize;
    let references=tensors.f32("knn.references")?;let labels=tensors.i32("knn.labels")?;
    if x.is_empty()||references.len()!=labels.len()*x.len()||labels.len()<k{return Err(invalid("HEM reference dimensions differ"))}
    let mut neighbors:Vec<(f32,usize)>=Vec::with_capacity(k+1);
    for (index,reference) in references.chunks_exact(x.len()).enumerate(){
        let distance=reference.iter().zip(&x).map(|(a,b)|{let d=(*a-*b) as f64;d*d}).sum::<f64>().sqrt() as f32;
        if neighbors.len()<k||distance<neighbors[k-1].0{
            let position=neighbors.partition_point(|(d,i)|*d<distance||(*d==distance&&*i<index));
            neighbors.insert(position,(distance,index));if neighbors.len()>k{neighbors.pop();}
        }
    }
    let mut counts=vec![0usize;classes];
    for &(_,index) in &neighbors{let c=labels[index];if c<0||c as usize>=classes{return Err(invalid("HEM neighbor class outside schema"))}counts[c as usize]+=1}
    let knn=counts.iter().enumerate().max_by(|a,b|a.1.cmp(b.1).then_with(||b.0.cmp(&a.0))).map(|(i,_)|i).ok_or_else(||invalid("Empty HEM vote"))?;
    let predicted=if binary<0.5&&(multi==0||knn==0){0}else if knn!=0{knn}else{multi};
    let matching=if predicted==0{1.0-binary}else{binary};
    let features=[(multiclass[predicted] as f32).clamp(1e-12,1.0).ln() as f64,
                  matching.clamp(1e-12,1.0).ln() as f64,if knn==multi{1.0}else{0.0}];
    let calibrator=&manifest["correctness_calibrator"];
    let coefficients=calibrator["coefficients"].as_array().filter(|v|v.len()==3).ok_or_else(||invalid("Missing HEM correctness coefficients"))?;
    let mut z=calibrator["intercept"].as_f64().ok_or_else(||invalid("Missing HEM correctness intercept"))?;
    for i in 0..3{z+=features[i]*coefficients[i].as_f64().ok_or_else(||invalid("Invalid HEM correctness coefficient"))?}
    Ok(HemDetails{predicted,confidence:neural::sigmoid(z),multiclass_probabilities:multiclass,binary_probability:binary as f64,
        knn_prediction:knn,neighbor_indices:neighbors.iter().map(|(_,i)|*i).collect(),neighbor_distances:neighbors.iter().map(|(d,_)|*d).collect(),transformed:x})
}
