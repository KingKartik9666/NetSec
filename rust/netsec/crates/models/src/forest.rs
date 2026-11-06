use crate::{invalid,ModelError,tensor::Tensors};
use serde_json::Value;

pub fn predict(manifest:&Value,tensors:&Tensors,row:&[f64])->Result<Vec<f64>,ModelError>{
    let spec=&manifest["forest"];
    let median=tensors.numbers(spec["imputer"].as_str().ok_or_else(||invalid("Missing RF imputer"))?)?;
    let classes=manifest["classes"].as_array().ok_or_else(||invalid("Missing RF classes"))?.len();
    if median.len()!=row.len(){return Err(invalid("RF input dimensions differ"))}
    let x:Vec<f64>=row.iter().zip(median).map(|(v,m)|if v.is_nan(){m}else{(*v as f32) as f64}).collect();
    let trees=spec["trees"].as_array().filter(|t|!t.is_empty()&&t.len()<=10000).ok_or_else(||invalid("Invalid RF tree count"))?;
    let mut probability=vec![0.0f64;classes];
    for tree in trees{
        let prefix=tree["prefix"].as_str().ok_or_else(||invalid("Missing RF tree prefix"))?;
        let left=tensors.i32(&format!("{prefix}.left"))?;let right=tensors.i32(&format!("{prefix}.right"))?;
        let feature=tensors.i32(&format!("{prefix}.feature"))?;let threshold=tensors.f64(&format!("{prefix}.threshold"))?;
        let leaf=tensors.i32(&format!("{prefix}.leaf_index"))?;
        if right.len()!=left.len()||feature.len()!=left.len()||threshold.len()!=left.len()||leaf.len()!=left.len(){return Err(invalid("RF tree array lengths differ"))}
        let pointers=tensors.u32(&format!("{prefix}.leaf_pointers"))?;
        let labels=tensors.u32(&format!("{prefix}.leaf_classes"))?;
        let probabilities=tensors.f64(&format!("{prefix}.leaf_probabilities"))?;
        if labels.len()!=probabilities.len(){return Err(invalid("RF sparse leaves differ"))}
        let mut index=0usize;let mut found=false;
        for _ in 0..left.len(){
            let l=*left.get(index).ok_or_else(||invalid("RF tree child out of bounds"))?;
            if l==-1{
                let position=usize::try_from(leaf[index]).map_err(|_|invalid("Invalid RF leaf index"))?;
                let begin=*pointers.get(position).ok_or_else(||invalid("Invalid RF leaf pointer"))? as usize;
                let end=*pointers.get(position+1).ok_or_else(||invalid("Invalid RF leaf pointer"))? as usize;
                if end<begin||end>labels.len(){return Err(invalid("RF sparse leaf range differs"))}
                for i in begin..end{let c=labels[i] as usize;if c>=classes{return Err(invalid("RF leaf class outside schema"))}probability[c]+=probabilities[i]}
                found=true;break
            }
            let f=usize::try_from(feature[index]).map_err(|_|invalid("Invalid RF split feature"))?;
            let value=*x.get(f).ok_or_else(||invalid("RF split feature outside schema"))?;
            index=usize::try_from(if value<=threshold[index]{l}else{right[index]}).map_err(|_|invalid("Invalid RF tree child"))?;
        }
        if !found{return Err(invalid("RF tree cycle"))}
    }
    for p in &mut probability{*p/=trees.len() as f64}
    Ok(probability)
}
