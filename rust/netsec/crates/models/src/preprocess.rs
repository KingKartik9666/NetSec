use crate::{invalid, ModelError, tensor::Tensors};
use serde_json::Value;

pub fn interpolation(x:f64, knots:&[f64], positions:&[f64])->f64 {
    if x < knots[0] {return positions[0]}
    let i=knots.partition_point(|k|*k<=x);
    if i==knots.len(){return positions[positions.len()-1]}
    if i==0{return positions[0]}
    let lo=i-1;
    positions[lo]+(positions[i]-positions[lo])*(x-knots[lo])/(knots[i]-knots[lo])
}

pub fn quantile_uniform(x:f64,knots:&[f64],positions:&[f64])->f64 {
    if x<=knots[0]{return 0.0}
    if x>=knots[knots.len()-1]{return 1.0}
    let hi=knots.partition_point(|k|*k<=x);
    let lo=knots.partition_point(|k|*k<x);
    if hi>lo {return (positions[lo]+positions[hi-1])*0.5}
    interpolation(x,knots,positions)
}

/// Acklam's rational inverse normal CDF, maximum relative error below 1.2e-9.
pub fn normal_quantile(p:f64)->f64 {
    const A:[f64;6]=[-3.969683028665376e1,2.209460984245205e2,-2.759285104469687e2,1.383577518672690e2,-3.066479806614716e1,2.506628277459239];
    const B:[f64;5]=[-5.447609879822406e1,1.615858368580409e2,-1.556989798598866e2,6.680131188771972e1,-1.328068155288572e1];
    const C:[f64;6]=[-7.784894002430293e-3,-3.223964580411365e-1,-2.400758277161838,-2.549732539343734,4.374664141464968,2.938163982698783];
    const D:[f64;4]=[7.784695709041462e-3,3.224671290700398e-1,2.445134137142996,3.754408661907416];
    if p<0.02425 || p>1.0-0.02425 {
        let q=(-2.0*(if p<0.5{p}else{1.0-p}).ln()).sqrt();
        let numerator=((((C[0]*q+C[1])*q+C[2])*q+C[3])*q+C[4])*q+C[5];
        let denominator=(((D[0]*q+D[1])*q+D[2])*q+D[3])*q+1.0;
        let value=numerator/denominator;
        if p<0.5{value}else{-value}
    }else{
        let q=p-0.5;let r=q*q;
        (((((A[0]*r+A[1])*r+A[2])*r+A[3])*r+A[4])*r+A[5])*q /
            (((((B[0]*r+B[1])*r+B[2])*r+B[3])*r+B[4])*r+1.0)
    }
}

fn array(value:&Value)->Result<Vec<f64>,ModelError>{value.as_array().ok_or_else(||invalid("Expected numeric array"))?.iter()
    .map(|v|v.as_f64().filter(|v|v.is_finite()).ok_or_else(||invalid("Invalid preprocessing number"))).collect()}

pub fn transform(row:&[f64],spec:&Value,tensors:&Tensors)->Result<Vec<f32>,ModelError>{
    let mut x:Vec<f32>=row.iter().map(|v|*v as f32).collect();
    match spec["kind"].as_str(){
        Some("identity")=>Ok(x),
        Some("tabm_v2")=>{
            let columns=spec["columns"].as_array().ok_or_else(||invalid("Missing TabM columns"))?;
            if columns.len()!=row.len(){return Err(invalid("Preprocessor column count differs"))}
            let mut mask=Vec::with_capacity(x.len());
            for (i,column) in columns.iter().enumerate(){
                let raw=x[i] as f64;let finite=raw.is_finite();mask.push(if finite{0.0}else{1.0});
                if column["flag"].as_bool()==Some(true){
                    if finite && raw!=0.0 && raw!=1.0{return Err(invalid("Nonbinary indicator"))}
                    x[i]=if finite{raw as f32}else{0.0};continue
                }
                let clean=if finite{raw}else{column["median"].as_f64().ok_or_else(||invalid("Missing median"))?};
                let v=if spec["scaling"]=="log_robust" {
                    let center=column["center"].as_f64().ok_or_else(||invalid("Missing center"))?;
                    let scale=column["scale"].as_f64().filter(|v|*v>0.0).ok_or_else(||invalid("Invalid scale"))?;
                    ((clean.signum()*clean.abs().ln_1p()-center)/scale).clamp(-30.0,30.0)
                }else{
                    let knots=array(&column["knots"])?;
                    if knots.len()<2{(clean-column["median"].as_f64().ok_or_else(||invalid("Missing median"))?).clamp(-30.0,30.0)}else{
                        let positions=array(&column["quantiles"])?;
                        if positions.len()!=knots.len(){return Err(invalid("Quantile sizes differ"))}
                        normal_quantile(interpolation(clean,&knots,&positions).clamp(1e-4,1.0-1e-4))
                    }
                };
                x[i]=v as f32;
            }
            x.extend(mask);Ok(x)
        }
        Some("pipeline")=>{
            for step in spec["steps"].as_array().ok_or_else(||invalid("Missing pipeline steps"))? {
                let field=|key:&str|step[key].as_str().ok_or_else(||invalid("Missing pipeline tensor name"));
                match step["kind"].as_str(){
                    Some("imputer")=>{
                        let statistics=tensors.numbers(field("statistics")?)?;
                        if statistics.len()!=x.len(){return Err(invalid("Imputer shape differs"))}
                        for (value,median) in x.iter_mut().zip(statistics){if !value.is_finite(){*value=median as f32}}
                    }
                    Some("signed_log1p")=>for value in &mut x{*value=value.signum()*value.abs().ln_1p()},
                    Some("sklearn_quantile")=>{
                        if step["distribution"]!="uniform"{return Err(invalid("Only frozen uniform sklearn quantiles supported"))}
                        let quantiles=tensors.f64(field("quantiles")?)?;
                        let references=tensors.f64(field("references")?)?;
                        let count=references.len();
                        if count<2 || quantiles.len()!=x.len()*count{return Err(invalid("Quantile tensor shape differs"))}
                        for (i,value) in x.iter_mut().enumerate(){*value=quantile_uniform(*value as f64,&quantiles[i*count..(i+1)*count],references) as f32;}
                    }
                    Some("minmax")=>{
                        let scale=tensors.f32(field("scale")?)?;let offset=tensors.f32(field("min")?)?;
                        let range=array(&step["range"])?;
                        if scale.len()!=x.len() || offset.len()!=x.len() || range.len()!=2{return Err(invalid("MinMax shape differs"))}
                        for i in 0..x.len(){x[i]*=scale[i];x[i]+=offset[i];if step["clip"].as_bool()==Some(true){x[i]=x[i].clamp(range[0] as f32,range[1] as f32)}}
                    }
                    Some("pca")=>{
                        let mean=tensors.f32(field("mean")?)?;let components=tensors.f32(field("components")?)?;
                        if mean.len()!=x.len() || components.len()%x.len()!=0{return Err(invalid("PCA shape differs"))}
                        // The saved pipeline uses float32 matrix products. Keep float32
                        // accumulation before subtracting the separately projected mean:
                        // float64 accumulation can change near-coincident KNN neighbors.
                        x=components.chunks(x.len()).map(|w|{
                            let product=w.iter().zip(&x).fold(0.0f32,|sum,(w,x)|sum+*w**x);
                            let projected=w.iter().zip(mean).fold(0.0f32,|sum,(w,x)|sum+*w**x);
                            product-projected
                        }).collect();
                    }
                    _=>return Err(invalid("Unsupported pipeline step"))
                }
            }
            if x.iter().any(|v|!v.is_finite()){return Err(invalid("Nonfinite preprocessed vector"))}Ok(x)
        }
        _=>Err(invalid("Unsupported preprocessor"))
    }
}
