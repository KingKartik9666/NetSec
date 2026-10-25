//! Grouped reference games and a measured local linear surrogate.
use crate::{constraints::{self,Prepared}, envelope, feature, group_details, math::{self,Rng}, Engine, Reference};
use serde_json::{json,Value};

pub(crate) fn shap(row:&[f64],reference:&Reference,prepared:&Prepared,engine:&mut Engine<'_>,rng:&mut Rng)->Result<Value,String>{
    let groups=constraints::coalesce(&prepared.groups,8.min(engine.remaining().saturating_sub(2)).max(1));
    let output=engine.score(row)?;let mut totals=vec![0.;groups.len()];let mut estimates=vec![Vec::new();groups.len()];
    let mut baselines=Vec::new();let mut attempts=0;
    while engine.remaining()>=groups.len()+1 && attempts<128 {
        engine.check()?;attempts+=1;
        if prepared.valid_rows.is_empty(){break;}
        let donor=&reference.rows[prepared.valid_rows[rng.index(prepared.valid_rows.len())]];
        let Some(mut point)=constraints::background(row,donor,&groups,&reference.feature_names) else{continue;};
        let mut previous=engine.score(&point)?;baselines.push(previous);
        let mut order=(0..groups.len()).collect::<Vec<_>>();rng.shuffle(&mut order);
        let mut deltas=vec![0.;groups.len()];
        for &index in &order {constraints::swap(&mut point,row,&groups[index]);
            if !constraints::valid(&point,&reference.feature_names){return Err("A joint Shapley intervention violated recognized constraints".into());}
            let current=engine.score(&point)?;deltas[index]=current-previous;previous=current;
        }
        for i in 0..groups.len(){totals[i]+=deltas[i];estimates[i].push(deltas[i]);}
    }
    if baselines.is_empty(){return Err("No compatible training-reference background fits this observation".into());}
    let n=baselines.len() as f64;let baseline=baselines.iter().sum::<f64>()/n;
    for value in &mut totals{*value/=n;}
    let reconstructed=baseline+totals.iter().sum::<f64>();
    let mut out=envelope("Sampled grouped Shapley values","local_final_model_score");
    out["features"]=json!(groups.iter().zip(&totals).map(|(g,&v)|feature(g,row,v)).collect::<Vec<_>>());
    out["quality"]=json!({"monte_carlo_standard_error":estimates.iter().filter_map(|v|math::standard_error(v)).reduce(f64::max),
        "reconstruction_error":(output-reconstructed).abs(),"sampled_paths":baselines.len()});
    out["details"]=json!({"baseline_score":baseline,"model_score":output,"reconstructed_score":reconstructed,"groups":group_details(&groups,reference),
        "game":"Joint donor backgrounds; actual selected-model score; other groups aggregated when resolution exceeds budget"});
    out["warnings"]=json!(["Monte Carlo grouped Shapley estimates are not exact individual-feature TreeSHAP; grouping changes attribution resolution.",
        "Reference sampling and grouping do not establish causal effects."]);
    Ok(out)
}

pub(crate) fn lime(row:&[f64],reference:&Reference,prepared:&Prepared,engine:&mut Engine<'_>,rng:&mut Rng)->Result<Value,String>{
    let groups=constraints::coalesce(&prepared.groups,8);
    let background=prepared.valid_rows.iter().find_map(|&i|constraints::background(row,&reference.rows[i],&groups,&reference.feature_names))
        .ok_or("No compatible reference background is available for a local surrogate")?;
    let mut design=Vec::new();let mut scores=Vec::new();let mut weights=Vec::new();let mut attempts=0;
    while engine.remaining()>0 && attempts<engine.limit*8 {
        engine.check()?;attempts+=1;
        let mask=(0..groups.len()).map(|_|if design.is_empty()||rng.coin(){1.}else{0.}).collect::<Vec<_>>();
        let mut point=background.clone();for (group,&selected) in groups.iter().zip(&mask){if selected==1.{constraints::swap(&mut point,row,group);}}
        if !constraints::valid(&point,&reference.feature_names){continue;}
        let score=engine.score(&point)?;let distance=mask.iter().filter(|&&v|v==0.).count() as f64/groups.len() as f64;
        weights.push((-distance.powi(2)/0.5625).exp());design.push(mask);scores.push(score);
    }
    if design.len()<groups.len()+2{return Err("Too few feasible perturbations for the local surrogate".into());}
    let coef=math::ridge(&design,&scores,&weights)?;
    let fitted=design.iter().map(|x|coef[0]+x.iter().zip(&coef[1..]).map(|(x,c)|x*c).sum::<f64>()).collect::<Vec<_>>();
    let total=weights.iter().sum::<f64>();let mean=scores.iter().zip(&weights).map(|(y,w)|y*w).sum::<f64>()/total;
    let sse=scores.iter().zip(&fitted).zip(&weights).map(|((y,p),w)|w*(y-p).powi(2)).sum::<f64>();
    let variance=scores.iter().zip(&weights).map(|(y,w)|w*(y-mean).powi(2)).sum::<f64>();
    let mut out=envelope("Native grouped local surrogate (LIME-style)","local_final_model_score");
    out["features"]=json!(groups.iter().zip(&coef[1..]).map(|(g,&v)|feature(g,row,v)).collect::<Vec<_>>());
    out["quality"]=json!({"local_fidelity_r2":if variance>1e-14{Some(1.-sse/variance)}else{None},"weighted_mse":sse/total,"fidelity_partition":"Fitted local perturbations; not independent validation"});
    out["details"]=json!({"surrogate_intercept":coef[0],"surrogate_prediction":coef.iter().sum::<f64>(),"perturbation_rows":design.len(),"groups":group_details(&groups,reference),"ridge":1e-6});
    out["warnings"]=json!(["This deterministic grouped weighted-ridge implementation is not the Python LIME package and does not claim identical neighborhoods or rankings.",
        "Surrogate coefficients approximate the chosen model near one reference background; inspect measured fidelity."]);
    Ok(out)
}
