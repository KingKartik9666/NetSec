use crate::{constraints::{self,Prepared},envelope,feature,math::{self,Rng},Engine,Reference};
use serde_json::{json,Value};

pub(crate) fn anchors(row:&[f64],reference:&Reference,prepared:&Prepared,engine:&mut Engine<'_>,_rng:&mut Rng)->Result<Value,String>{
    let predicted=engine.evaluate(row)?.class_index;
    let candidates=prepared.groups.iter().filter(|g|g.indices.iter().any(|&i|row[i].is_finite())).take(4).collect::<Vec<_>>();
    let allowance=(engine.remaining()/candidates.len().max(1)).max(1);
    let mut best:Option<(usize,usize,f64,usize)>=None;
    for (index,group) in candidates.iter().enumerate(){
        let matched=prepared.valid_rows.iter().copied().filter(|&j|group.indices.iter().all(|&i|{
            let (a,b)=(row[i],reference.rows[j][i]);if a.is_nan()||b.is_nan(){return a.is_nan()&&b.is_nan();}
            (a-b).abs()<=prepared.scales[i]*0.15
        })).collect::<Vec<_>>();
        let mut successes=0;let mut tested=0;
        for &j in matched.iter().take(allowance){if engine.remaining()==0{break;}
            successes+=usize::from(engine.evaluate(&reference.rows[j])?.class_index==predicted);tested+=1;}
        let lower=math::wilson(successes,tested).0;
        if best.as_ref().is_none_or(|(_,_,old,_)|lower>*old){best=Some((index,tested,lower,successes));}
    }
    let mut out=envelope("Reference-supported conditional anchor search","local_model_decision");
    let (index,tested,lower,successes)=best.unwrap_or((0,0,0.,0));
    let condition=candidates.get(index);
    let coverage=condition.map(|g|prepared.valid_rows.iter().filter(|&&j|g.indices.iter().all(|&i|{
        let (a,b)=(row[i],reference.rows[j][i]);(a.is_nan()&&b.is_nan())||((a-b).abs()<=prepared.scales[i]*0.15)
    })).count() as f64/prepared.valid_rows.len().max(1) as f64).unwrap_or(0.);
    out["quality"]=json!({"precision":if tested>0{Some(successes as f64/tested as f64)}else{None},"precision_lower_bound":lower,
        "reference_coverage":coverage,"distinct_reference_support":tested,"confidence_interval":"Wilson 95% descriptive interval; dependent training observations may violate independence"});
    out["details"]=json!({"state":if lower>=0.95{"supported_anchor"}else{"insufficient_support"},"target_class_index":predicted,
        "condition_group":condition.map(|g|&g.name),"normalized_tolerance":0.15,"candidate_groups_tested":candidates.len(),"decision_change_verified":false});
    out["warnings"]=json!(["Conditional agreement on a small approved training reference is not a universal precision guarantee or causal rule.","A supported anchor requires its measured lower bound to reach 0.95; otherwise support remains insufficient."]);
    Ok(out)
}

pub(crate) fn counterfactual(row:&[f64],reference:&Reference,prepared:&Prepared,engine:&mut Engine<'_>,_rng:&mut Rng)->Result<Value,String>{
    let original=engine.evaluate(row)?;
    let mut candidates=prepared.valid_rows.iter().copied().map(|i|(i,constraints::distance(row,&reference.rows[i],prepared)))
        .filter(|(_,d)|d.is_finite()&&*d>0.).collect::<Vec<_>>();
    candidates.sort_by(|a,b|a.1.total_cmp(&b.1));
    let mut found=None;
    for (index,distance) in candidates {if engine.remaining()==0{break;}
        let prediction=engine.evaluate(&reference.rows[index])?;
        if prediction.class_index!=original.class_index {found=Some((index,distance,prediction));break;}
    }
    let mut out=envelope("Constrained observed-reference counterfactual search","local_model_decision");
    out["details"]=if let Some((index,distance,prediction))=found {
        let point=&reference.rows[index];
        out["features"]=json!(prepared.groups.iter().filter(|g|g.indices.iter().any(|&i|row[i].is_finite()&&row[i]!=point[i]))
            .map(|g|feature(g,row,prediction.score-original.score)).collect::<Vec<_>>());
        json!({"state":"observed_reference_counterfactual","counterfactual":point.iter().map(|v|v.is_finite().then_some(*v)).collect::<Vec<_>>(),
            "reference_index":index,"reference_sha256":reference.sha256,"distance":distance,"original_class_index":original.class_index,
            "candidate_class_index":prediction.class_index,"decision_change_verified":engine.semantics=="actual_model_decision",
            "missingness_preserved":true,"candidate_is_observed_reference":true})
    }else{json!({"state":"no_feasible_counterfactual","counterfactual":null,"decision_change_verified":false})};
    out["warnings"]=json!(["This search only tests observed, physically feasible reference rows with the same measured/missing mask. It does not prove a causal fix or actionable mitigation.",
        "Per-group displayed score differences describe the complete candidate, not independently attributed interventions."]);
    Ok(out)
}
