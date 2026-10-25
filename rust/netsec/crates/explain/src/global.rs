use crate::{constraints::{self,Prepared},envelope,feature,unsupported,math::Rng,Engine,Reference,Settings};
use serde_json::{json,Value};

pub(crate) fn permutation(row:&[f64],reference:&Reference,prepared:&Prepared,settings:&Settings,engine:&mut Engine<'_>,rng:&mut Rng)->Result<Value,String>{
    if reference.labels.len()!=reference.rows.len(){return Err("Grouped permutation requires aligned reference labels".into());}
    let groups=constraints::coalesce(&prepared.groups,8);
    let n=(engine.remaining()/(groups.len()+1)).min(prepared.valid_rows.len()).min(32);
    if n<2{return Ok(unsupported("Grouped permutation importance","global_training_reference","insufficient_support","At least two labeled reference observations must fit the evaluation budget"));}
    let indices=&prepared.valid_rows[..n];let mut baseline=Vec::new();
    for &index in indices{let score=engine.score(&reference.rows[index])?;let truth=usize::from(reference.labels[index]==settings.target_index) as f64;baseline.push((score-truth).powi(2));}
    let mut features=Vec::new();let mut groups_audit=Vec::new();
    for group in &groups {
        let mut order=(0..n).collect::<Vec<_>>();rng.shuffle(&mut order);
        if order.iter().enumerate().all(|(i,&j)|i==j){order.rotate_left(1);}
        let mut deltas=Vec::new();
        for (i,&index) in indices.iter().enumerate(){
            if engine.remaining()==0{break;}
            let donor=&reference.rows[indices[order[i]]];let mut point=reference.rows[index].clone();
            if !constraints::compatible(&point,donor,group){continue;}
            constraints::swap(&mut point,donor,group);if !constraints::valid(&point,&reference.feature_names){continue;}
            let score=engine.score(&point)?;let truth=usize::from(reference.labels[index]==settings.target_index) as f64;
            deltas.push((score-truth).powi(2)-baseline[i]);
        }
        let delta=if deltas.is_empty(){0.}else{deltas.iter().sum::<f64>()/deltas.len() as f64};
        if !deltas.is_empty(){features.push(feature(group,row,delta));}
        groups_audit.push(json!({"group":group.name,"feasible_pairs":deltas.len(),"importance":if deltas.is_empty(){None}else{Some(delta)}}));
    }
    let mut out=envelope("Grouped permutation importance","global_training_reference");out["features"]=json!(features);
    out["quality"]=json!({"reference_rows":n,"baseline_brier_loss":baseline.iter().sum::<f64>()/n as f64});
    out["details"]=json!({"estimand":"One-vs-rest target Brier-loss increase on approved training reference, after joint-group permutation","groups":groups_audit,"independent_assessment":false});
    out["warnings"]=json!(["This is global reference importance, not a local signed explanation of this individual decision.","A single bounded permutation has sampling variability; negative estimates are retained. Missingness-incompatible pairs are excluded."]);
    Ok(out)
}

pub(crate) fn ale(row:&[f64],reference:&Reference,prepared:&Prepared,engine:&mut Engine<'_>,_rng:&mut Rng)->Result<Value,String>{
    let groups=prepared.groups.iter().filter(|g|g.indices.len()==1&&g.kind=="numeric").take(4).collect::<Vec<_>>();
    if groups.is_empty(){return Ok(unsupported("First-order accumulated local effects","global_training_reference","unsupported","ALE requires an unlinked scalar numeric group; categorical, presence and linked-statistic groups are not independently varied"));}
    let mut features=Vec::new();let mut curves=Vec::new();
    for group in groups {
        let index=group.indices[0];let mut values=prepared.valid_rows.iter().map(|&i|reference.rows[i][index]).filter(|v|v.is_finite()).collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);values.dedup();if values.len()<2{continue;}
        let bins=values.len().saturating_sub(1).min(4);let mut accumulated=0.;let mut points=Vec::new();
        for bin in 0..bins {
            let low=values[bin*(values.len()-1)/bins];let high=values[(bin+1)*(values.len()-1)/bins];
            let mut differences=Vec::new();
            for &r in &prepared.valid_rows {
                if differences.len()>=3||engine.remaining()<2{break;}
                let value=reference.rows[r][index];if !value.is_finite()||value<low||value>high{continue;}
                let mut a=reference.rows[r].clone();let mut b=a.clone();a[index]=low;b[index]=high;
                if !constraints::valid(&a,&reference.feature_names)||!constraints::valid(&b,&reference.feature_names){continue;}
                differences.push(engine.score(&b)?-engine.score(&a)?);
            }
            if differences.is_empty(){continue;}
            let delta=differences.iter().sum::<f64>()/differences.len() as f64;accumulated+=delta;
            points.push(json!({"lower":low,"upper":high,"local_delta":delta,"accumulated":accumulated,"support":differences.len()}));
        }
        if points.is_empty(){continue;}
        let center=points.iter().map(|p|p["accumulated"].as_f64().unwrap_or(0.)).sum::<f64>()/points.len() as f64;
        for point in &mut points {point["centered_effect"]=json!(point["accumulated"].as_f64().unwrap_or(0.)-center);}
        let effect=points.iter().find(|p|row[index]<=p["upper"].as_f64().unwrap_or(f64::INFINITY)).or_else(||points.last()).map(|p|p["centered_effect"].as_f64().unwrap_or(0.)).unwrap_or(0.);
        features.push(feature(group,row,effect));curves.push(json!({"feature":group.name,"bins":points}));
    }
    let mut out=envelope("Bounded first-order accumulated local effects","global_training_reference");out["features"]=json!(features);
    out["details"]=json!({"state":if curves.is_empty(){"insufficient_support"}else{"completed"},"curves":curves,"centering":"Equal weight over supported reference-quantile bins"});
    out["warnings"]=json!(["ALE describes a model's accumulated local response on a small training reference, not a causal traffic effect.","Only independently valid scalar numeric groups are varied. Linked counts and presence groups remain unavailable for scalar ALE."]);
    Ok(out)
}
