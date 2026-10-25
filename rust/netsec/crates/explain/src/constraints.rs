//! Conservative donor grouping and candidate rejection, never silent clipping.
use crate::{check_cancel,FeatureGroup,Reference};
use std::collections::{BTreeMap,BTreeSet};
use std::sync::atomic::AtomicBool;

pub(crate) struct Prepared {
    pub(crate) groups:Vec<FeatureGroup>,
    pub(crate) valid_rows:Vec<usize>,
    pub(crate) warnings:Vec<String>,
    pub(crate) scales:Vec<f64>,
}

fn find(parents:&mut [usize],mut i:usize)->usize{
    while parents[i]!=i {parents[i]=parents[parents[i]];i=parents[i];}i
}
fn join(parents:&mut [usize],a:usize,b:usize){let a=find(parents,a);let b=find(parents,b);parents[b]=a;}
fn family(name:&str)->String{
    let name=name.split("__").next().unwrap_or(name);
    for suffix in ["_std_deviation","_variance","_count","_avg","_mean","_min","_max","_std"]{
        if let Some(base)=name.strip_suffix(suffix){return base.to_string();}
    }
    name.to_string()
}

pub(crate) fn valid(values:&[f64],names:&[String])->bool{
    if values.len()!=names.len()||values.iter().any(|v|v.is_infinite()){return false;}
    let mut stats:BTreeMap<String,(Option<f64>,Option<f64>,Option<f64>)>=BTreeMap::new();
    let packets=names.iter().position(|n|n=="network_packets_all_count").map(|i|values[i]).filter(|v|v.is_finite());
    for (name,&v) in names.iter().zip(values){
        if !v.is_finite(){continue;}
        if name.contains("__present_")||name.ends_with("__is_missing"){
            if v!=0.&&v!=1. {return false;}
        }
        if name.starts_with("network_")&&name.ends_with("_count"){
            if v<0.||v.fract()!=0. {return false;}
            if (name.starts_with("network_tcp-flags-")||name=="network_packets_src_count"||name=="network_packets_dst_count")
                &&packets.is_some_and(|p|v>p){return false;}
        }
        if name.ends_with("_std_deviation")&&v<0.{return false;}
        let base=family(name);let item=stats.entry(base).or_default();
        if name.ends_with("_min"){item.0=Some(v);}
        if name.ends_with("_max"){item.1=Some(v);}
        if name.ends_with("_avg")||name.ends_with("_mean"){item.2=Some(v);}
    }
    stats.values().all(|&(min,max,mean)|{
        !(min.zip(max).is_some_and(|(a,b)|a>b)
            ||min.zip(mean).is_some_and(|(a,b)|a>b)
            ||mean.zip(max).is_some_and(|(a,b)|a>b))
    })
}

pub(crate) fn prepare(row:&[f64],reference:&Reference,cancel:&AtomicBool)->Result<Prepared,String>{
    let n=reference.feature_names.len();
    if n==0||n>2048||reference.rows.is_empty()||reference.rows.len()>4096{
        return Err("Explanation requires 1..2048 features and 1..4096 approved reference rows".into());
    }
    if row.len()!=n||!valid(row,&reference.feature_names){return Err("Input violates feature dimensions or recognized physical/count constraints".into());}
    if reference.feature_names.iter().collect::<BTreeSet<_>>().len()!=n||reference.feature_names.iter().any(|v|v.is_empty()){
        return Err("Reference feature names must be unique and nonempty".into());
    }
    if !reference.labels.is_empty()&&reference.labels.len()!=reference.rows.len(){return Err("Reference labels are not row-aligned".into());}
    if reference.sha256.trim().is_empty(){return Err("Reference provenance digest is required".into());}
    let mut parents=(0..n).collect::<Vec<_>>();let mut claimed=vec![false;n];
    let mut group_names=BTreeSet::new();
    for group in &reference.groups{
        check_cancel(cancel)?;
        if group.indices.is_empty()||group.name.is_empty()||!group_names.insert(&group.name){return Err("Invalid or duplicate feature group".into());}
        for &i in &group.indices{
            if i>=n||claimed[i]{return Err("Feature groups contain out-of-range or overlapping indices".into());}
            claimed[i]=true;join(&mut parents,group.indices[0],i);
        }
    }
    // All network measurements and their protocol/port/presence indicators stay
    // together. Fine-grained, independent flag interventions are not defensible.
    let network=reference.feature_names.iter().enumerate().filter(|(_,name)|name.starts_with("network_")).map(|(i,_)|i).collect::<Vec<_>>();
    for &i in network.iter().skip(1){join(&mut parents,network[0],i);}
    let mut families:BTreeMap<String,usize>=BTreeMap::new();
    for (i,name) in reference.feature_names.iter().enumerate(){
        check_cancel(cancel)?;
        if let Some(&other)=families.get(&family(name)){join(&mut parents,other,i);}else{families.insert(family(name),i);}
    }
    let mut components:BTreeMap<usize,Vec<usize>>=BTreeMap::new();
    for i in 0..n{components.entry(find(&mut parents,i)).or_default().push(i);}
    let mut groups=Vec::new();let mut coarsened=false;
    for indices in components.into_values(){
        let original=reference.groups.iter().find(|g|g.indices.len()==indices.len()&&g.indices.iter().all(|i|indices.contains(i)));
        let (name,kind)=if let Some(g)=original{(g.name.clone(),g.kind.clone())}
            else if indices.iter().any(|&i|reference.feature_names[i].starts_with("network_")){
                coarsened=true;("Network measurements and indicators".into(),"linked_network".into())
            }else if indices.len()>1{coarsened=true;(format!("{} (linked measurements)",family(&reference.feature_names[indices[0]])),"linked".into())}
            else{(reference.feature_names[indices[0]].clone(),"unknown".into())};
        groups.push(FeatureGroup{name,indices,kind});
    }
    groups.sort_by_key(|g|g.indices[0]);
    let mut valid_rows=Vec::new();let mut low=vec![f64::INFINITY;n];let mut high=vec![f64::NEG_INFINITY;n];
    for (r,values) in reference.rows.iter().enumerate(){
        check_cancel(cancel)?;
        if values.len()!=n||values.iter().any(|v|v.is_infinite()){return Err("Invalid reference dimensions or infinite observations".into());}
        if !valid(values,&reference.feature_names){continue;}
        valid_rows.push(r);
        for i in 0..n{if values[i].is_finite(){low[i]=low[i].min(values[i]);high[i]=high[i].max(values[i]);}}
    }
    let scales=low.iter().zip(high).map(|(a,b)|if (b-a).is_finite()&&b-a>0.{b-a}else{1.}).collect();
    let mut warnings=vec!["Donor substitutions are reference-supported model probes, not observed new traffic or causal interventions.".into()];
    if coarsened{warnings.push("Linked network/count/presence and summary-statistic features were coarsened into joint donor groups.".into());}
    if valid_rows.len()!=reference.rows.len(){warnings.push(format!("{} reference rows violating recognized constraints were excluded; source observations were not modified.",reference.rows.len()-valid_rows.len()));}
    Ok(Prepared{groups,valid_rows,warnings,scales})
}

pub(crate) fn same_mask(a:&[f64],b:&[f64])->bool{a.iter().zip(b).all(|(x,y)|x.is_nan()==y.is_nan())}
pub(crate) fn compatible(a:&[f64],b:&[f64],group:&FeatureGroup)->bool{
    group.indices.iter().all(|&i|a[i].is_nan()==b[i].is_nan())
}
pub(crate) fn swap(target:&mut [f64],donor:&[f64],group:&FeatureGroup){for &i in &group.indices{target[i]=donor[i];}}
pub(crate) fn background(row:&[f64],donor:&[f64],groups:&[FeatureGroup],names:&[String])->Option<Vec<f64>>{
    let mut result=row.to_vec();for group in groups{if compatible(row,donor,group){swap(&mut result,donor,group);}}
    valid(&result,names).then_some(result)
}

pub(crate) fn coalesce(groups:&[FeatureGroup],maximum:usize)->Vec<FeatureGroup>{
    if groups.len()<=maximum{return groups.to_vec();}
    let keep=maximum.saturating_sub(1);let mut result=groups[..keep].to_vec();
    result.push(FeatureGroup{name:"Other grouped features (joint player)".into(),
        indices:groups[keep..].iter().flat_map(|g|g.indices.iter().copied()).collect(),kind:"aggregate".into()});result
}

pub(crate) fn distance(a:&[f64],b:&[f64],prepared:&Prepared)->f64{
    if !same_mask(a,b){return f64::INFINITY;}
    prepared.groups.iter().map(|g|{
        let finite=g.indices.iter().filter(|&&i|a[i].is_finite()).copied().collect::<Vec<_>>();
        if finite.is_empty(){0.}else{finite.iter().map(|&i|(a[i]-b[i]).abs()/prepared.scales[i]).sum::<f64>()/finite.len() as f64}
    }).sum::<f64>()/prepared.groups.len() as f64
}
