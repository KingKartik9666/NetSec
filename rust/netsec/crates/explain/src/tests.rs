use super::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

fn fixture() -> Reference {
    let mut rows=Vec::new(); let mut labels=Vec::new();
    for i in 0..32 { let x=(i%4) as f64/3.; let y=((i/4)%4) as f64/3.;
        rows.push(vec![x,y]); labels.push(usize::from(x>0.5)); }
    Reference { feature_names: vec!["x".into(),"y".into()], rows,labels,
        groups:vec![FeatureGroup{name:"x".into(),indices:vec![0],kind:"numeric".into()},
                    FeatureGroup{name:"y".into(),indices:vec![1],kind:"numeric".into()}],
        sha256:"synthetic-test-only".into() }
}

fn settings() -> Settings { Settings { units:"model score".into(),budget:64,..Settings::default() } }
fn value(v:&serde_json::Value,key:&str)->f64 { v[key].as_f64().unwrap() }

#[test]
fn grouped_shapley_reconstructs_linear_score_and_is_deterministic() {
    let reference=fixture(); let cancel=AtomicBool::new(false);
    let f=|x:&[f64]|Ok(2.*x[0]+3.*x[1]+1.);
    let a=explain(Method::Shap,&[1.,1.],&reference,&settings(),&f,&cancel).unwrap();
    let b=explain(Method::Shap,&[1.,1.],&reference,&settings(),&f,&cancel).unwrap();
    assert_eq!(a,b);
    assert!((value(&a["details"],"reconstructed_score")-6.).abs()<1e-10);
    assert_eq!(a["features"].as_array().unwrap().len(),2);
    assert!(a["quality"]["monte_carlo_standard_error"].is_number());
}

#[test]
fn shapley_keeps_interaction_in_the_actual_nonlinear_game() {
    let reference=fixture(); let cancel=AtomicBool::new(false);
    let f=|x:&[f64]|Ok(x[0]*x[1]);
    let a=explain(Method::Shap,&[1.,1.],&reference,&settings(),&f,&cancel).unwrap();
    assert!((value(&a["details"],"reconstructed_score")-1.).abs()<1e-10);
    assert!(a["features"].as_array().unwrap().iter().all(|v|value(v,"contribution")>0.));
}

#[test]
fn all_methods_honor_a_hard_actual_callback_budget() {
    let reference=fixture(); let cancel=AtomicBool::new(false);
    for method in [Method::Shap,Method::Lime,Method::Anchors,Method::Counterfactual,Method::Permutation,Method::Ale] {
        let calls=AtomicUsize::new(0); let cfg=Settings{budget:16,..settings()};
        let f=|x:&[f64]|{calls.fetch_add(1,Ordering::SeqCst);Ok(x[0])};
        let out=explain(method,&[1.,1.],&reference,&cfg,&f,&cancel).unwrap();
        assert!(calls.load(Ordering::SeqCst)<=16,"{method:?}");
        assert_eq!(out["quality"]["evaluations"].as_u64().unwrap() as usize,calls.load(Ordering::SeqCst));
    }
}

#[test]
fn cancellation_stops_before_and_during_scoring() {
    let reference=fixture(); let cancel=AtomicBool::new(true);let calls=AtomicUsize::new(0);
    let f=|_:&[f64]|{calls.fetch_add(1,Ordering::SeqCst);Ok(0.5)};
    assert!(explain(Method::Shap,&[1.,1.],&reference,&settings(),&f,&cancel).unwrap_err().contains("cancel"));
    assert_eq!(calls.load(Ordering::SeqCst),0);
    cancel.store(false,Ordering::SeqCst);
    let f=|_:&[f64]|{cancel.store(true,Ordering::SeqCst);Ok(0.5)};
    assert!(explain(Method::Lime,&[1.,1.],&reference,&settings(),&f,&cancel).unwrap_err().contains("cancel"));
}

#[test]
fn every_local_perturbation_preserves_the_observed_missingness_mask() {
    let mut reference=fixture();reference.feature_names.push("unmeasured".into());
    for row in &mut reference.rows { row.push(f64::NAN); }
    let cancel=AtomicBool::new(false); let f=|x:&[f64]|{assert!(x[2].is_nan());Ok(x[0])};
    for m in [Method::Shap,Method::Lime,Method::Anchors,Method::Counterfactual] {
        let result=explain(m,&[1.,1.,f64::NAN],&reference,&settings(),&f,&cancel);
        assert!(result.is_ok(),"{m:?}: {result:?}");
    }
}

#[test]
fn related_packet_counts_are_swapped_jointly_and_invalid_rows_rejected() {
    let mut r=fixture();r.feature_names=vec!["network_packets_all_count".into(),"network_tcp-flags-syn_count".into()];
    r.rows=vec![vec![2.,1.],vec![8.,7.],vec![5.,4.]];r.labels=vec![0,1,1];
    let cancel=AtomicBool::new(false);let f=|x:&[f64]|{assert!(x[1]<=x[0]);Ok(x[1]/x[0])};
    let out=explain(Method::Shap,&[2.,1.],&r,&settings(),&f,&cancel).unwrap();
    assert_eq!(out["features"].as_array().unwrap().len(),1);
    assert!(explain(Method::Lime,&[2.,3.],&r,&settings(),&f,&cancel).is_err());
}

#[test]
fn constant_anchor_support_is_insufficient_to_certify_precision() {
    let mut r=fixture();r.rows.truncate(2);r.labels.truncate(2);
    let out=explain(Method::Anchors,&[1.,1.],&r,&settings(),&|_|Ok(0.9),&AtomicBool::new(false)).unwrap();
    assert_eq!(out["details"]["state"],"insufficient_support");
    assert!(out["quality"]["precision_lower_bound"].as_f64().unwrap()<0.95);
}

#[test]
fn counterfactual_never_fabricates_a_success() {
    let out=explain(Method::Counterfactual,&[1.,1.],&fixture(),&settings(),&|_|Ok(0.8),&AtomicBool::new(false)).unwrap();
    assert_eq!(out["details"]["state"],"no_feasible_counterfactual");
    assert!(out["details"]["counterfactual"].is_null());
}

#[test]
fn lime_reports_measured_local_surrogate_fidelity() {
    let out=explain(Method::Lime,&[1.,1.],&fixture(),&settings(),&|x|Ok(x[0]+2.*x[1]),&AtomicBool::new(false)).unwrap();
    assert!(out["quality"]["local_fidelity_r2"].as_f64().unwrap()>0.5);
    assert!(out["details"]["surrogate_prediction"].is_number());
}

#[test]
fn permutation_is_explicitly_global_labeled_target_brier_loss() {
    let out=explain(Method::Permutation,&[1.,1.],&fixture(),&Settings{target_index:1,..settings()},&|x|Ok(x[0]),&AtomicBool::new(false)).unwrap();
    assert_eq!(out["scope"],"global_training_reference");
    assert!(out["details"]["estimand"].as_str().unwrap().contains("Brier"));
    assert!(value(&out["features"][0],"contribution")>0.);
}

#[test]
fn ale_constant_model_is_flat_and_unsupported_groups_are_typed() {
    let out=explain(Method::Ale,&[1.,1.],&fixture(),&settings(),&|_|Ok(0.2),&AtomicBool::new(false)).unwrap();
    assert!(out["features"].as_array().unwrap().iter().all(|v|value(v,"contribution").abs()<1e-12));
    let mut r=fixture();for g in &mut r.groups {g.kind="categorical".into();}
    let out=explain(Method::Ale,&[1.,1.],&r,&settings(),&|_|Ok(0.2),&AtomicBool::new(false)).unwrap();
    assert_eq!(out["details"]["state"],"unsupported");
}

#[test]
fn bad_references_and_nonfinite_scores_are_errors() {
    let mut r=fixture();r.groups[1].indices=vec![0];
    assert!(explain(Method::Lime,&[1.,1.],&r,&settings(),&|_|Ok(0.),&AtomicBool::new(false)).is_err());
    assert!(explain(Method::Shap,&[1.,1.],&fixture(),&settings(),&|_|Ok(f64::NAN),&AtomicBool::new(false)).is_err());
}

#[test]
fn counterfactual_uses_multiclass_decisions_even_below_half_probability() {
    let predict=|x:&[f64]|Ok(Evaluation{score:0.2,class_index:if x[0]>0.5{7}else{9}});
    let out=explain_prediction(Method::Counterfactual,&[1.,1.],&fixture(),&settings(),&predict,&AtomicBool::new(false)).unwrap();
    assert_eq!(out["details"]["original_class_index"],7);
    assert_eq!(out["details"]["candidate_class_index"],9);
    assert_eq!(out["details"]["decision_change_verified"],true);
    assert_eq!(out["details"]["decision_semantics"],"actual_model_decision");
}

#[test]
fn anchor_reports_actual_multiclass_target_instead_of_probability_threshold() {
    let predict=|x:&[f64]|Ok(Evaluation{score:0.2,class_index:if x[0]>0.5{7}else{9}});
    let out=explain_prediction(Method::Anchors,&[1.,1.],&fixture(),&settings(),&predict,&AtomicBool::new(false)).unwrap();
    assert_eq!(out["details"]["target_class_index"],7);
    assert_eq!(out["quality"]["precision"],1.0);
}

#[test]
fn infeasible_permutations_are_unavailable_instead_of_zero_importance() {
    let mut reference=fixture();
    reference.rows=vec![vec![1.,f64::NAN],vec![f64::NAN,1.]];reference.labels=vec![0,1];
    let out=explain_prediction(Method::Permutation,&[1.,f64::NAN],&reference,&settings(),&|_|Ok(Evaluation{score:0.2,class_index:0}),&AtomicBool::new(false)).unwrap();
    assert!(out["features"].as_array().unwrap().is_empty());
    assert!(out["details"]["groups"].as_array().unwrap().iter().all(|g|g["importance"].is_null()));
}
