"""Measured group selection inside fitting data; final validation stays sealed."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import xgboost as xgb

from framework.training.live_study.protocol import ROOT,FEATURES,atomic,read,relative,digest,fingerprint,versions
from .schema import group
from .selection import remove_redundant,purged_folds


def metrics(y,p,classes):
    predicted=p.argmax(1);n=classes
    cm=np.bincount(n*y.astype(int)+predicted,minlength=n*n).reshape(n,n)
    den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*cm.diagonal(),den,out=np.zeros(n,float),where=den>0)
    benign=y==0;attack=~benign
    return dict(accuracy=float(np.mean(predicted==y)),macro_f1=float(f1.mean()),
        false_positive_rate=float(np.mean(predicted[benign]!=0)) if benign.any() else None,
        attack_recall=float(np.mean(predicted[attack]!=0)) if attack.any() else None,
        rows=len(y),supported_classes=int(np.count_nonzero(cm.sum(1))))


def train_fold(x,y,indices,train,val,classes,device,rounds):
    params=dict(objective='multi:softprob',num_class=classes,tree_method='hist',device=device,
        max_depth=5,min_child_weight=2,eta=.08,subsample=.9,colsample_bytree=.9,
        max_bin=128,nthread=2,seed=42,reg_lambda=2,eval_metric='mlogloss')
    model=xgb.train(params,xgb.DMatrix(x[train][:,indices],label=y[train],nthread=2),num_boost_round=rounds)
    if device=='cuda':
        import json
        actual=json.loads(model.save_config())['learner']['generic_param']['device']
        if not actual.startswith('cuda'):raise RuntimeError('No silent CPU substitution for explicit CUDA screening')
    probability=model.predict(xgb.DMatrix(x[val][:,indices],nthread=2))
    return model,metrics(y[val],probability,classes)


def run(output,device='cpu',rounds=100):
    output=Path(output);manifest=read(output/'data/manifest.json')
    if (output/'selection.json').exists():raise FileExistsError('Preserve existing selection; use a new revision')
    start_time=time.monotonic();all_x=np.load(output/'data/X_train.npy');all_y=np.load(output/'data/y_train.npy')
    roles=np.load(output/'data/roles.npy');fit=np.flatnonzero(roles==0)
    x,y=all_x[fit],all_y[fit];names=manifest['feature_names'];classes=len(manifest['classes'])
    start=np.load(output/'data/window_start_train.npy')[fit];end=np.load(output/'data/window_end_train.npy')[fit]
    baseline=[names.index(name) for name in FEATURES]
    folds=purged_folds(x[:,baseline],y,start,end)
    np.savez_compressed(output/'screening_folds.npz',**{f'{part}_{i}':fit[ids] for i,fold in enumerate(folds) for part,ids in zip(('fit','assess'),fold)})
    kept,removed=remove_redundant(x,names)
    if len(kept)<2:raise ValueError('Insufficient varying measured features')
    groups={name:[i for i in kept if group(names[i])==name] for name in dict.fromkeys(group(names[i]) for i in kept)}
    def evaluate(indices,label,retain_models=False):
        scores=[];models=[]
        for fold,(train,val) in enumerate(folds):
            model,score=train_fold(x,y,indices,train,val,classes,device,rounds)
            scores.append(score)
            if retain_models:models.append(model)
            print(f'{label} fold{fold+1}: accuracy={score["accuracy"]:.5f} macro_f1={score["macro_f1"]:.5f}',flush=True)
        return scores,models
    def mean(scores,key):return float(np.mean([s[key] for s in scores]))
    reference,_=evaluate(baseline,'seven-count baseline')
    scores,models=evaluate(kept,'broad candidates',True)
    importances={}
    for name,members in groups.items():
        deltas=[]
        for fold,(_,val) in enumerate(folds):
            for repeat in range(2):
                matrix=x[val][:,kept].copy();rng=np.random.default_rng(42+fold*13+repeat)
                order=rng.permutation(len(matrix));columns=[kept.index(i) for i in members]
                matrix[:,columns]=matrix[order][:,columns]
                score=metrics(y[val],models[fold].predict(xgb.DMatrix(matrix,nthread=2)),classes)
                deltas.append(scores[fold]['macro_f1']-score['macro_f1'])
        importances[name]=dict(features=[names[i] for i in members],mean_macro_f1_loss=float(np.mean(deltas)),
            permutation_macro_f1_losses=deltas)
    current=kept.copy();current_scores=scores;ablations=[]
    # Removal is conservative: no fixed top-k and no threshold that discards
    # highly correlated yet distinct packet measurements without refitting.
    candidates=sorted((g for g in groups if importances[g]['mean_macro_f1_loss']<=.002),
        key=lambda g:importances[g]['mean_macro_f1_loss'])[:3]
    for name in candidates:
        proposal=[i for i in current if i not in groups[name]]
        if len(proposal)<max(32,int(.8*len(kept))):continue
        proposed,_=evaluate(proposal,f'ablate {name}')
        accepted=(mean(proposed,'macro_f1')>=mean(current_scores,'macro_f1')-.002
            and mean(proposed,'accuracy')>=mean(current_scores,'accuracy')-.002
            and all(a['macro_f1']>=b['macro_f1']-.005 for a,b in zip(proposed,current_scores)))
        ablations.append(dict(group=name,accepted=accepted,scores=proposed))
        if accepted:
            for i in groups[name]:removed[names[i]]=dict(reason='purged_fit_cv_group_ablation_noninferior',group=name)
            current,current_scores=proposal,proposed
    selection=dict(status='completed',profile=manifest['profile'],candidate_features=len(names),
        nonredundant_features=len(kept),selected_features=[names[i] for i in current],selected_indices=current,
        selected_feature_count=len(current),feature_schema_fingerprint=fingerprint([names[i] for i in current]),
        model='fixed XGBoost screening model; not a full retuning of all families',device=device,rounds=rounds,
        objective='macro F1 over all84 declared classes; class coverage reported per fold',
        selection_data_role='three purged chronological folds entirely inside original fitting role',
        fitting_rows=len(fit),fold_sizes=[dict(fit=len(a),assess=len(b)) for a,b in folds],
        baseline_scores=reference,all_nonredundant_scores=scores,selected_scores=current_scores,
        mean_baseline_accuracy=mean(reference,'accuracy'),mean_baseline_macro_f1=mean(reference,'macro_f1'),
        mean_selected_accuracy=mean(current_scores,'accuracy'),mean_selected_macro_f1=mean(current_scores,'macro_f1'),
        dropped_features=removed,group_permutation_importance=importances,ablations=ablations,
        policy='Drop missing/constants/exact duplicates/fixed10s rescalings; preserve correlated distinct measurements. At most three weak group ablations; retain at least80 percent of nonredundant features.',
        elapsed_seconds=time.monotonic()-start_time,versions=versions(),seed=42,
        validation_loaded=False,test_loaded=False,live_accuracy_verified=False,publisher_equivalence_verified=False,
        data_manifest_sha256=digest(output/'data/manifest.json'),folds_sha256=digest(output/'screening_folds.npz'))
    atomic(output/'selection.json',selection)
    atomic(output/'selected_profile.json',dict(schema_version='npcap-network-v1',feature_names=selection['selected_features'],
        feature_schema_fingerprint=selection['feature_schema_fingerprint'],classes=manifest['classes'],window_seconds=10,
        missing_policy='null stays unknown; fitting-only imputation belongs to a newly trained model',
        stage='feature_selection_complete_new_models_required',native_model_promoted=False,publisher_equivalence_verified=False,
        selection=relative(output/'selection.json'),selection_sha256=digest(output/'selection.json')))
    lines=['# Broad packet feature selection','',f'Selected {len(current)} of {len(names)} candidates; {len(kept)} were nonredundant.',
        '',f'Within-fitting purged CV mean accuracy: seven counts {mean(reference,"accuracy"):.4%}; selected {mean(current_scores,"accuracy"):.4%}.',
        f'Within-fitting purged CV mean macro F1: seven counts {mean(reference,"macro_f1"):.4%}; selected {mean(current_scores,"macro_f1"):.4%}.',
        '', 'These are exploratory feature-screening estimates, not final original-validation or live-network scores.',
        'Selection never reads original validation or held-out test arrays. New model training/native export is still required.',
        '', '| Group | Selected fields |','|---|---:|']
    lines += [f'| {name} | {sum(i in current for i in members)} |' for name,members in groups.items()]
    lines += ['', 'Full ordered profile and per-feature drop reasons: selected_profile.json and selection.json.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'SELECTED {len(current)} FEATURES; no original validation or held-out test loaded',flush=True)
    return selection
