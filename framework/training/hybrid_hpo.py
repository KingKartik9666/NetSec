"""Persistent TPE policy and a training-only hybrid compatibility objective."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from framework.training.hybrid_blend import fit_blend, blend, quick_metrics, log_loss

FAMILIES = ('tabm_ne','xgboost','kan','tabpack')


def sample_recipe(trial, family, base):
    cfg = deepcopy(base['config']); arm = base['arm']
    if family == 'tabm_ne':
        cfg.update(width=trial.suggest_categorical('width',[128,256,384,512]),
            depth=trial.suggest_int('depth',1,4), k=16,
            d_embedding=trial.suggest_categorical('d_embedding',[8,12,16,24]),
            n_bins=trial.suggest_categorical('n_bins',[8,16,32,48,64]),
            scaling=trial.suggest_categorical('scaling',['log_robust','quantile']),
            lr=trial.suggest_float('lr',.0003,.003,log=True),
            weight_decay=trial.suggest_float('weight_decay',1e-6,.01,log=True),
            dropout=trial.suggest_float('dropout',0.,.35),
            smoothing=trial.suggest_float('smoothing',0.,.05), epochs=60,patience=8,
            runtime_seconds=2400)
    elif family == 'xgboost':
        cfg.update(max_depth=trial.suggest_int('max_depth',3,9),
            n_estimators=trial.suggest_int('n_estimators',400,1200,step=100),
            lr=trial.suggest_float('lr',.02,.15,log=True),
            min_child_weight=trial.suggest_float('min_child_weight',1.,16.,log=True),
            reg_lambda=trial.suggest_float('reg_lambda',.01,20.,log=True),
            reg_alpha=trial.suggest_float('reg_alpha',1e-8,2.,log=True),
            subsample=trial.suggest_float('subsample',.6,1.),
            colsample_bytree=trial.suggest_float('colsample_bytree',.6,1.),
            max_bin=trial.suggest_categorical('max_bin',[128,256]),
            runtime_seconds=2400,loss='native_ce',weight_strategy='none')
    elif family == 'kan':
        hidden=trial.suggest_categorical('hidden',['64,32','128,64','128,64,32','256,64'])
        cfg.update(hidden=[int(x) for x in hidden.split(',')],
            grid_size=trial.suggest_categorical('grid_size',[3,5,7,9]),
            lr=trial.suggest_float('lr',.0002,.003,log=True),
            weight_decay=trial.suggest_float('weight_decay',1e-6,.01,log=True),
            spline_regularization=trial.suggest_float('spline_regularization',1e-7,.001,log=True),
            scaling=trial.suggest_categorical('scaling',['log_minmax','quantile']),
            weight_strategy=trial.suggest_categorical('weight_strategy',['none','sqrt_balanced']),
            batch_size=trial.suggest_categorical('batch_size',[256,512,1024]),
            epochs=50,patience=8,runtime_seconds=2400)
    elif family == 'tabpack':
        capacity=trial.suggest_categorical('capacity',[.5,.75,1.,1.25])
        extra=trial.suggest_categorical('extra_depth',[0,1])
        dropout=trial.suggest_categorical('dropout_shift',[-.05,0.,.1])
        lr=trial.suggest_float('lr_scale',.4,2.5,log=True)
        decay=trial.suggest_float('decay_scale',.2,5.,log=True)
        cfg.update(d_block=[max(64,int(round(w*capacity/16))*16) for w in cfg['d_block']],
            n_blocks=[min(4,n+extra) for n in cfg['n_blocks']],
            dropout=[max(0.,min(.4,p+dropout)) for p in cfg['dropout']],
            lr=[v*lr for v in cfg['lr']],weight_decay=[v*decay for v in cfg['weight_decay']],
            scaling=trial.suggest_categorical('scaling',['quantile','log_robust']),
            batch_size=trial.suggest_categorical('batch_size',[128,256,512]),
            epochs=60,patience=8,runtime_seconds=2400)
    else:
        raise ValueError('Unknown hybrid branch family')
    if family != 'tabpack':
        arm=trial.suggest_categorical('arm',['full','strict'])
    return dict(family=family,config=cfg,arm=arm,seed=42)


def baseline_parameters(family, base):
    c=base['config']
    if family=='tabm_ne':
        names=('width','depth','d_embedding','n_bins','scaling','lr','weight_decay','dropout','smoothing')
        values={name:c[name] for name in names}
    elif family=='xgboost':
        values={name:c[name] for name in ('max_depth','n_estimators','lr','max_bin')}
        values.update(min_child_weight=1.,reg_lambda=1.,reg_alpha=1e-8,subsample=.8,colsample_bytree=1.)
    elif family=='kan':
        values={name:c[name] for name in ('grid_size','lr','weight_decay','scaling','weight_strategy','batch_size')}
        values.update(hidden=','.join(map(str,c['hidden'])),spline_regularization=1e-7)
    else:
        values=dict(capacity=1.,extra_depth=0,dropout_shift=0.,lr_scale=1.,decay_scale=1.,
                    scaling=c['scaling'],batch_size=c['batch_size'])
    if family!='tabpack':values['arm']=base['arm']
    return values


def trial_study(folder, family):
    import optuna
    if family not in FAMILIES and not family.startswith('combiner_'):
        raise ValueError('Invalid TPE study name')
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # Recreate a deterministic sampler for each persisted ask index. Its next
    # proposal depends on the saved observations and this recorded index seed.
    storage='sqlite:///'+(folder/'optuna.sqlite3').as_posix()
    existing=optuna.study.get_all_study_summaries(storage)
    count=next((s.n_trials for s in existing if s.study_name==family),0)
    index=FAMILIES.index(family) if family in FAMILIES else 9
    sampler=optuna.samplers.TPESampler(seed=42000+1000*index+count,
        n_startup_trials=4,multivariate=True,group=True)
    return optuna.create_study(study_name=family,storage=storage,load_if_exists=True,
        direction='maximize',sampler=sampler)


def fingerprint(request):
    text=json.dumps({k:request[k] for k in ('family','arm','config','seed')},sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


def ask_recipe(folder, family, base, anchors=None):
    import optuna
    study=trial_study(folder,family)
    for frozen in study.trials:
        if frozen.state==optuna.trial.TrialState.RUNNING:
            if 'request' in frozen.user_attrs:
                request=frozen.user_attrs['request']
                if anchors is not None and request.get('anchors')!=anchors:
                    raise ValueError('Frozen TPE partners changed on resume')
                return request
            study.tell(frozen.number,state=optuna.trial.TrialState.FAIL)
    if not study.trials:
        study.enqueue_trial(baseline_parameters(family,base),user_attrs={'baseline':True})
    trial=study.ask();request=sample_recipe(trial,family,base)
    adopted=bool(trial.user_attrs.get('baseline'))
    if adopted:
        # Exact historical recipe, including implicit defaults, is retained.
        request=dict(family=family,**deepcopy(base))
    request.update(job=f'{family}_trial{trial.number:03d}',phase='search',
        trial_number=trial.number,baseline=adopted,anchors=anchors or {})
    key=fingerprint(request)
    if any(t.user_attrs.get('fingerprint')==key for t in study.trials if t.number!=trial.number):
        trial.set_user_attr('duplicate_recipe',True)
        study.tell(trial,state=optuna.trial.TrialState.PRUNED)
        return None
    trial.set_user_attr('fingerprint',key);trial.set_user_attr('request',request)
    return request


def tell_result(folder, family, number, value=None, state='complete'):
    import optuna
    study=trial_study(folder,family)
    frozen=next(t for t in study.trials if t.number==number)
    if frozen.state.is_finished():return
    if state=='complete':
        if value is None or not np.isfinite(value):raise ValueError('A completed trial needs a measured score')
        study.tell(number,float(value))
    else:
        study.tell(number,state={'fail':optuna.trial.TrialState.FAIL,'pruned':optuna.trial.TrialState.PRUNED}[state])


def affected_hybrids(family):
    combinations={'tabm_xgboost':['tabm_ne','xgboost'],
        'tabm_xgboost_kan':['tabm_ne','xgboost','kan'],
        'tabpack_xgboost':['tabpack','xgboost']}
    return {name:members for name,members in combinations.items() if family in members}


def score_hybrids(probabilities, labels, roles, combinations):
    y=np.asarray(labels);roles=np.asarray(roles)
    fit,score=roles==0,roles==1
    if not fit.any() or not score.any():raise ValueError('Distinct nonempty meta fitting/scoring roles required')
    results={}
    for name,names in combinations.items():
        p=np.stack([probabilities[n] for n in names])
        cfg=fit_blend(p[:,fit],y[fit],names)
        q=blend(p[:,score],cfg,names)
        metrics=quick_metrics(y[score],q);metrics['log_loss']=log_loss(q,y[score])
        results[name]=dict(metrics=metrics,combiner=cfg)
    accuracy=float(np.mean([r['metrics']['accuracy'] for r in results.values()]))
    f1=float(np.mean([r['metrics']['macro_f1'] for r in results.values()]))
    return dict(objective=accuracy+1e-6*f1,mean_accuracy=accuracy,mean_macro_f1=f1,
        hybrids=results,meta_fit_rows=int(fit.sum()),meta_score_rows=int(score.sum()),
        objective_partition='original_train',checkpoint_selected_on_meta_rows=False)
