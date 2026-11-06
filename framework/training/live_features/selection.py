"""Generous feature selection: redundancy checks and purged group ablation."""
from __future__ import annotations

import numpy as np
from framework.training.live_study.data import split_roles


def remove_redundant(x,names):
    retained=[];removed={}
    for i,name in enumerate(names):
        col=x[:,i];finite=col[np.isfinite(col)]
        if not len(finite):removed[name]=dict(reason='all_missing_on_fitting_rows');continue
        if len(finite)==len(col) and np.all(finite==finite[0]):
            removed[name]=dict(reason='constant_on_fitting_rows',value=float(finite[0]));continue
        duplicate=next((j for j in retained if np.array_equal(col,x[:,j],equal_nan=True)),None)
        if duplicate is not None:
            removed[name]=dict(reason='exact_training_duplicate',representative=names[duplicate]);continue
        # At a fixed ten seconds these rates convey exactly the counter value
        # in another unit. Ratios and correlated distinct measurements remain.
        if name.endswith('_count__per_second') and name.removesuffix('__per_second') in [names[j] for j in retained]:
            removed[name]=dict(reason='fixed_window_deterministic_rescaling',representative=name.removesuffix('__per_second'));continue
        retained.append(i)
    return retained,removed


def purged_folds(x,y,start,end):
    roles=split_roles(x,y,start,end)
    folds=[]
    for role in (1,2,3):
        train=np.flatnonzero((roles>=0)&(roles<role));val=np.flatnonzero(roles==role)
        if not len(train) or not len(val):raise ValueError('Insufficient distinct chronological data for feature screening')
        folds.append((train,val))
    return folds
