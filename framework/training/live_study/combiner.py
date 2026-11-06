"""Convex global blending and supported-class shrinkage on the blend partition."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from framework.training.hybrid_blend import fit_blend, validate_probabilities


def fit(probabilities, labels, names, penalty=None):
    p=validate_probabilities(probabilities);y=np.asarray(labels,dtype=int)
    result=fit_blend(p,y,names)
    result['objective']='independent holdout blending log loss; not OOF'
    if penalty is None:return result
    if not np.isfinite(penalty) or penalty<=0:raise ValueError('Positive class shrinkage required')
    m,n,c=p.shape;global_w=np.asarray(result['weights'])
    # A few examples cannot establish a separate class-specific weighting rule.
    active=np.flatnonzero(np.bincount(y,minlength=c)>=10)
    if not len(active):
        result['classwise_unavailable']='No class has ten blending observations'
        return result
    def unpack(flat):
        weights=np.broadcast_to(global_w[:,None],(m,c)).copy()
        weights[:,active]=softmax(flat.reshape(m,len(active)),axis=0)
        return weights
    def objective(flat):
        weights=unpack(flat);q=np.einsum('mnc,mc->nc',p,weights)
        total=q.sum(1);correct=np.maximum(q[np.arange(n),y],1e-12)
        value=np.mean(np.log(total)-np.log(correct))
        derivative=np.broadcast_to((1/total)[:,None],(n,c)).copy()
        derivative[np.arange(n),y]-=1/correct
        gradient=np.einsum('nc,mnc->mc',derivative,p)[:,active]/n
        w=weights[:,active];delta=w-global_w[:,None]
        value+=penalty*np.square(delta).sum()/len(active)
        gradient+=2*penalty*delta/len(active)
        return value,(w*(gradient-(gradient*w).sum(0))).ravel()
    initial=np.broadcast_to(np.log(np.maximum(global_w,.01))[:,None],(m,len(active))).copy()
    fitted=minimize(objective,initial.ravel(),jac=True,method='L-BFGS-B',
        options={'maxiter':1000,'ftol':1e-9,'gtol':1e-6})
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError(f'Class blend did not converge: {fitted.message}')
    result.update(kind='classwise',weights=unpack(fitted.x).tolist(),global_weights=global_w.tolist(),
        penalty=float(penalty),iterations=int(fitted.nit),classwise_supported_indices=active.tolist(),
        minimum_blending_class_support=10,unsupported_classes_use_global_weights=True)
    return result
