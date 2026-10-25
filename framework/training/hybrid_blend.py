"""Small nonnegative probability combiners fitted on training-only OOF data."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


def quick_metrics(y, p):
    """Small CPU-only metric helper; the orchestration process needs no torch."""
    from framework.training.trees import _macro_f1
    y = np.asarray(y, dtype=int); pred = p.argmax(1)
    return dict(accuracy=float(np.mean(y == pred)), macro_f1=_macro_f1(y, pred, p.shape[1]))


def validate_probabilities(probabilities):
    p = np.asarray(probabilities)
    if p.ndim != 3 or min(p.shape) < 1 or p.shape[2] < 2:
        raise ValueError('Expected [models, observations, classes] probabilities')
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1+2e-6):
        raise ValueError('Probabilities must be finite and nonnegative')
    if not np.allclose(p.sum(2), 1., rtol=0, atol=5e-6):
        raise ValueError('Probabilities must sum to one')
    return p


def blend(probabilities, config, names):
    p = validate_probabilities(probabilities)
    if list(names) != config['names'] or len(names) != len(p) or len(set(names)) != len(names):
        raise ValueError('Branch names/order differ from the saved combiner')
    if p.shape[2] != config['n_classes']:
        raise ValueError('Class dimensions differ')
    w = np.asarray(config['weights'], dtype=float)
    expected = (len(p),) if config['kind'] == 'global' else (len(p), p.shape[2])
    if w.shape != expected or not np.isfinite(w).all() or np.any(w < 0) or not np.allclose(w.sum(0), 1):
        raise ValueError('Invalid combiner weights')
    q = np.einsum('mnc,m->nc', p, w) if w.ndim == 1 else np.einsum('mnc,mc->nc', p, w)
    sums = q.sum(1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError('Combiner produced zero probability mass')
    return (q/sums).astype(np.float32)


def fit_blend(probabilities, labels, names, penalty=None):
    p = validate_probabilities(probabilities)
    y = np.asarray(labels)
    m, n, c = p.shape
    if len(names) != m or len(set(names)) != m:
        raise ValueError('One unique branch name required per probability array')
    if y.shape != (n,) or not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= c)):
        raise ValueError('Labels must be aligned integer class indices')
    true_p = np.asarray(p[:, np.arange(n), y], dtype=float)

    def global_objective(w):
        q = np.maximum(w @ true_p, 1e-12)
        return -np.log(q).mean(), -(true_p/q).mean(1)

    result = minimize(global_objective, np.full(m, 1/m), jac=True, method='SLSQP',
        bounds=[(0, 1)]*m, constraints={'type':'eq', 'fun':lambda w: w.sum()-1,
                                      'jac':lambda w: np.ones(m)},
        options={'ftol':1e-10, 'maxiter':200})
    if not result.success:
        raise RuntimeError(f'Blend optimization failed: {result.message}')
    global_w = np.maximum(result.x, 0); global_w /= global_w.sum()
    config = dict(kind='global', names=list(names), n_classes=c, weights=global_w.tolist(),
                  fitting_rows=n, objective='OOF log loss', optimizer_success=True)
    if penalty is None:
        return config
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Class blend requires positive shrinkage')

    # Normalize across classes after class-specific mixing. The gradient includes
    # that normalization; treating q as already normalized would fit the wrong loss.
    def objective(flat):
        w = softmax(flat.reshape(m, c), axis=0)
        q = np.einsum('mnc,mc->nc', p, w)
        z = q.sum(1)
        correct = np.maximum(q[np.arange(n), y], 1e-12)
        value = np.mean(np.log(z)-np.log(correct))
        derivative = np.broadcast_to((1/z)[:, None], (n, c)).copy()
        derivative[np.arange(n), y] -= 1/correct
        grad = np.einsum('nc,mnc->mc', derivative, p)/n
        delta = w-global_w[:, None]
        value += penalty*np.square(delta).sum()/c
        grad += (2*penalty/c)*delta
        return value, (w*(grad-(grad*w).sum(0))).ravel()

    initial = np.broadcast_to(np.log(np.maximum(global_w, .01))[:, None], (m, c)).copy()
    result = minimize(objective, initial.ravel(), jac=True, method='L-BFGS-B',
        options={'maxiter':150, 'ftol':1e-9, 'gtol':1e-6})
    if not np.isfinite(result.fun) or not result.success:
        raise RuntimeError(f'Class blend optimization failed: {result.message}')
    config.update(kind='classwise', weights=softmax(result.x.reshape(m, c), axis=0).tolist(),
                  global_weights=global_w.tolist(), penalty=penalty, iterations=int(result.nit))
    return config


def log_loss(p, y):
    return float(-np.log(np.maximum(np.asarray(p)[np.arange(len(y)), y], 1e-12)).mean())


def greedy_members(probabilities, labels, steps=8):
    """Select a small probability ensemble on early-stopping labels only."""
    p = validate_probabilities(probabilities)
    y = np.asarray(labels, dtype=int)
    true = p[:, np.arange(len(y)), y]
    counts = np.zeros(len(p), dtype=int)
    total = np.zeros(len(y), dtype=float)
    best, best_counts = float('inf'), None
    for i in range(steps):
        losses = -np.log(np.maximum((total[None]+true)/(i+1), 1e-12)).mean(1)
        index = int(losses.argmin())
        counts[index] += 1; total += true[index]
        if losses[index] < best:
            best, best_counts = float(losses[index]), counts.copy()
    return (best_counts/best_counts.sum()).tolist()
