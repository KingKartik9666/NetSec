"""Actual training, bounded early stopping and raw-schema CPU checkpoint replay."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

from framework.training.tabm_data import TabMPreprocessor
from framework.training.tabm_models import TabMClassifier, member_loss


def quick_metrics(y, p):
    n = p.shape[1]
    cm = np.bincount(n*np.asarray(y, dtype=int)+p.argmax(1), minlength=n*n).reshape(n, n)
    tp = cm.diagonal()
    den = cm.sum(0)+cm.sum(1)
    f1 = np.divide(2*tp, den, out=np.zeros(n, dtype=float), where=den != 0)
    return dict(accuracy=float(tp.sum()/cm.sum()), macro_f1=float(f1.mean()))


@torch.inference_mode()
def predict(model, preprocessor, X, device='cpu', batch_size=256):
    model.eval()
    result = []
    for start in range(0, len(X), batch_size):
        values = torch.from_numpy(preprocessor.transform(X[start:start+batch_size])).to(device)
        result.append(model.predict_proba(values).cpu().numpy())
    p = np.concatenate(result)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1., atol=2e-6):
        raise ValueError('Invalid model probabilities')
    return p


def load_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    prep = TabMPreprocessor.from_dict(saved['preprocessor'])
    model = TabMClassifier(saved['family'], prep, len(saved['classes']), saved['config'])
    model.load_state_dict(saved['state_dict']); model.to(device); model.eval()
    return model, prep, saved


def train(X, y, tuning_X, tuning_y, manifest, request, output, device='cpu', preprocessor=None):
    """Only training and tuning labels are accepted; no calibration/assessment access."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if manifest['benign_index'] != 0:
        raise ValueError('The pinned DataSense label schema must have benign index0')
    n_classes = len(manifest['classes'])
    if len(y) != len(X) or min(y) < 0 or max(y) >= n_classes:
        raise ValueError('Invalid training labels')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU substitute')
    torch.set_num_threads(2)
    seed = int(request['seed']); cfg = request['config']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
    prep = preprocessor or TabMPreprocessor(cfg['scaling'], cfg['n_bins']).fit(X, manifest['feature_names'])
    model = TabMClassifier(request['family'], prep, n_classes, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=3)
    counts = np.bincount(y, minlength=n_classes)
    weights = np.power(np.maximum(counts, 1).mean()/np.maximum(counts, 1), cfg['weight_power'])
    weights = np.clip(weights, .2, 10.)
    weights /= np.mean(weights[np.asarray(y, dtype=int)])
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    best_key, stale, checkpoint = (-1., -1.), 0, output/'model.pt'
    started = time.monotonic(); epochs = 0
    use_amp = device == 'cuda' and torch.cuda.is_bf16_supported()
    stop_reason = 'epoch_limit'
    for epoch in range(int(cfg['epochs'])):
        model.train(); total_loss, seen = 0., 0
        order = rng.permutation(len(y))
        for start in range(0, len(y), cfg['batch_size']):
            if time.monotonic()-started > cfg['runtime_seconds']:
                stop_reason = 'runtime_limit'; break
            ids = order[start:start+cfg['batch_size']]
            batch = torch.from_numpy(prep.transform(X[ids])).to(device)
            target = torch.tensor(np.asarray(y[ids]), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(batch)
                loss = member_loss(logits, target, model.hierarchical, weights_tensor,
                                   cfg['subtype_weight'], cfg['smoothing'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); seen += len(ids); total_loss += loss.item()*len(ids)
        if not seen:
            break
        p = predict(model, prep, tuning_X, device)
        score = quick_metrics(tuning_y, p); key = (score['accuracy'], score['macro_f1'])
        epochs = epoch+1
        telemetry = dict(epoch=epochs, rows_seen=seen, full_epoch=seen == len(y),
            training_loss=total_loss/seen, tuning=score, elapsed_seconds=time.monotonic()-started,
            learning_rate=optimizer.param_groups[0]['lr'])
        with (output/'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(telemetry)+'\n')
        print(json.dumps(telemetry), flush=True)
        if key > best_key:
            best_key, stale = key, 0
            state = dict(family=request['family'], config=cfg, classes=manifest['classes'],
                benign_index=0, feature_names=manifest['feature_names'], preprocessor=prep.to_dict(),
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                seed=seed, best_epoch=epochs, selection_metrics=score)
            temporary = checkpoint.with_suffix('.tmp'); torch.save(state, temporary); os.replace(temporary, checkpoint)
        else:
            stale += 1
        scheduler.step(score['accuracy'])
        if stale >= cfg['patience']:
            stop_reason = 'early_stopping'; break
        if stop_reason == 'runtime_limit':
            break
    if not checkpoint.exists():
        raise RuntimeError('Budget expired before a checkpoint could be evaluated')
    model, prep, saved = load_checkpoint(checkpoint, device)
    tuning_p = predict(model, prep, tuning_X, device)
    # CPU checkpoint replay on a deterministic sample; full final inference uses CPU below.
    sample = tuning_X[:min(256, len(tuning_X))]
    reference = predict(model, prep, sample, device)
    cpu, replay_prep, _ = load_checkpoint(checkpoint)
    replay = predict(cpu, replay_prep, sample)
    np.testing.assert_allclose(reference, replay, atol=5e-5, rtol=5e-4)
    if not np.array_equal(reference.argmax(1), replay.argmax(1)):
        raise AssertionError('CPU/GPU checkpoint decisions differ')
    details = dict(epochs=epochs, best_epoch=saved['best_epoch'], stop_reason=stop_reason,
        selection_metrics=quick_metrics(tuning_y, tuning_p), training_seconds=time.monotonic()-started,
        training_device=device, mixed_precision='bfloat16' if use_amp else 'float32',
        parameter_count=sum(v.numel() for v in model.parameters()), checkpoint_reload_verified=True,
        cpu_replay_rows=len(sample), gpu_peak_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
    return cpu, replay_prep, details
