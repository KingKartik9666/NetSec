"""Bounded neural branch fitting and explicit numerical replay evidence."""
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
from framework.training.tabm_fit import quick_metrics
from framework.training.hybrid_blend import greedy_members


def parity_diagnostics(reference, replay):
    a, b = np.asarray(reference), np.asarray(replay)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid replay arrays')
    changed = a.argmax(1) != b.argmax(1)
    sorted_p = np.sort(a, axis=1)
    return dict(rows=len(a), max_absolute_difference=float(np.max(np.abs(a-b))),
        mean_absolute_difference=float(np.mean(np.abs(a-b))),
        changed_decisions=int(changed.sum()),
        changed_reference_margins=(sorted_p[:, -1]-sorted_p[:, -2])[changed].tolist())


def build(family, prep, n_classes, cfg, device):
    if family == 'tabpack':
        from framework.training.tabpack_backend import make_pack
        return make_pack(2*len(prep.feature_names), n_classes, cfg, device)
    if family != 'tabm_ne':
        raise ValueError('Hybrid neural branch must be flat TabM NE or TabPack')
    model = TabMClassifier(family, prep, n_classes, cfg).to(device)
    return model, torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])


@torch.inference_mode()
def member_predictions(model, prep, X, family, device='cpu', batch_size=256):
    model.eval(); chunks = []
    for start in range(0, len(X), batch_size):
        x = torch.from_numpy(prep.transform(X[start:start+batch_size])).to(device)
        logits = model(x, None).transpose(0, 1) if family == 'tabpack' else model(x)
        chunks.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(chunks).transpose(1, 0, 2)


@torch.inference_mode()
def predict(model, prep, X, family, weights, device='cpu', batch_size=256):
    # Stream batches, avoiding an N x members x 84 allocation for full OOF data.
    chunks = []
    for start in range(0, len(X), batch_size):
        p = member_predictions(model, prep, X[start:start+batch_size], family, device, batch_size)
        chunks.append(np.einsum('mnc,m->nc', p, weights).astype(np.float32))
    result = np.concatenate(chunks)
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1., atol=3e-6):
        raise ValueError('Invalid neural probabilities')
    return result


def load(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    prep = TabMPreprocessor.from_dict(saved['preprocessor'])
    if saved['feature_names'] != prep.feature_names or len(set(saved['classes'])) != len(saved['classes']):
        raise ValueError('Neural checkpoint/preprocessor schema differs')
    model, optimizer = build(saved['family'], prep, len(saved['classes']), saved['config'], device)
    del optimizer
    model.load_state_dict(saved['state_dict']); model.eval()
    weights = saved.get('member_weights', [1/saved['config']['k']]*saved['config']['k']) if saved['family'] != 'tabpack' else saved['member_weights']
    return model, prep, saved, weights


def train(family, X, y, early_X, early_y, manifest, config, seed, output, device='cpu'):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output/'model.pt'
    if checkpoint.exists():
        raise FileExistsError('Use a new attempt; preserve the existing checkpoint')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(2); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed); torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    prep = TabMPreprocessor(config['scaling'], config['n_bins']).fit(X, manifest['feature_names'])
    model, optimizer = build(family, prep, len(manifest['classes']), config, device)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=3)
                 if family == 'tabm_ne' else None)
    count = config['pack_size'] if family == 'tabpack' else config['k']
    weights = [1/count]*count
    rng = np.random.default_rng(seed); best = (-1., -1.); stale = 0
    stop_reason = 'epoch_limit'; epochs = 0
    for epoch in range(config['epochs']):
        model.train(); seen = 0; loss_sum = 0.
        for ids in np.array_split(rng.permutation(len(y)), max(1, int(np.ceil(len(y)/config['batch_size'])))):
            if time.monotonic()-started > config['runtime_seconds'] and checkpoint.exists():
                stop_reason = 'runtime_limit'; break
            x = torch.from_numpy(prep.transform(X[ids])).to(device)
            target = torch.as_tensor(np.asarray(y[ids]), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if family == 'tabpack':
                logits = model(x, None)
                # Sum member mean losses: each packed model retains the gradient
                # scale of independent training with its own optimizer settings.
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.repeat(count))*count
            else:
                use_amp = device == 'cuda' and torch.cuda.is_bf16_supported()
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(x)
                    loss = member_loss(logits, target, False, None, 1., config.get('smoothing', 0.))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite neural branch loss')
            loss.backward()
            if family != 'tabpack':
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); seen += len(ids); loss_sum += loss.item()*len(ids)
        if not seen:
            break
        member_p = member_predictions(model, prep, early_X, family, device)
        if family == 'tabpack':
            weights = greedy_members(member_p, early_y)
        p = np.einsum('mnc,m->nc', member_p, weights)
        score = quick_metrics(early_y, p); key = (score['accuracy'], score['macro_f1'])
        epochs = epoch+1
        entry = dict(epoch=epochs, fitting_rows_seen=seen, full_epoch=seen==len(y),
            mean_loss=loss_sum/seen, early_stopping=score, member_weights=weights,
            elapsed_seconds=time.monotonic()-started)
        with (output/'epochs.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry)+'\n')
        print(json.dumps(entry), flush=True)
        if key > best:
            best, stale = key, 0
            state = {k:v.detach().as_subclass(torch.Tensor).cpu().clone() for k,v in model.state_dict().items()}
            saved = dict(family=family, config=config, classes=manifest['classes'], seed=seed,
                feature_names=manifest['feature_names'], benign_index=manifest['benign_index'],
                preprocessor=prep.to_dict(), state_dict=state, member_weights=weights,
                best_epoch=epochs, selection_metrics=score)
            torch.save(saved, checkpoint)
        else:
            stale += 1
        if scheduler is not None:
            scheduler.step(score['accuracy'])
        del member_p
        if stale >= config['patience']:
            stop_reason = 'early_stopping'; break
        if stop_reason == 'runtime_limit':
            break
    if not checkpoint.exists():
        raise RuntimeError('No evaluated neural checkpoint')
    model, prep, saved, weights = load(checkpoint, device)
    ids = np.linspace(0, len(early_y)-1, min(512, len(early_y)), dtype=int)
    sample = early_X[ids]
    reference = predict(model, prep, sample, family, weights, device)
    cpu, replay_prep, _, replay_weights = load(checkpoint)
    cpu_p = predict(cpu, replay_prep, sample, family, replay_weights)
    diagnostic = parity_diagnostics(reference, cpu_p)
    if diagnostic['max_absolute_difference'] > .001 or diagnostic['changed_decisions'] > max(1, int(.01*len(sample))):
        raise RuntimeError(f'CPU/GPU numerical discrepancy needs investigation: {diagnostic}')
    other, other_prep, _, other_weights = load(checkpoint)
    replay = predict(other, other_prep, sample, family, other_weights)
    np.testing.assert_allclose(cpu_p, replay, rtol=0, atol=0)
    return dict(training_seconds=time.monotonic()-started, epochs=epochs,
        best_epoch=saved['best_epoch'], stop_reason=stop_reason, training_device=device,
        cpu_gpu_diagnostics=diagnostic, cpu_checkpoint_reload_verified=True,
        parameter_count=sum(p.numel() for p in model.parameters()), member_weights=weights)
