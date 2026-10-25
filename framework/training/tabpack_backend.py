"""Pinned upstream TabPack arithmetic with isolated Python 3.11 type compatibility.

The checkout is unmodified. Our bounded training loop is a separate experiment,
not a reproduction of the paper's full dynamic resampling protocol.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT/'external/tabpack'
COMMIT = '05a89e21b955f12de84889d662e15ca534019aaa'
PINS = {
    'src/project/nn.py': 'b1fd1cb6d90831888b41741c89bf785355d4736fb486505528565419f24a0031',
    'src/project/optim.py': '002a07dbb352030c1baef6db6ad54d27f8817325d2257b0adb4f0c08eb87b566',
    'src/project/tabpack.py': '60ea6c5eb699460642cda43da116eaeaa66cbe928716c101d22a0e23ca16d6fb',
    'src/vendor/muon.py': '491b93367068d7c91bf72b89a4df76880581780d8bb8b3befb63441a2559eba3',
}
NAMESPACE = '_imdai_tabpack_upstream'


def upstream_pins(root=UPSTREAM):
    result = {}
    for name, expected in PINS.items():
        path = Path(root)/name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'TabPack upstream source changed: {name}')
        result[name] = actual
    return result


def _module(name, source, filename, extra=None):
    module = types.ModuleType(name)
    module.__file__ = str(filename)
    module.__package__ = NAMESPACE
    if extra:
        module.__dict__.update(extra)
    sys.modules[name] = module
    exec(compile('from __future__ import annotations\n'+source, str(filename), 'exec'), module.__dict__)
    return module


def upstream_modules():
    upstream_pins()  # Check even when cached; never silently run changed sources.
    if NAMESPACE+'.model' in sys.modules:
        return sys.modules[NAMESPACE+'.model'], sys.modules[NAMESPACE+'.optim']
    package = types.ModuleType(NAMESPACE); package.__path__ = []
    sys.modules[NAMESPACE] = package
    nn_path = UPSTREAM/'src/project/nn.py'
    nn_source = nn_path.read_text(encoding='utf-8').replace(
        'from lib.types import KWArgs', 'from typing import Any\nKWArgs = dict[str, Any]')
    upstream_nn = _module(NAMESPACE+'.nn', nn_source, nn_path)
    muon_path = UPSTREAM/'src/vendor/muon.py'
    muon = _module(NAMESPACE+'.muon', muon_path.read_text(encoding='utf-8'), muon_path)
    optim_path = UPSTREAM/'src/project/optim.py'
    optim_source = optim_path.read_text(encoding='utf-8').replace('import vendor.muon', '')
    # The only PEP-695 generic function in this module is a typing-only overload.
    optim_source = optim_source.replace('def _maybe_unsqueeze[T](', 'def _maybe_unsqueeze(')
    optim = _module(NAMESPACE+'.optim', optim_source, optim_path,
                    {'vendor':types.SimpleNamespace(muon=muon)})
    model_path = UPSTREAM/'src/project/tabpack.py'
    source = model_path.read_text(encoding='utf-8')
    # Execute the real model definitions without importing the paper's experiment
    # manager, dataset downloads, environment setup or Python-3.12-only CLI.
    source = source[source.index('def _make_num_module('):source.index('class ApplyModel(')]
    import torch
    model = _module(NAMESPACE+'.model', source, model_path,
        {'project':types.SimpleNamespace(nn=upstream_nn), 'nn':torch.nn, 'Tensor':torch.Tensor,
         'torch':torch})
    return model, optim


def pack_configuration(seed=42):
    # Eight predeclared settings: no validation-driven search-space expansion.
    return dict(seed=int(seed), pack_size=8, n_blocks=[2, 3, 2, 3, 1, 2, 3, 2],
        d_block=[128, 192, 256, 256, 384, 192, 128, 384],
        dropout=[0., .1, .1, .2, .1, .2, 0., .1],
        lr=[.001, .001, .0005, .001, .0005, .002, .0005, .001],
        weight_decay=[.0001, .0003, .001, .0001, .0003, .0001, .001, .0003],
        scaling='quantile', n_bins=16, batch_size=256, epochs=60, patience=8,
        runtime_seconds=3600, upstream_commit=COMMIT,
        protocol='bounded fixed 8-member upstream TabPack with online greedy selection')


def make_pack(n_features, n_classes, config, device='cpu'):
    import torch
    upstream, optim = upstream_modules()
    torch.manual_seed(int(config['seed']))
    model = upstream.ModelPack(n_num_features=n_features, cat_cardinalities=[],
        n_classes=n_classes, pack_size=config['pack_size'], num_embeddings=None,
        n_blocks=config['n_blocks'], max_n_blocks=max(config['n_blocks']),
        d_block=config['d_block'], max_d_block=max(config['d_block']),
        dropout=config['dropout']).to(device)
    optimizer = optim.AdamWPack(model.parameters(), lr=config['lr'],
        weight_decay=config['weight_decay'], pack_size=config['pack_size'])
    return model, optimizer
