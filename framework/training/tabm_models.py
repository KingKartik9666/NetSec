"""Official TabM backbone with our masked input and hierarchical adaptations."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from tabm import EnsembleView, LinearEnsemble, make_tabm_backbone
from rtdl_num_embeddings import PiecewiseLinearEmbeddings

FAMILIES = ('tabm_ne', 'tabm_hier', 'tabm_hier_ne')


def member_probabilities(logits, hierarchical=False):
    if not hierarchical:
        return logits.float().softmax(-1)
    attack = logits[..., :1].float().sigmoid()
    subtype = logits[..., 1:].float().softmax(-1)
    return torch.cat((1-attack, attack*subtype), dim=-1)


def member_loss(logits, labels, hierarchical=False, weights=None, subtype_weight=1., smoothing=0.):
    batch, members, classes = logits.shape
    targets = labels[:, None].expand(-1, members)
    if hierarchical:
        mask = targets != 0
        binary = F.binary_cross_entropy_with_logits(logits[..., 0].float(), mask.float(), reduction='none')
        subtype = F.cross_entropy(logits[..., 1:].float().reshape(-1, classes-1),
            (targets-1).clamp_min(0).reshape(-1), reduction='none', label_smoothing=smoothing).reshape(batch, members)
        losses = binary + subtype_weight*mask*subtype
    else:
        losses = F.cross_entropy(logits.float().reshape(-1, classes), targets.reshape(-1),
                                reduction='none', label_smoothing=smoothing).reshape(batch, members)
    if weights is not None:
        losses = losses*weights[labels, None]
    return losses.mean()


class TabMClassifier(nn.Module):
    def __init__(self, family, preprocessor, n_classes, config):
        super().__init__()
        if family not in FAMILIES or n_classes < 2:
            raise ValueError('Unknown TabM family or class vocabulary')
        self.hierarchical = family != 'tabm_ne'
        self.embedded = family != 'tabm_hier' and bool(preprocessor.embedded_indices)
        self.register_buffer('numeric_indices', torch.tensor(preprocessor.embedded_indices, dtype=torch.long))
        self.register_buffer('bypass_indices', torch.tensor(preprocessor.bypass_indices, dtype=torch.long))
        k, width = int(config['k']), int(config['width'])
        dimension = 2*len(preprocessor.feature_names)
        chunks = [1]*dimension
        self.embeddings = None
        if self.embedded:
            d = int(config['d_embedding'])
            self.embeddings = PiecewiseLinearEmbeddings(
                [torch.tensor(b, dtype=torch.float32) for b in preprocessor.bins],
                d_embedding=d, activation=False, version='B')
            dimension = len(preprocessor.embedded_indices)*d + len(preprocessor.bypass_indices)
            chunks = [d]*len(preprocessor.embedded_indices)+[1]*len(preprocessor.bypass_indices)
        self.view = EnsembleView(k=k)
        self.backbone = make_tabm_backbone(d_in=dimension, n_blocks=int(config['depth']),
            d_block=width, dropout=float(config['dropout']), k=k, arch_type='tabm',
            start_scaling_init='normal' if self.embedded else 'random-signs', start_scaling_init_chunks=chunks)
        self.head = LinearEnsemble(width, n_classes, k=k)

    def forward(self, X):
        if self.embedded:
            X = torch.cat((self.embeddings(X[:, self.numeric_indices]).flatten(1),
                           X[:, self.bypass_indices]), dim=-1)
        return self.head(self.backbone(self.view(X)))

    def predict_proba(self, X):
        return member_probabilities(self(X), self.hierarchical).mean(1)
