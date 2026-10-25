"""Lossless DataSense audit primitives. No model, label inference, or test access."""
from __future__ import annotations

import ast
import csv
import hashlib
import ipaddress
import json
import re
from pathlib import Path

import numpy as np


def parse_list(text, kind):
    """Return a canonical measured set and original length; unknown is an error."""
    if not isinstance(text, str) or len(text) > 2_000_000:
        raise ValueError('Expected bounded list text')
    try:
        value = json.loads(text.replace("'", '"'))
    except (ValueError, RecursionError):
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError, RecursionError) as exc:
            raise ValueError('Invalid list syntax') from exc
    if not isinstance(value, list):
        raise ValueError('Expected list; unknown is not an empty observation')
    tokens = []
    for token in value:
        if kind == 'ports':
            if isinstance(token, str) and token.isascii() and token.isdigit():
                token = int(token)
            if type(token) is not int or not 0 <= token <= 65535:
                raise ValueError('Invalid port')
        else:
            if not isinstance(token, str) or not token or len(token) > 200:
                raise ValueError('Invalid list token')
            if kind == 'ips':
                try:
                    ipaddress.ip_address(token)
                except ValueError as exc:
                    raise ValueError('Invalid IP address') from exc
            if kind == 'macs' and not re.fullmatch(r'[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}', token):
                raise ValueError('Invalid MAC address')
        tokens.append(token)
    return tuple(sorted(set(tokens))), len(tokens)


def selected_rows(source, row_ids, expected_rows=None):
    """Only yield whitelisted data; other CSV records are traversed, not analyzed."""
    indexes = np.asarray(row_ids, dtype=np.int64)
    if len(indexes) and (np.any(np.diff(indexes) <= 0) or indexes[0] < 0):
        raise ValueError('Selected IDs must be sorted and unique')
    csv.field_size_limit(8 * 1024 * 1024)
    opener = __import__('gzip').open if str(source).endswith('.gz') else open
    position = 0
    count = 0
    with opener(source, 'rt', encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        headers = next(reader)
        if len(headers) != len(set(headers)):
            raise ValueError('Duplicate column names')
        for source_id, values in enumerate(reader):
            count += 1
            if position < len(indexes) and source_id == indexes[position]:
                if len(values) != len(headers):
                    raise ValueError(f'Malformed selected record {source_id}')
                yield source_id, dict(zip(headers, values))
                position += 1
    if position != len(indexes) or (expected_rows is not None and count != expected_rows):
        raise ValueError('Source row count or selected IDs changed')


def zero_bitmap(values):
    return np.packbits(np.asarray(values) == 0, axis=1, bitorder='little')


def no_signal_mask(values, list_lengths):
    return np.all(values == 0, axis=1) & np.all(list_lengths == 0, axis=1)


def censored_presence(tokens, vocabulary, recorded_count):
    """Unknown absence in an observed capped list is NaN, never a guessed zero."""
    observed = set(tokens)
    censored = len(observed) >= 1000 and recorded_count > len(observed)
    unknown = censored or not np.isfinite(recorded_count)
    values = [1. if token in observed else (np.nan if unknown else 0.) for token in vocabulary]
    values.append(1. if observed.difference(vocabulary) else (np.nan if unknown else 0.))
    return np.asarray(values, dtype=np.float32)


def mask_unsupported_statistics(values, names):
    """Model representation: zero summaries with zero supporting events are NA.

    This is an explicit derived interpretation, not a repair of the raw CSV.
    Counts, legitimate measured zeros and all signed observations stay intact.
    """
    cleaned = np.array(values, copy=True)
    changed = np.zeros(cleaned.shape, dtype=bool)
    lookup = {name:i for i,name in enumerate(names)}
    for i,name in enumerate(names):
        if not name.endswith(('_avg','_min','_max','_std_deviation')):
            continue
        count = 'log_messages_count' if name.startswith('log_') else 'network_packets_all_count'
        if count not in lookup:
            continue
        unsupported = (values[:,lookup[count]] == 0) & (values[:,i] == 0)
        cleaned[unsupported,i] = np.nan
        changed[unsupported,i] = True
    return cleaned, changed


def lossless_groups(values, labels):
    """Compress exact (input, label) repetitions; verify every inverse mapping."""
    labels = np.asarray(labels)
    hashes = np.empty(len(labels), dtype='S32')
    for start in range(0, len(labels), 4096):
        block = np.array(values[start:start+4096], dtype=np.float32, order='C', copy=True)
        block[block == 0] = 0
        block[np.isnan(block)] = np.nan
        for j,row in enumerate(block):
            hashes[start+j] = hashlib.sha256(row.tobytes() + int(labels[start+j]).to_bytes(8,'little',signed=True)).digest()
    _, first, inverse, counts = np.unique(hashes, return_index=True, return_inverse=True, return_counts=True)
    order = np.argsort(first)
    remap = np.empty(len(order), dtype=np.int64)
    remap[order] = np.arange(len(order))
    first, inverse, counts = first[order], remap[inverse], counts[order]
    for start in range(0, len(labels), 4096):
        mapped = first[inverse[start:start+4096]]
        if not np.array_equal(values[mapped], values[start:start+4096], equal_nan=True):
            raise ValueError('Input grouping failed exact reconstruction')
        if not np.array_equal(labels[mapped], labels[start:start+4096]):
            raise ValueError('Grouping merged different labels')
    return first, inverse, counts


def weighted_smoothed_derivatives(margins, labels, weights, smoothing=.05):
    probabilities = np.exp(margins - margins.max(axis=1,keepdims=True))
    probabilities /= probabilities.sum(axis=1,keepdims=True)
    hessian = np.maximum(2*probabilities*(1-probabilities),1e-6)
    gradient = probabilities - smoothing/probabilities.shape[1]
    gradient[np.arange(len(labels)),labels] -= 1-smoothing
    gradient *= weights[:,None]
    hessian *= weights[:,None]
    return gradient,hessian


def exact_redundancy(values, names, protected=()):
    """Find only exact training constants/copies; never use correlation cutoffs."""
    retained, duplicates, constants, buckets = [], {}, {}, {}
    for column, name in enumerate(names):
        vector = np.array(values[:, column], copy=True)
        if name not in protected and np.all(vector == vector[0]):
            constants[name] = float(vector[0])
            continue
        vector[vector == 0] = 0
        vector[np.isnan(vector)] = np.nan
        key = hashlib.sha256(vector.tobytes()).hexdigest()
        same = next((i for i in buckets.get(key, [])
                     if np.array_equal(values[:, i], vector, equal_nan=True)), None)
        if same is not None and name not in protected:
            duplicates[name] = names[same]
        else:
            retained.append(column)
            buckets.setdefault(key, []).append(column)
    return {'retained': retained, 'duplicates': duplicates, 'constants': constants}


def binary_information(positive_counts, class_counts):
    """Training-only mutual information of exported-list occurrence and target.

    Counts are classes by candidate tokens. Absence here means not exported,
    not proof that a port was absent from a potentially capped capture list.
    """
    positive = np.asarray(positive_counts, dtype=np.float64)
    totals = np.asarray(class_counts, dtype=np.float64)[:, None]
    if positive.ndim != 2 or len(totals) != len(positive):
        raise ValueError('Expected class-by-token counts')
    if np.any(positive < 0) or np.any(positive > totals) or totals.sum() <= 0:
        raise ValueError('Invalid occurrence counts')
    n = totals.sum()
    information = np.zeros(positive.shape[1])
    for joint in (positive, totals-positive):
        denominator = totals * joint.sum(axis=0, keepdims=True)
        ratio = np.ones_like(joint)
        np.divide(joint*n, denominator, out=ratio, where=denominator>0)
        term = np.zeros_like(joint)
        np.log(ratio, out=term, where=joint>0)
        information += (joint*term).sum(axis=0)/n
    return information


def calibration_partition(labels, starts, ends, hashes):
    """Chronological halves within each validation class; strict gap + copies.

    This is an exploratory calibration-assessment split, never a locked test.
    Very short classes are evaluation-only instead of inventing calibration data.
    """
    from framework.data.research_data import prevent_duplicate_crossing
    labels,starts,ends,hashes=map(np.asarray,(labels,starts,ends,hashes))
    if not (labels.shape==starts.shape==ends.shape==hashes.shape) or np.any(ends<starts):
        raise ValueError('Misaligned calibration provenance')
    result=np.full(len(labels),-1,dtype=np.int8)
    for label in np.unique(labels):
        ids=np.flatnonzero(labels==label);times=np.unique(starts[ids])
        if len(times)<6:
            result[ids]=1
            continue
        boundary=times[len(times)//2]
        result[ids[ends[ids]<boundary]]=0
        result[ids[starts[ids]>boundary]]=1
    return prevent_duplicate_crossing(result,hashes)[0]


def temperature_probabilities(probabilities, temperature):
    """Scalar monotone calibration; scores change, class decisions stay fixed."""
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=2 or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Expected normalized multiclass probabilities')
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError('Temperature must be positive')
    z=np.log(np.clip(p,1e-12,1))/temperature
    z-=z.max(axis=1,keepdims=True)
    calibrated=np.exp(z);calibrated/=calibrated.sum(axis=1,keepdims=True)
    return calibrated


def verify_probability_parity(original,reloaded,atol=1e-7):
    """Bound backend rounding independently of mandatory identical decisions."""
    np.testing.assert_array_equal(np.asarray(original).argmax(1),np.asarray(reloaded).argmax(1))
    np.testing.assert_allclose(reloaded,original,atol=atol,rtol=0)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
