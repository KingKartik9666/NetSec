"""77-class native bundle (cascade v3): schema, labels, calibration contract and exporter helpers."""
import json
from pathlib import Path

import numpy as np
import pytest

from tools.native_export import export_v3

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'artifacts/application/native_models_v1'
RECON_SUBTYPES = ('recon_host-disc-arp-ping', 'recon_host-disc-tcp-ack-ping', 'recon_host-disc-tcp-syn-ping',
                  'recon_host-disc-tcp-syn-stealth', 'recon_host-disc-udp-ping', 'recon_os-scan',
                  'recon_port-scan', 'recon_vuln-scan')


def bundle():
    path = BUNDLE / 'catalog.json'
    if not path.is_file():
        pytest.skip('native bundle not built')
    catalog = json.loads(path.read_text(encoding='utf-8'))
    if len(catalog['classes']) != 77:
        pytest.skip('application bundle is not the 77-class export')
    return catalog


def test_catalog_is_77_class_with_single_recon_class():
    catalog = bundle()
    assert len(catalog['feature_names']) == 445 and catalog['classes'][0] == 'benign'
    assert 'recon_scan' in catalog['classes'] and 'recon_ping-sweep' in catalog['classes']
    assert not set(RECON_SUBTYPES) & set(catalog['classes'])
    assert catalog['test_loaded'] is False and catalog['default_model'] == 'tabm_xgboost_kan'
    assert len([m for m in catalog['models'] if not m['auxiliary']]) == 10


def test_samples_and_reference_carry_no_recon_subtype_labels():
    catalog = bundle()
    for name, partition in (('samples.json', 'validation'), ('reference.json', 'train')):
        rows = json.loads((BUNDLE / name).read_text(encoding='utf-8'))
        assert rows['partition'] == partition and rows['classes'] == catalog['classes']
        labels = {row['label'] for row in rows['rows']}
        assert not labels & set(RECON_SUBTYPES)
        assert all(len(row['values']) == 445 for row in rows['rows'])
    assert 'recon_scan' in {row['label'] for row in json.loads((BUNDLE / 'samples.json').read_text())['rows']}


def test_every_model_has_its_calibration_contract():
    catalog = bundle()
    for entry in catalog['models']:
        manifest = json.loads((BUNDLE / entry['manifest']).read_text(encoding='utf-8'))
        assert manifest['classes'] == catalog['classes'] and manifest['feature_names'] == catalog['feature_names']
        if entry['kind'] == 'hem':
            assert len(manifest['correctness_calibrator']['coefficients']) == 3
            assert manifest['label'] == 'HEM-derived'
            continue
        cal = manifest['calibration']
        assert cal['kind'] in ('temperature', 'vector', 'identity') and cal['calibration_fit_role'] == 'calibration'
        assert cal['binary_platt']['fit_role'] == 'calibration' and cal['binary_platt']['selection_role'] == 'tuning'
        if cal['kind'] == 'vector':
            assert len(cal['bias']) == 77 and cal['scale'] > 0


def test_apply_cal_decides_vector_by_calibrated_argmax_and_temperature_by_raw():
    raw = np.array([[0.7, 0.2, 0.1, 0.0]])
    platt = dict(slope=1.0, intercept=0.0)
    vector = dict(kind='vector', scale=1.5, alpha=0.01, bias=[0.0, 2.5, 0.0, -2.5], binary_platt=platt)
    predicted, p, attack = export_v3.apply_cal(raw, vector)
    assert predicted[0] == 1 and np.isclose(p.sum(), 1) and np.isclose(attack[0], 0.3)
    temperature = dict(kind='temperature', temperature=0.85, alpha=0.01, binary_platt=platt)
    predicted, p, _ = export_v3.apply_cal(raw, temperature)
    assert predicted[0] == 0
    np.testing.assert_allclose(p[0], [0.7481042221075982, 0.17316215400904042, 0.07773765025247581,
                                      0.0009959736308856324], atol=1e-6)


def test_platt_keeps_identity_when_tuning_error_does_not_drop():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    attack = np.where(y == 1, 0.9, 0.1)                       # already perfectly calibrated
    raw = np.column_stack([1 - attack, attack])
    roles = np.repeat([0, 1, 2], [1500, 1500, 1000])
    result = export_v3.platt(raw, y, roles)
    assert result['fit_role'] == 'calibration'
    if not result['adopted']:
        assert (result['slope'], result['intercept']) == (1.0, 0.0)
    else:
        assert result['tuning_after']['ece'] < result['tuning_before']['ece']
