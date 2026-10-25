"""Isolation and selection safeguards for the new cleaned-data study."""
import numpy as np
import pytest


def test_plan_covers_four_families_both_arms_and_rf_retuning():
    from experiment.train_verified_clean import configurations
    rows = configurations()
    assert len(rows) == 24
    assert {r['family'] for r in rows} == {'random_forest', 'xgboost', 'kan', 'hem'}
    for family in {r['family'] for r in rows}:
        assert {r['arm'] for r in rows if r['family'] == family} == {'full', 'strict'}
    rf = [r['config'] for r in rows if r['family'] == 'random_forest']
    assert len({str(r) for r in rf}) == 3


def test_temporal_tuning_calibration_assessment_have_no_touching_windows():
    from experiment.train_verified_clean import split_validation
    y = np.repeat([0, 1], 60)
    starts = np.tile(np.arange(60) * 10, 2)
    ends = starts + 12
    hashes = np.arange(120)
    codes = split_validation(y, starts, ends, hashes)
    assert set(codes) == {-1, 0, 1, 2}
    for label in (0, 1):
        for left, right in ((0, 1), (1, 2), (0, 2)):
            assert ends[(codes == left) & (y == label)].max() < starts[(codes == right) & (y == label)].min()


def test_selection_ignores_full_validation_and_assessment_accuracy():
    from experiment.train_verified_clean import choose_best
    rows = [dict(family='kan', selection_metrics={'accuracy': .7, 'macro_f1': .6},
                 metrics={'accuracy': .99}),
            dict(family='kan', selection_metrics={'accuracy': .8, 'macro_f1': .5},
                 metrics={'accuracy': .1})]
    assert choose_best(rows, 'kan') is rows[1]


def test_duplicate_vectors_cannot_cross_validation_roles():
    from experiment.train_verified_clean import split_validation
    y = np.zeros(60, dtype=int)
    starts = np.arange(60) * 10
    hashes = np.arange(60)
    hashes[50] = hashes[0]
    codes = split_validation(y, starts, starts + 1, hashes)
    assert codes[0] == 0 and codes[50] == -1


def test_training_loader_rejects_test_and_unrecognized_arms(tmp_path):
    from experiment.train_verified_clean import training_paths
    with pytest.raises(ValueError):
        training_paths(tmp_path, 'test')


def test_real_rf_worker_saves_reloadable_predictions_and_calibrator(tmp_path, monkeypatch):
    import json
    from experiment import train_verified_clean as study
    data, output = tmp_path / 'data', tmp_path / 'study'
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'DATA', data)
    monkeypatch.setattr(study, 'STUDY', output)
    base = data / 'model'; base.mkdir(parents=True)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(180, 8)).astype('float32')
    V = rng.normal(size=(90, 8)).astype('float32')
    y = (X[:, 0] > 0).astype(int)
    v = (V[:, 0] > 0).astype(int)
    v[::4] = 1-v[::4]  # Held-out errors make correctness calibration identifiable.
    for name, a in dict(X_train=X, y_train=y, row_ids_train=np.arange(180),
                        X_validation=V, y_validation=v, row_ids_validation=np.arange(1000, 1090)).items():
        np.save(base / (name + '.npy'), a)
    study.write(base / 'manifest.json', dict(classes=['benign', 'attack'], benign_index=0))
    output.mkdir(exist_ok=True)
    np.save(output / 'validation_roles.npy', np.repeat([0, 1, 2], 30))
    hashes = {p.relative_to(data).as_posix(): study.digest(p) for p in base.iterdir()}
    study.write(output / 'study.json', dict(input_hashes=hashes,
                roles_sha256=study.digest(output / 'validation_roles.npy')))
    run = output / 'runs/rf/attempt_000'; run.mkdir(parents=True)
    study.write(run / 'request.json', dict(family='random_forest', arm='full', seed=42,
                recipe='rf', phase='search', config=dict(n_estimators=5, n_jobs=1, max_depth=3)))
    study.fit(run / 'request.json')
    result = json.loads((run / 'result.json').read_text())
    assert result['checkpoint_reload_verified'] and result['test_loaded'] is False
    result['run_directory'] = run.relative_to(tmp_path).as_posix()
    calibration = study.calibrate(result)
    assert calibration['calibration_rows'] == 30
    assert calibration['assessment_rows'] == 30
    assert calibration['predictions_unchanged']
