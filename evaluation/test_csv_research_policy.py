"""CPU policy checks use toy metrics, never claim fitted research results."""
from copy import deepcopy
import importlib
import importlib.util

import pytest

from experiment.csv_study_policy import controls as original_controls


def policy():
    assert importlib.util.find_spec('experiment.csv_research_policy'), 'research policy is missing'
    return importlib.import_module('experiment.csv_research_policy')


def request(family='kan', **changes):
    row = dict(model_id=family, variant='toy', seed=42, input_profile='full_csv',
               protocol='fixed_train_validation', config=deepcopy(original_controls()[family]),
               pipeline=dict(feature_selection='none', feature_engineering=False,
                             resampling='none', order='selection_then_resampling'))
    row.update(changes)
    return row


def result(seed, **metrics):
    row = request(seed=seed)
    row.update(metrics=dict(accuracy=.96, macro_f1=.95, macro_recall=.92,
                            false_positive_rate=.015), checkpoint_reload_verified=True,
               cuda_verified=True)
    row['metrics'].update(metrics)
    return row


def baseline():
    row = result(42, accuracy=.8, macro_recall=.92, false_positive_rate=.015)
    row['variant'] = 'control'
    return {'kan': row}


def test_fingerprints_ignore_labels_budgets_and_inactive_parameters():
    p = policy()
    first = request()
    second = deepcopy(first)
    second.update(variant='renamed', watchdog_seconds=9999, decision={'label': 'ignored'})
    second['config'].update(runtime_seconds=4000, smoothing=.5, focal_gamma=8.)
    second['pipeline'].update(order='resampling_then_selection', resampling_params={'k': 99})
    assert p.fingerprint(first) == p.fingerprint(second)
    assert len(p.fingerprint(first)) == 64


@pytest.mark.parametrize('change', [dict(seed=43), dict(model_id='hem'),
                                  dict(input_profile='partial_csv'), dict(protocol='other'),
                                  dict(input_sha256='different')])
def test_fingerprints_separate_data_family_seed_and_protocol(change):
    p = policy()
    assert p.fingerprint(request()) != p.fingerprint(request(**change))


@pytest.mark.parametrize('identity', ['protocol_id', 'prepared_manifest_sha256',
                                     'train_row_ids_sha256', 'validation_row_ids_sha256',
                                     'prepared', 'dataset_train_sha256'])
def test_prepared_data_and_protocol_identities_change_fit_and_repeat_keys(identity):
    p = policy()
    first = request(**{identity: 'original'})
    second = request(**{identity: 'different'})
    assert p.fingerprint(first) != p.fingerprint(second)
    assert p.recipe_key(first) != p.recipe_key(second)


def test_active_pipeline_parameters_and_order_are_effective():
    p = policy()
    first = request()
    first['pipeline'].update(feature_selection='mi64', resampling='smotenc',
                             resampling_params={'k_neighbors': 5})
    second = deepcopy(first)
    second['pipeline']['order'] = 'resampling_then_selection'
    assert p.fingerprint(first) != p.fingerprint(second)
    second['pipeline']['order'] = first['pipeline']['order']
    second['pipeline']['resampling_params']['k_neighbors'] = 3
    assert p.fingerprint(first) != p.fingerprint(second)


def test_recipes_preserve_controls_and_emit_every_compatible_solo_pair():
    p = policy()
    base = original_controls()
    snapshot = deepcopy(base)
    recipes, rejected = p.build_recipes(base)
    assert base == snapshot
    assert len({p.fingerprint(r) for r in recipes}) == len(recipes)
    assert all(r['seed'] == 42 for r in recipes)
    for family in ('xgboost', 'kan', 'hem'):
        family_rows = [r for r in recipes if r['model_id'] == family]
        control = next(r for r in family_rows if r['variant'] == 'control')
        assert control['config'] == base[family]
        assert any(r['pipeline']['resampling'] == 'ctgan' for r in family_rows)
        for order in ('selection_then_resampling', 'resampling_then_selection'):
            assert any(r['pipeline']['feature_selection'] == 'mi128' and
                       r['pipeline']['resampling'] == 'smotenc' and
                       r['pipeline']['order'] == order for r in family_rows)
        if family != 'xgboost':
            assert any(r['config'].get('loss') == 'focal' and
                       r['config']['scheduler'] == 'plateau' for r in family_rows)
    assert any(r['reason'] == 'mutually_exclusive' for r in rejected)
    assert any(r['reason'] == 'duplicate_effective_recipe' for r in rejected)
    assert not any(r['model_id'] == 'xgboost' and r['config'].get('loss') == 'focal'
                   for r in recipes)


@pytest.mark.parametrize('family', ['xgboost', 'kan', 'hem'])
def test_capacity_escalates_then_rotates_bounded_tiers(family):
    p = policy()
    leader = request(family)
    before = deepcopy(leader)
    rows = [p.capacity_trial(family, leader, stage) for stage in range(10)]
    assert leader == before
    assert len({p.fingerprint(r) for r in rows}) == len(rows)
    assert [r['watchdog_seconds'] for r in rows[:3]] == [1200, 3600, 14400]
    assert all(r['config']['runtime_seconds'] == r['watchdog_seconds'] - 180 for r in rows)
    if family == 'xgboost':
        assert [r['config']['max_depth'] for r in rows[:3]] == [8, 12, 16]
        assert [r['config']['num_boost_round'] for r in rows[:3]] == [1500, 3000, 6000]
        assert max(r['config']['max_depth'] for r in rows) <= 16
    elif family == 'kan':
        assert rows[2]['config']['hidden'] == [512, 256, 128]
        assert rows[2]['config']['grid_size'] == 15
        assert max(max(r['config']['hidden']) for r in rows) <= 512
    else:
        assert rows[2]['config']['epochs'] == 256
        assert max(r['config']['pca_components'] for r in rows) <= 170
        assert all(r['model_id'] == 'hem' for r in rows)


def test_gate_requires_identical_three_seed_recipe_and_verified_execution():
    p = policy()
    rows = [result(s) for s in (42, 43, 44)]
    report = p.success_report(rows, baseline())
    assert report['success'] is True
    assert report['candidates'][0]['seeds'] == [42, 43, 44]
    rows[-1]['config']['lr'] = .01
    assert p.success_report(rows, baseline())['success'] is False
    rows[-1] = result(44)
    rows[-1]['checkpoint_reload_verified'] = False
    assert p.success_report(rows, baseline())['success'] is False
    rows[-1]['checkpoint_reload_verified'] = True
    rows[-1]['cuda_verified'] = False
    assert p.success_report(rows, baseline())['success'] is False


@pytest.mark.parametrize('metrics', [dict(accuracy=.949), dict(macro_recall=.909),
                                     dict(false_positive_rate=.0201), dict(accuracy=float('nan'))])
def test_gate_blocks_below_target_material_regression_and_nonfinite_metrics(metrics):
    p = policy()
    report = p.success_report([result(42), result(43), result(44, **metrics)], baseline())
    assert report['success'] is False
    assert report['rejections']


def test_gate_requires_control_metrics_and_counts_threshold_boundaries():
    p = policy()
    rows = [result(s, macro_recall=.91, false_positive_rate=.02) for s in (42, 43, 44)]
    assert p.success_report(rows, baseline())['success'] is True
    assert p.success_report(rows, {})['success'] is False
    rows[-1]['metrics']['false_positive_rate'] = None
    assert p.success_report(rows, baseline())['success'] is False


def test_score_accuracy_first_macro_f1_tiebreak():
    p = policy()
    assert p.score(result(42)) == (.96, .95)


def test_optuna_proposals_persist_and_avoid_reissued_effective_recipes(tmp_path):
    pytest.importorskip('optuna')
    p = policy()
    db = tmp_path / 'nested' / 'study.sqlite3'
    leader = request('xgboost')
    first = p.propose_tpe(db, 'xgboost', leader, [], 0)
    recovered = p.propose_tpe(db, 'xgboost', leader, [], 0)
    assert recovered == first
    assert first['seed'] == 42 and first['model_id'] == 'xgboost'
    assert db.is_file()
    assert len([k for k in first['config'] if first['config'][k] != leader['config'].get(k)]) == 1
    p.tell_trial(db, 'xgboost', first['optuna_trial_number'], .81)
    second = p.propose_tpe(db, 'xgboost', leader, [first], 1)
    assert first['optuna_study_name'] == second['optuna_study_name']
    assert first['optuna_trial_number'] != second['optuna_trial_number']
    assert p.fingerprint(first) != p.fingerprint(second)
    import optuna
    study = optuna.load_study(study_name=first['optuna_study_name'],
                              storage='sqlite:///' + db.resolve().as_posix())
    assert study.trials[first['optuna_trial_number']].value == .81


def test_optuna_reports_and_records_pruned_failed_trials(tmp_path):
    pytest.importorskip('optuna')
    p = policy()
    db = tmp_path / 'study.sqlite3'
    row = p.propose_tpe(db, 'kan', request(), [], 0)
    assert p.report_trial(db, 'kan', row['optuna_trial_number'], 1, .8) is False
    p.tell_trial(db, 'kan', row['optuna_trial_number'], None, state='pruned')
    import optuna
    study = optuna.load_study(study_name=row['optuna_study_name'],
                              storage='sqlite:///' + db.resolve().as_posix())
    trial = study.trials[row['optuna_trial_number']]
    assert trial.intermediate_values == {1: .8}
    assert trial.state == optuna.trial.TrialState.PRUNED
    # Recovery can tell the same terminal outcome again, but cannot rewrite it.
    p.tell_trial(db, 'kan', row['optuna_trial_number'], None, state='pruned')
    with pytest.raises(ValueError):
        p.tell_trial(db, 'kan', row['optuna_trial_number'], .9)


def test_fixed_rf_cannot_be_tuned_or_given_capacity():
    p = policy()
    with pytest.raises(ValueError):
        p.capacity_trial('rf', request(), 0)


def test_capacity_cannot_relabel_another_model_family():
    p = policy()
    with pytest.raises(ValueError):
        p.capacity_trial('hem', request('kan'), 0)


def test_fingerprint_equates_integral_numbers_and_excludes_prior_tuner_metadata():
    p = policy()
    row = request()
    row['config']['batch_size'] = 512.
    assert p.fingerprint(row) == p.fingerprint(request())
    row.update(optuna_trial_number=9, optuna_study_name='old')
    capacity = p.capacity_trial('kan', row, 0)
    assert 'optuna_trial_number' not in capacity


def test_fingerprint_normalizes_implicit_focal_gamma_and_inactive_engineering():
    p = policy()
    first = request()
    first['config']['loss'] = 'focal'
    second = deepcopy(first)
    second['config']['focal_gamma'] = 2
    second['pipeline']['feature_engineering_params'] = {'unused': True}
    assert p.fingerprint(first) == p.fingerprint(second)


def test_pruned_duplicate_asks_do_not_count_as_measured_pruning_evidence(tmp_path):
    pytest.importorskip('optuna')
    p = policy()
    row = p.propose_tpe(tmp_path / 'study.sqlite3', 'xgboost', request('xgboost'), [], 0)
    assert row['config']['learning_rate'] != .08


@pytest.mark.parametrize('family', ['xgboost', 'kan', 'hem'])
def test_class_weighting_has_solo_and_compatible_pair_evidence(family):
    p = policy()
    rows, rejected = p.build_recipes(original_controls())
    rows = [r for r in rows if r['model_id'] == family]
    alternative = 'sqrt_balanced' if family == 'hem' else 'none'
    assert any(r['variant'].startswith('solo_') and
               r['config']['weight_strategy'] == alternative and
               r['pipeline']['resampling'] == 'none' for r in rows)
    assert any(r['variant'].startswith('solo_') and
               r['config']['weight_strategy'] == 'balanced' for r in rows)
    for sampler in ('smotenc', 'borderline_smote', 'adasyn', 'ctgan'):
        assert any(r['variant'].startswith('pair_') and
                   r['config']['weight_strategy'] == alternative and
                   r['pipeline']['resampling'] == sampler for r in rows)
    assert any(r['model_id'] == family and r['reason'] == 'mutually_exclusive' and
               all(name.startswith('weight_') for name in r['factors']) for r in rejected)


def test_active_augmentation_defaults_deduplicate_and_inactive_ctgan_epochs_are_ignored():
    p = policy()
    implicit = request()
    implicit['pipeline']['resampling'] = 'smotenc'
    explicit = deepcopy(implicit)
    explicit['pipeline']['resampling_params'] = dict(max_extra_fraction=.2,
        max_per_class_multiplier=2, k_neighbors=5, ctgan_epochs=900)
    assert p.fingerprint(implicit) == p.fingerprint(explicit)
    explicit['pipeline']['resampling_params']['max_extra_fraction'] = .4
    assert p.fingerprint(implicit) != p.fingerprint(explicit)
    ctgan = deepcopy(implicit)
    ctgan['pipeline']['resampling'] = 'ctgan'
    changed = deepcopy(ctgan)
    changed['pipeline']['resampling_params'] = {'ctgan_epochs': 40}
    assert p.fingerprint(ctgan) != p.fingerprint(changed)


@pytest.mark.parametrize('sampler', ['smotenc', 'ctgan'])
@pytest.mark.parametrize('family', ['xgboost', 'kan', 'hem'])
def test_tpe_tunes_active_selection_and_augmentation_without_changing_sampler(tmp_path, sampler, family):
    pytest.importorskip('optuna')
    p = policy()
    leader = request(family)
    leader['pipeline'].update(feature_selection='mi64', resampling=sampler)
    snapshot = deepcopy(leader)
    db = tmp_path / (sampler + '.sqlite3')
    emitted = []
    for index in range(8, 13 if sampler == 'ctgan' else 12):
        row = p.propose_tpe(db, family, leader, emitted, index)
        emitted.append(row)
        p.tell_trial(db, family, row['optuna_trial_number'], .8)
    assert leader == snapshot
    assert all(r['pipeline']['resampling'] == sampler for r in emitted)
    assert any(r['pipeline']['feature_selection'] == 'mi128' for r in emitted)
    defaults = {'max_extra_fraction': .2, 'max_per_class_multiplier': 2, 'k_neighbors': 5}
    for parameter, default in defaults.items():
        assert any(r['pipeline'].get('resampling_params', {}).get(parameter, default) != default
                   for r in emitted), parameter
    if sampler == 'ctgan':
        assert any(r['pipeline']['resampling_params'].get('ctgan_epochs', 20) != 20 for r in emitted)
    else:
        assert all('ctgan_epochs' not in r['pipeline'].get('resampling_params', {}) for r in emitted)
    assert len({p.fingerprint(r) for r in emitted}) == len(emitted)


def test_generated_recipe_discards_inherited_inactive_sampler_parameters():
    p = policy()
    controls = {family: request(family) for family in ('xgboost', 'kan', 'hem')}
    for row in controls.values():
        row['pipeline'].update(resampling='ctgan', resampling_params={'ctgan_epochs': 40})
    rows, _ = p.build_recipes(controls)
    neighbors = [r for r in rows if r['variant'] != 'control' and
                 r['pipeline']['resampling'] in ('smotenc', 'borderline_smote', 'adasyn')]
    assert neighbors
    assert all('ctgan_epochs' not in r['pipeline'].get('resampling_params', {}) for r in neighbors)
