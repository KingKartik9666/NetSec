"""Protect evidence when preparing an isolated, lossless DataSense candidate."""
import importlib
import importlib.util

import numpy as np
import pytest


def quality():
    assert importlib.util.find_spec('experiment.datasense_quality') is not None, 'Quality audit implementation required'
    return importlib.import_module('experiment.datasense_quality')


def test_empty_observation_is_distinct_from_unknown_list():
    q = quality()
    assert q.parse_list('[]', 'ports') == ((), 0)
    assert q.parse_list("['0', '443', '443']", 'ports') == ((0, 443), 3)
    for value in ('', 'NaN', 'null', '[true]', '[65536]', '[-1]', '[1.5]', '__import__("os").getcwd()'):
        with pytest.raises(ValueError):
            q.parse_list(value, 'ports')


def test_protocol_and_log_lists_preserve_rare_tokens():
    q = quality()
    assert q.parse_list("['mqtt', 'tcp', 'mqtt']", 'protocols') == (('mqtt', 'tcp'), 3)
    assert q.parse_list("['float', 'int']", 'log') == (('float', 'int'), 2)


def test_source_reader_does_not_validate_unselected_payloads(tmp_path):
    q = quality()
    source = tmp_path / 'source.csv'
    source.write_text('label4,value\na,0\nLOCKED,not-a-number\nb,-0.78\n', encoding='utf-8')
    result = list(q.selected_rows(source, np.array([0, 2]), expected_rows=3))
    assert result == [(0, {'label4': 'a', 'value': '0'}), (2, {'label4': 'b', 'value': '-0.78'})]
    with pytest.raises(ValueError, match='row count'):
        list(q.selected_rows(source, np.array([0, 2]), expected_rows=4))


def test_zero_mask_records_cells_and_does_not_treat_missing_as_zero():
    q = quality()
    values = np.array([[0., -0., np.nan, 2.], [-0.78, 0., 4., np.inf]])
    packed = q.zero_bitmap(values)
    np.testing.assert_array_equal(np.unpackbits(packed, axis=1, count=4, bitorder='little'),
                                  [[1, 1, 0, 0], [0, 1, 0, 0]])


def test_redundancy_is_lossless_and_not_merely_high_correlation():
    q = quality()
    values = np.array([[0, 0, 7, 0], [1, 1, 7, 1.001], [2, 2, 7, 2.]])
    mapping = q.exact_redundancy(values, ['a', 'a_copy', 'constant', 'similar'])
    assert mapping['retained'] == [0, 3]
    assert mapping['duplicates'] == {'a_copy': 'a'}
    assert mapping['constants'] == {'constant': 7.0}


def test_no_signal_needs_both_numeric_and_categorical_absence():
    q = quality()
    x = np.array([[0, 0], [0, 0], [0, np.nan], [1, 0]])
    lengths = np.array([[0], [1], [0], [0]])
    np.testing.assert_array_equal(q.no_signal_mask(x, lengths), [True, False, False, False])


def test_censored_list_does_not_claim_unlisted_port_is_absent():
    q = quality()
    complete = q.censored_presence((80,), [80, 443], 1)
    np.testing.assert_equal(complete, [1, 0, 0])
    partial = q.censored_presence(tuple(range(1000)), [80, 8080], 65000)
    assert partial[0] == 1 and np.isnan(partial[1]) and partial[2] == 1


def test_absence_masking_retains_true_zero_and_negative_observations():
    q = quality()
    names = ['log_messages_count', 'log_data-ranges_min', 'log_data-ranges_max',
             'network_packets_all_count', 'network_packet-size_avg', 'network_time-delta_min']
    values = np.array([[0, 0, 0, 0, 0, 0], [2, -0.78, 0, 3, 60, -1e-6]], dtype=float)
    cleaned, changed = q.mask_unsupported_statistics(values, names)
    np.testing.assert_equal(cleaned[0], [0, np.nan, np.nan, 0, np.nan, np.nan])
    np.testing.assert_equal(cleaned[1], values[1])
    assert changed.sum() == 4
    np.testing.assert_equal(values[0], [0, 0, 0, 0, 0, 0])


def test_redundancy_grouping_preserves_labels_multiplicity_and_nan():
    q = quality()
    x = np.array([[0, np.nan], [1, 2], [0, np.nan], [0, np.nan]], dtype=np.float32)
    y = np.array([0, 1, 0, 2])
    representatives, inverse, counts = q.lossless_groups(x, y)
    np.testing.assert_equal(representatives, [0, 1, 3])
    np.testing.assert_equal(counts, [2, 1, 1])
    np.testing.assert_equal(x[representatives][inverse], x)
    np.testing.assert_equal(y[representatives][inverse], y)


def test_weighted_group_loss_equals_expanded_observations():
    q = quality()
    margins = np.array([[.2,-.3],[.8,.1]],dtype=np.float32)
    y = np.array([0,1])
    weights = np.array([3,1],dtype=np.float32)
    grouped_g,grouped_h=q.weighted_smoothed_derivatives(margins,y,weights)
    expanded_g,expanded_h=q.weighted_smoothed_derivatives(margins[[0,0,0,1]],y[[0,0,0,1]],np.ones(4,dtype=np.float32))
    np.testing.assert_allclose(grouped_g[0],expanded_g[:3].sum(0),rtol=1e-6)
    np.testing.assert_allclose(grouped_h[0],expanded_h[:3].sum(0),rtol=1e-6)
    np.testing.assert_allclose(grouped_g[1],expanded_g[3],rtol=1e-6)


def test_port_selection_rewards_information_not_frequency():
    q=quality()
    # Port 0 is in every record. Port 1 separates classes. Port 2 is uninformative.
    present=np.array([[50,50,25],[50,0,25]])
    scores=q.binary_information(present,np.array([50,50]))
    assert scores[1] > .69
    assert abs(scores[0]) < 1e-12 and abs(scores[2]) < 1e-12


def test_calibration_partition_purges_touching_windows_and_repeated_inputs():
    q=quality()
    labels=np.zeros(12,dtype=np.int64)
    starts=np.arange(12,dtype=np.int64)*10
    ends=starts+10
    hashes=np.arange(12,dtype=np.int64)
    hashes[9]=hashes[1]
    split=q.calibration_partition(labels,starts,ends,hashes)
    assert split[5]==-1 and split[6]==-1
    assert split[1]==0 and split[9]==-1
    assert split[7]==1
    assert ends[split==0].max()<starts[split==1].min()


def test_temperature_calibration_preserves_classes_and_normalization():
    q=quality()
    p=np.array([[.9,.1],[.2,.8],[1.,0.],[.5,.5]])
    calibrated=q.temperature_probabilities(p,2.)
    np.testing.assert_array_equal(calibrated.argmax(1),p.argmax(1))
    np.testing.assert_allclose(calibrated.sum(1),1.)
    assert calibrated[0,0]<p[0,0]


def test_streamed_preprocessing_matches_train_only_reference(tmp_path):
    from experiment.datasense_preprocessing import fit_streamed_preprocessing
    from framework.training.neural import _fit_preprocessing
    train=np.array([[0.,np.nan,1.],[2.,4.,1.],[-2.,8.,1.],[10.,2.,1.]],dtype=np.float32)
    val=np.array([[200.,np.nan,2.],[-200.,4.,1.]],dtype=np.float32)
    config={'scaling':'log_minmax'}
    expected,a,b=_fit_preprocessing('kan',train,val,config,42)
    actual,x,v=fit_streamed_preprocessing('kan',train,val,config,42,tmp_path,batch_size=2)
    np.testing.assert_allclose(x,a,atol=1e-6)
    np.testing.assert_allclose(v,b,atol=1e-6)
    np.testing.assert_allclose(actual.transform(val),expected.transform(val),atol=1e-6)


def test_streamed_pca_training_scores_match_saved_inference(tmp_path):
    from experiment.datasense_preprocessing import fit_streamed_preprocessing
    train=np.random.default_rng(42).normal(size=(40,8)).astype(np.float32)
    train[0,0]=np.nan
    config={'scaling':'log_minmax','pca_components':3}
    prep,transformed,val=fit_streamed_preprocessing('hem',train,train[:3],config,42,tmp_path,batch_size=10)
    np.testing.assert_allclose(transformed,prep.transform(train),atol=1e-6)
    np.testing.assert_allclose(val,transformed[:3],atol=1e-6)


def test_probability_reload_accepts_rounding_but_rejects_decision_changes():
    q=quality()
    original=np.array([[.8,.2]],dtype=np.float32)
    rounded=original.copy();rounded[0,0]=np.nextafter(rounded[0,0],np.float32(1))
    q.verify_probability_parity(original,rounded,atol=2e-7)
    with pytest.raises(AssertionError):
        q.verify_probability_parity(np.array([[.50000004,.49999996]]),np.array([[.49999996,.50000004]]),atol=2e-7)
