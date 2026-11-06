"""Observable feature-selection contracts; no locked test or capture required."""
import numpy as np
import pytest

from framework.training.live_features.schema import core_names, extended_names, derive
from framework.training.live_features.selection import remove_redundant, purged_folds


def test_broad_profile_preserves_families_and_excludes_unjustified_values():
    core=core_names();extended=extended_names()
    assert len(core)==68 and len(extended)==73
    for key in ('packet-size','header-length','ip-length','time-delta','window-size','ttl','mss'):
        assert any(key in name for name in core)
    for name in core:
        assert not name.startswith(('label','log_','device_'))
        assert not any(part in name for part in ('__present_', 'fragmentation-score','interval-packets','protocols_','macs_dst_count','payload'))


def test_derivation_keeps_unknown_separate_from_zero():
    names=['network_packets_all_count','network_tcp-flags-syn_count','network_packets_src_count',
        'network_packets_dst_count','network_packet-size_avg','network_packet-size_std_deviation']
    x=np.array([[10,2,4,6,100,20],[0,0,0,0,0,0],[10,np.nan,4,6,100,20]],np.float32)
    out=derive(x,names,10)
    assert out['network_tcp-flags-syn__fraction_of_packets'][0]==pytest.approx(.2)
    assert np.isnan(out['network_tcp-flags-syn__fraction_of_packets'][1:]).all()
    assert out['network_src_packet_fraction'][0]==pytest.approx(.4)
    assert out['network_packet_size_coefficient_of_variation'][0]==pytest.approx(.2)


def test_redundancy_preserves_distinct_highly_correlated_measurements():
    a=np.arange(100,dtype=np.float32);b=a.copy();b[-1]+=.001
    x=np.column_stack((a,b,a,np.ones(100),np.full(100,np.nan)))
    kept,removed=remove_redundant(x,['size_avg','size_max','copy','constant','missing'])
    assert kept==[0,1]
    assert removed['copy']['reason']=='exact_training_duplicate'
    assert removed['constant']['reason']=='constant_on_fitting_rows'
    assert removed['missing']['reason']=='all_missing_on_fitting_rows'


def test_selector_folds_purge_intervals_and_duplicate_vectors():
    n=600;y=np.repeat([0,1],n//2)
    x=np.column_stack((np.arange(n)+1,np.zeros((n,6)))).astype(np.float32)
    start=np.r_[np.arange(300),10000+np.arange(300)].astype(np.int64)*5_000_000_000
    end=start+10_000_000_000
    folds=purged_folds(x,y,start,end)
    assert len(folds)==3
    for train,val in folds:
        assert not set(map(tuple,x[train])).intersection(map(tuple,x[val]))
        for label in (0,1):
            assert end[train[y[train]==label]].max()<start[val[y[val]==label]].min()
