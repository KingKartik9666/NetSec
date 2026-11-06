"""Dataset feature meanings are explicit; extractability is not semantic parity."""
from __future__ import annotations

import numpy as np

STATS=('avg','max','min','std_deviation')
FLAGS=('ack','fin','psh','rst','syn','urg')
SUMMARY_GROUPS=('header-length','ip-flags','ip-length','mss','packet-size',
    'payload-length','tcp-flags','time-delta','ttl','window-size')
RAW_NAMES=tuple(sorted(['network_fragmentation-score','network_fragmented-packets','network_interval-packets']+
    [f'network_{group}_{stat}' for group in SUMMARY_GROUPS for stat in STATS]+
    [f'network_{group}_{direction}_count' for group in ('ips','macs','packets','ports','protocols') for direction in ('all','src','dst')]+
    [f'network_tcp-flags-{flag}_count' for flag in FLAGS]))
WITHHELD={
    'network_fragmentation-score':'Publisher formula is not established',
    'network_interval-packets':'Publisher formula and units are not established',
    'network_macs_dst_count':'Source defect: copied source MAC count; native destination count remains separate',
    **{f'network_protocols_{direction}_count':'Publisher application/dissector labels cannot be inferred from L4 headers'
       for direction in ('all','src','dst')}}
PAYLOAD={f'network_payload-length_{stat}':'Protocol payload inclusion is not established; extended research profile only' for stat in STATS}


def derivative_names(extended=False):
    return ([f'network_packets_{d}_count__per_second' for d in ('all','src','dst')]+
        [f'network_tcp-flags-{f}__fraction_of_packets' for f in FLAGS]+
        [f'network_{d}_packet_fraction' for d in ('src','dst')]+
        ['network_packet_size_coefficient_of_variation']+
        (['network_payload_fraction'] if extended else [])+
        ['network_packets_per_reported_ip','network_packets_per_reported_port'])


def core_names():
    return [name for name in RAW_NAMES if name not in WITHHELD and name not in PAYLOAD]+derivative_names()


def extended_names():
    return [name for name in RAW_NAMES if name not in WITHHELD]+derivative_names(True)


def ratio(numerator,denominator):
    numerator,denominator=np.broadcast_arrays(numerator,denominator)
    result=np.full(numerator.shape,np.nan,dtype=np.float32)
    valid=np.isfinite(numerator)&np.isfinite(denominator)&(denominator>0)
    np.divide(numerator,denominator,out=result,where=valid)
    return result


def derive(x,names,seconds):
    index={name:i for i,name in enumerate(names)}
    def col(name):
        return np.asarray(x[:,index[name]],dtype=np.float32) if name in index else np.full(len(x),np.nan,dtype=np.float32)
    packets=col('network_packets_all_count');result={}
    for direction in ('all','src','dst'):
        name=f'network_packets_{direction}_count'
        result[name+'__per_second']=ratio(col(name),seconds)
    for flag in FLAGS:
        result[f'network_tcp-flags-{flag}__fraction_of_packets']=ratio(col(f'network_tcp-flags-{flag}_count'),packets)
    for direction in ('src','dst'):
        result[f'network_{direction}_packet_fraction']=ratio(col(f'network_packets_{direction}_count'),packets)
    for name,numerator,denominator in (
        ('network_packet_size_coefficient_of_variation','network_packet-size_std_deviation','network_packet-size_avg'),
        ('network_payload_fraction','network_payload-length_avg','network_packet-size_avg'),
        ('network_packets_per_reported_ip','network_packets_all_count','network_ips_all_count'),
        ('network_packets_per_reported_port','network_packets_all_count','network_ports_all_count')):
        result[name]=ratio(col(numerator),col(denominator))
    return result


def group(name):
    if '__fraction' in name or '_packet_fraction' in name:return 'ratios'
    if '__per_second' in name:return 'rates'
    if 'coefficient' in name or 'per_reported' in name or 'payload_fraction' in name:return 'ratios'
    if 'tcp-flags-' in name:return 'flag_counts'
    for token in ('packet-size','ip-length','header-length','ttl','mss','window-size','tcp-flags','time-delta'):
        if token in name:return token
    for token in ('ips','macs','ports','packets'):
        if f'network_{token}_' in name:return token
    if 'fragmented-packets' in name:return 'fragmentation'
    return 'other'
