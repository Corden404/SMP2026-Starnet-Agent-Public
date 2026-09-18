"""Prospectively registered public-state experiments; no external calls."""
from __future__ import annotations
import argparse
import concurrent.futures
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import socket
import time

from benchmark import LoggedEnv
from experiment_round3 import fast_model, TopChoiceLLM
from experiment_round5 import FiniteEnv, PublicEnv
from round4_seeds import generate, FAMILIES
from round6_common import ROOT, D5_ARCHIVE, D5_SOURCE, digest, file_hash, load_d5_module, people, save

OUT = ROOT / 'agent_notes/experiments/round7'

def specifications(pool):
    count = {'pilot': 48, 'confirm': 192, 'stress': 48}[pool]
    result = []
    for index in range(count):
        f, rep = index % 6, index // 6
        result.append(dict(case=f'r7-{pool}-{index:03}', pool='r7-'+pool,
            family=FAMILIES[f], scale=(.55, 1., 1.5)[(rep+f)%3],
            persona_mix=(rep//3+f)%3, rep=rep, mode=(
                ('degree_w','degree_r','community','w_r')[rep%4] if pool=='stress' else 'independent'),
            n=50, ood=pool=='stress', generator_version=1))
    return result

def prepare():
    protocol = dict(schema=1, baseline_sha256=file_hash(D5_ARCHIVE),
        variants=['D5','U3','X1','S48','S49'],
        U3='Frozen round6 U3 parameters; no retuning against round7.',
        X1='Whole-budget structural add/drop/swap with optimal fixed-mean communication allocation; fixed limits before pilot.',
        scan='Persistent scan stop; unseen nodes retained in a public-data mean completion. Diagnostic, not oracle.',
        selection={'pilot':48,'confirm':192,'stress':48,
            'pilot_gate':'paired mean > 0, at least 3 positive cases in at least 2 families, mean after removing best case > 0, worst-decile mean >= -5',
            'confirm_gate':'paired bootstrap 95% lower bound > 0; worst-decile mean >= -5; no resource failures; confirm unchanged candidate once',
            'stress_gate':'paired mean >= 0; worst-decile mean >= -10; no resource failures',
            'submission':'Only a candidate passing all gates gets packaged and real-service validation; no leaderboard submission automatically.'},
        prior_pools='No old sealed final pools read', network='socket connections blocked in every local worker',
        seeds={p:[{'spec':s,'sha256':digest(generate(s))} for s in specifications(p)] for p in ('pilot','confirm','stress')})
    path=OUT/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != protocol: raise RuntimeError('Protocol drift')
    else: save(path,protocol)
    return protocol

def classes(variant):
    if variant=='D5': return load_d5_module().ParticipantSquadModel
    if variant=='U3':
        from round6_feedback import U3
        return U3
    if variant=='X1':
        from round7_exchange import X1
        return X1
    from round7_scan import CLASSES
    return CLASSES[variant]

def initialize(model):
    model.search_records=[]
    model.feedback_records=[]
    model.exchange_records=[]
    model.scan_diagnostic_records=[]
    model.scan_stop_reached=False
    return model

def source_hashes(variant):
    names=['experiment_round7.py','round6_common.py','round4_seeds.py',
        'experiment_round3.py','experiment_round5.py','benchmark.py','round6_feedback.py']
    if variant=='X1':names.append('round7_exchange.py')
    if variant.startswith('S'):names.append('round7_scan.py')
    return {name:file_hash(ROOT/'workspace'/name) for name in names} | {'frozen_d5':file_hash(D5_SOURCE)}

def run_one(spec,variant,integration=False):
    network_attempts=[]
    def deny(*args,**kwargs):
        network_attempts.append('blocked')
        raise RuntimeError('Network disabled during local experiment')
    socket.socket.connect=deny
    socket.socket.connect_ex=deny
    socket.create_connection=deny
    seed=generate(spec)
    identity=dict(spec=spec,variant=variant,seed_sha256=digest(seed),
                  code=source_hashes(variant),integration=integration)
    target=OUT/('integration' if integration else spec['pool'])/(spec['case']+'-'+variant+'.json')
    if target.exists():
        old=json.loads(target.read_text(encoding='utf-8'))
        if old['identity']!=identity:raise RuntimeError('Cache identity mismatch: '+str(target))
        return {k:old[k] for k in ('case','variant','final_score','elapsed_seconds')}
    raw=FiniteEnv(seed);logged=LoggedEnv(raw);cls=classes(variant)
    started=time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        model=initialize(cls(PublicEnv(logged),copy.deepcopy(people()),TopChoiceLLM()) if integration
                         else fast_model(PublicEnv(logged),copy.deepcopy(people()),cls))
        for _ in range(seed['global_setting']['max_api_calls']):
            if model.step():break
        else:raise RuntimeError('Operation fuse reached')
    result=dict(identity=identity,case=spec['case'],variant=variant,spec=spec,
        final_score=raw.evaluate(),elapsed_seconds=time.monotonic()-started,
        remaining_budget=raw.get_remaining_budget(),stop_reason=model.stop_reason,
        logical_calls=model.llm_counter.call_count,operation_count=model.operation_count,
        actions=logged.actions,search_records=model.search_records,
        feedback_records=model.feedback_records,exchange_records=model.exchange_records,
        scan_diagnostic_records=model.scan_diagnostic_records,network_attempts=len(network_attempts))
    if network_attempts:raise RuntimeError('Unexpected network attempt')
    save(target,result)
    return {k:result[k] for k in ('case','variant','final_score','elapsed_seconds')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pool',choices=['pilot','confirm','stress'],default='pilot')
    ap.add_argument('--variants',nargs='+',default=['D5','U3']);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--limit',type=int);ap.add_argument('--integration',action='store_true')
    args=ap.parse_args();prepare()
    specs=specifications(args.pool)[:args.limit]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(run_one,s,v,args.integration) for s in specs for v in args.variants]
        for f in concurrent.futures.as_completed(futures):print(json.dumps(f.result()),flush=True)

if __name__=='__main__':main()
