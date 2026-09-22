"""Prospective crossed F1 follow-up: exact old policy, new independent cohorts.

No historical result is pooled. Only test design changes. All old source/evidence
is SHA-protected. Main=216 (54 cells x4); stress=54 (all cells x1).
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import contextlib
import copy
import hashlib
import importlib.util
import io
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
from scipy.stats import t as student_t
import round11_experiment as old
from round11_api import JournalLLM, MODELS, config as api_config, probe, APICreditExhausted, ServiceAbort
from round4_seeds import generate, FAMILIES, SCALES
from experiment_round5 import FiniteEnv, PublicEnv
from benchmark import LoggedEnv
from round9_lock import ExclusiveBatch

ROOT = old.ROOT
OUT = ROOT / 'agent_notes/experiments/round13-crossed'
VARIANTS = ('S44', 'F1')
COUNTS = dict(main=216, stress=54, glm=48, integration=6)
read, save, sha, digest = old.read, old.save, old.sha, old.digest
CELLS = tuple(itertools.product(range(6), range(3), range(3)))
DEPENDENCIES = tuple(old.DEPENDENCIES) + ('workspace/round13_crossed.py', 'tests/test_round13.py')


def preserved():
    original = read(OUT / 'original-workspace.json')
    drift = [p for p, h in original['files'].items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if drift or head != original['head'] or branch != original['branch'] or sha(old.ARCHIVE) != old.ARCHIVE_SHA:
        raise RuntimeError('Protected source/evidence drift: ' + repr(drift))
    return dict(status='PASS', protected_files=len(original['files']), head=head,
                branch=branch, s44_sha256=sha(old.ARCHIVE))


def specs(pool):
    if pool not in COUNTS:
        raise ValueError(pool)
    rows = []
    if pool in ('main', 'stress'):
        cells = [(rep, f, s, p) for rep in range(4 if pool == 'main' else 1) for f, s, p in CELLS]
    elif pool == 'glm':
        # Fixed balanced fractional design: 8 of9 cells per family; not called a full factorial.
        cells = [(0, f, s, p) for f, s, p in CELLS if 3 * s + p != (2 * f + 1) % 9]
    else:
        cells = [(0, f, f % 3, (f + 1) % 3) for f in range(6)]
    for index, (rep, f, s, p) in enumerate(cells):
        stress = pool == 'stress'
        mode = ('degree_w', 'degree_r', 'community', 'w_r')[(f + 3 * s + p) % 4] if stress else 'independent'
        # Density/OOD crossed within each family, never selected using outcomes.
        dense = stress and (f + s + 2 * p) % 2 == 0
        rows.append(dict(case=f'r13-{pool}-{index:03d}', pool='r13-' + pool,
                         family=FAMILIES[f], scale=SCALES[s], persona_mix=p, rep=rep,
                         mode=mode, n=50, ood=dense, generator_version=1))
    assert len(rows) == COUNTS[pool]
    return rows


def coverage(pool):
    rows = specs(pool)
    cells = collections.Counter((s['family'], s['scale'], s['persona_mix']) for s in rows)
    return dict(graphs=len(rows), cells=len(cells), counts=sorted(set(cells.values())),
                families=dict(collections.Counter(s['family'] for s in rows)),
                modes=dict(collections.Counter(s['mode'] for s in rows)),
                ood=dict(collections.Counter(str(s['ood']) for s in rows)))


def source_files(variant):
    # Do not rebuild or modify F1. Exact bytes from the already frozen round11 study.
    files = old.source_files(variant)
    frozen = ROOT / 'agent_notes/experiments/round11-feedback/packages' / variant / 'team_submission'
    for name, content in files.items():
        if (frozen / name).read_bytes() != content:
            raise RuntimeError('Frozen F1/S44 changed: ' + name)
    return files


def fingerprint():
    return dict(files={p: sha(ROOT / p) for p in DEPENDENCIES},
                packages={v: {p: hashlib.sha256(b).hexdigest() for p, b in source_files(v).items()} for v in VARIANTS},
                archive=old.ARCHIVE_SHA)


def load(variant):
    name = 'round13_exact_' + variant
    if name not in sys.modules:
        folder = OUT / 'packages' / variant / 'team_submission'
        for relative, content in source_files(variant).items():
            p = folder / relative
            if p.exists():
                if p.read_bytes() != content:
                    raise RuntimeError('Staged source drift')
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open('xb') as f:
                    f.write(content)
        spec = importlib.util.spec_from_file_location(name, folder / 'starnet_model.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def prepare():
    preserved()
    entries = {p: [dict(spec=s, sha256=digest(generate(s))) for s in specs(p)] for p in COUNTS}
    hashes = [r['sha256'] for rs in entries.values() for r in rs]
    previous = set()
    for directory in ('round9', 'round10-s42', 'round11-feedback', 'round12-active-scan'):
        data = read(ROOT / 'agent_notes/experiments' / directory / 'protocol.json')
        for rs in data.get('seeds', {}).values():
            previous.update(r['sha256'] for r in rs)
    if len(hashes) != len(set(hashes)) or set(hashes) & previous:
        raise RuntimeError('Seed collision with prior study')
    for v in VARIANTS:
        load(v)
    protocol = dict(baseline='S44', candidate='F1', policy_change='NONE: byte-identical round11 F1',
        rationale='User-approved new independent follow-up after sparse development and incomplete factorial coverage. Old gate stays failed.',
        counts=COUNTS, seeds=entries, code=fingerprint(), coverage={p:coverage(p) for p in COUNTS},
        design='Main54 cells x4 independent graphs; stress54 cells x1. Stress modes/density partially balanced, not a full mode factorial. No old data pooled.',
        primary='Equally weighted cell/graph mean candidate minus S44. 95% paired t and family-stratified bootstrap; cell bootstrap also descriptive. Main confirmation needs BOTH t and family-bootstrap lower>0.',
        gates='Always complete main216 AND stress54. Mean main>0 AND stress>=0 => eligible for fixed48 real-LLM pairs; positive intervals crossing0 only exploratory. Otherwise stop F1. without-best-two is descriptive, not veto. No result-dependent N extension.',
        real_gate='48 NEW graphs, one full real-LLM game per strategy/graph, no cross-model pooling. Recommend official exploratory trial only if real mean>0 and stress nonnegative plus actual package acceptance; confirmed additionally needs positive lower intervals.',
        workers=2, api_order=list(MODELS), api_switch='Credit failure only. Whole same-model cohorts. Named model unavailable is logged and next authorized fallback may be tried. No uncertain request retries.',
        source_protection=True, no_upload=True, no_git_commit=True, no_strategy_tuning=True,
        diagnostic_followup='Separate all32 V2 states and all73 F1 opportunities; these old graphs never enter confirmation. V3 only after diagnosing reproducible action-value discrepancy; separate frozen protocol required.')
    save(OUT / 'protocol.json', protocol, True)
    print(json.dumps(dict(prepared=True, coverage=protocol['coverage'], originals=preserved())), flush=True)


def check():
    p = read(OUT / 'protocol.json')
    if p['code'] != fingerprint():
        raise RuntimeError('Frozen evaluator/policy changed')
    preserved()
    return p


def receipt_equal(a, b):
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    for x in (a, b):
        if isinstance(x, dict) and 'neighbors' in x:
            x['neighbors'] = sorted(x['neighbors'])
    return a == b


def action_bag(row):
    return collections.Counter((a['action'], tuple(a['args'])) for a in row['actions'] if a['action'] != 'scan_node')


def folder(pool, mode='fast', model=None):
    return OUT / 'runs' / (pool + '-' + mode + ('-' + model if model else ''))


def run_one(pool, index, variant, mode='fast', api_model=None):
    p = check()
    entry = p['seeds'][pool][index]
    spec = entry['spec']; seed = generate(spec)
    if digest(seed) != entry['sha256']:
        raise RuntimeError('Seed hash changed')
    target = folder(pool, mode, api_model)
    path = target / (spec['case'] + '-' + variant + '.json')
    stop = target / 'STOP.json'
    identity = dict(seed_sha256=entry['sha256'], spec=spec, variant=variant, mode=mode,
                    code=p['code'], model_config=api_config(api_model) if api_model else None)
    if path.exists():
        result = read(path)
        if result.get('identity') != identity or result.get('status') != 'PASS':
            raise RuntimeError('Incomplete/different existing record: ' + str(path))
        return result
    if stop.exists():
        raise ServiceAbort('Prior failure requires inspection')
    row = dict(identity=identity, case=spec['case'], spec=spec, variant=variant, status='STARTED')
    save(path, row)
    raw = FiniteEnv(seed); env = LoggedEnv(raw)
    module = load(variant); module.coefficients.cache_clear()
    llm = model = None; chains = []; started = time.monotonic()
    try:
        with contextlib.ExitStack() as stack:
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if mode != 'glm':
                for name in ('socket.socket.connect', 'socket.socket.connect_ex', 'socket.create_connection'):
                    stack.enter_context(patch(name, old.denied))
            llm = JournalLLM(api_model, path.with_suffix('.http.jsonl'), stop) if mode == 'glm' else old.TopChoiceLLM()
            model = old.initialize(module, env, mode=mode, llm=llm)
            for role in ('scout', 'commander'):
                agent = getattr(model, role); original = agent._run_chain
                def trace(name, data, original=original, role=role):
                    before = copy.deepcopy(data); answer = original(name, data)
                    chains.append(dict(role=role, chain=name, operation=model.operation_count,
                                       input=before, output=copy.deepcopy(answer), real_llm=mode == 'glm'))
                    return answer
                agent._run_chain = trace
            for _ in range(120):
                if model.step():
                    break
            else:
                raise RuntimeError('Resource fuse reached')
        if not (model.operation_count == len(env.actions) < 120 and llm.http_request_count <= model.llm_counter.call_count < 120):
            raise RuntimeError('Resource accounting mismatch')
        used = sum(old.COSTS[a['action']] for a in env.actions)
        if abs(100 - used - raw.get_remaining_budget()) > 1e-8 or len(raw._scanned) != 44:
            raise RuntimeError('Budget/scan mismatch')
        row.update(status='PASS', final_score=raw.evaluate(), remaining_budget=raw.get_remaining_budget(),
            operations=model.operation_count, logical_calls=model.llm_counter.call_count, http_calls=llm.http_request_count,
            elapsed_seconds=time.monotonic()-started, actions=env.actions, chains=chains,
            feedback_records=getattr(model, 'feedback_records', []), search_records=model.search_records,
            stop_reason=model.stop_reason, metadata=getattr(llm, 'response_metadata', []),
            action_counts=dict(collections.Counter(a['action'] for a in env.actions)))
        save(path, row)
    except BaseException as exc:
        row.update(status='FAILED', error_type=type(exc).__name__, actions=env.actions, chains=chains,
                   http_calls=getattr(llm,'http_request_count',0), elapsed_seconds=time.monotonic()-started)
        save(path,row); save(stop,dict(case=spec['case'],variant=variant,error_type=type(exc).__name__,no_retry=True))
        raise
    finally:
        module.coefficients.cache_clear()
    return row


def group(pool, index, mode, api_model):
    spec = specs(pool)[index]
    # Main: each cell's four replicates have exactly two of each execution order.
    cell = index % 54
    order = VARIANTS if (spec['rep'] + cell) % 2 == 0 else VARIANTS[::-1]
    rows = [run_one(pool,index,v,mode,api_model) for v in order]
    return dict(case=spec['case'], scores={r['variant']:r['final_score'] for r in rows})


def summarize(pool, mode='fast', api_model=None):
    p = check(); target = folder(pool,mode,api_model)
    pairs=[]; rows=[]
    for entry in p['seeds'][pool]:
        spec=entry['spec']; b,c=[read(target/(spec['case']+'-'+v+'.json')) for v in VARIANTS]
        for r,v in ((b,'S44'),(c,'F1')):
            if r['status']!='PASS' or r['identity']['code']!=p['code'] or r['identity']['seed_sha256']!=entry['sha256'] or r['variant']!=v:
                raise RuntimeError('Invalid complete pair')
            if r['identity']['model_config'] != (api_config(api_model) if api_model else None):
                raise RuntimeError('Mixed API model/config')
        rows.extend((b,c))
        pairs.append(dict(case=spec['case'],family=spec['family'],scale=spec['scale'],persona_mix=spec['persona_mix'],
            delta=c['final_score']-b['final_score'],same_action_multiset=action_bag(b)==action_bag(c),
            proposed_pairs=sum(e['event']=='proposed' for e in c['feedback_records']),
            replacements=sum(bool(e.get('replaced')) for e in c['feedback_records'])))
    ds=np.array([r['delta'] for r in pairs]); n=len(ds); rng=np.random.default_rng(13092026)
    boot=np.zeros(10000)
    for family in FAMILIES:
        xs=np.array([r['delta'] for r in pairs if r['family']==family])
        boot += rng.choice(xs,(len(boot),len(xs)),replace=True).sum(axis=1)/n
    sd=float(np.std(ds,ddof=1)); half=float(student_t.ppf(.975,n-1))*sd/math.sqrt(n)
    interval=[float(ds.mean()-half),float(ds.mean()+half)]
    result=dict(pool=pool,mode=mode,api_model=api_model,n=n,games=2*n,mean=float(ds.mean()),sd=sd,
        ci95_t=interval,ci95_family_bootstrap=np.quantile(boot,[.025,.975]).tolist(),
        wins=int(sum(ds>1e-8)),ties=int(sum(abs(ds)<=1e-8)),losses=int(sum(ds< -1e-8)),
        without_best_two=float(np.mean(np.sort(ds)[:-2])),min=float(ds.min()),max=float(ds.max()),
        worst_decile=float(np.mean(np.sort(ds)[:max(1,math.ceil(.1*n))])),
        action_multiset_changes=sum(not x['same_action_multiset'] for x in pairs),
        proposed_pairs=sum(x['proposed_pairs'] for x in pairs),replacements=sum(x['replacements'] for x in pairs),
        families={f:float(np.mean([x['delta'] for x in pairs if x['family']==f])) for f in FAMILIES},
        max_operations=max(r['operations'] for r in rows),max_calls=max(r['logical_calls'] for r in rows),
        http_calls=sum(r['http_calls'] for r in rows),max_seconds=max(r['elapsed_seconds'] for r in rows),
        truncations={v:sum(bool(s['plans'][0].get('search_truncated')) for r in rows if r['variant']==v for s in r['search_records'] if s['plans']) for v in VARIANTS},
        scope='Custom synthetic graph population only; not Tianchi hidden score. Fixed-cell stress one replicate: intervals descriptive.',pairs=pairs)
    if pool=='main':
        cell_boot=np.zeros(10000)
        for f,s,person in CELLS:
            xs=np.array([r['delta'] for r in pairs if (r['family'],r['scale'],r['persona_mix'])==(FAMILIES[f],SCALES[s],person)])
            assert len(xs)==4
            cell_boot += rng.choice(xs,(len(cell_boot),4),replace=True).sum(axis=1)/n
        result['ci95_cell_bootstrap']=np.quantile(cell_boot,[.025,.975]).tolist()
    save(OUT/'summaries'/(pool+'-'+mode+('-'+api_model if api_model else '')+'.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'}),flush=True)
    return result


def batch(pool,mode='fast',api_model=None,start=0,stop=None):
    check(); stop=COUNTS[pool] if stop is None else stop
    if not 0<=start<stop<=COUNTS[pool]: raise ValueError('Invalid fixed cohort range')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        pending={}; cursor=start; count=0
        while cursor<min(stop,start+2):
            f=ex.submit(group,pool,cursor,mode,api_model);pending[f]=cursor;cursor+=1
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                pending.pop(f)
                try: answer=f.result()
                except BaseException:
                    for other in pending:other.cancel()
                    raise
                count+=1
                if count%6==0 or count==stop-start or mode=='glm':
                    print(json.dumps(dict(pool=pool,completed_in_range=count,range=[start,stop],**answer)),flush=True)
                if cursor<stop:
                    new=ex.submit(group,pool,cursor,mode,api_model);pending[new]=cursor;cursor+=1
    if all((folder(pool,mode,api_model)/(s['case']+'-'+v+'.json')).exists() for s in specs(pool) for v in VARIANTS):
        return summarize(pool,mode,api_model)
    return dict(status='FIXED_BLOCK_COMPLETE',pool=pool,start=start,stop=stop)


def decision():
    c=read(OUT/'summaries/main-fast.json');s=read(OUT/'summaries/stress-fast.json')
    assert c['n']==216 and s['n']==54
    eligible=c['mean']>0 and s['mean']>=0
    confirmed=eligible and min(c['ci95_t'][0],c['ci95_family_bootstrap'][0])>0
    result=dict(status='LOCAL_CONFIRMED' if confirmed else 'LOCAL_POSITIVE_INCONCLUSIVE' if eligible else 'KEEP_S44_LOCAL_GATE_FAILED',
        eligible_for_real=eligible,local_confirmed=confirmed,main_mean=c['mean'],stress_mean=s['mean'],
        counts=COUNTS,no_old_data_pooled=True,no_sample_extension=True,originals=preserved())
    save(OUT/'local-decision.json',result,True);print(json.dumps(result),flush=True)
    return result


def audit():
    check(); count=0; max_error=0.;incomplete=[]
    for path in sorted((OUT/'runs').glob('*/*.json')):
        row=read(path)
        if row.get('status')!='PASS':incomplete.append(str(path.relative_to(OUT)));continue
        raw=FiniteEnv(generate(row['spec']))
        for a in row['actions']:
            if not receipt_equal(getattr(raw,a['action'])(*a['args']),a['result']):
                raise RuntimeError('Receipt drift '+str(path))
        error=abs(raw.evaluate()-row['final_score']);max_error=max(max_error,error)
        if error>1e-8 or raw.get_remaining_budget()!=row['remaining_budget'] or len(row['actions'])!=row['operations']:
            raise RuntimeError('Audit score/budget mismatch')
        count+=1
    result=dict(status='PASS' if not incomplete else 'INCOMPLETE',games=count,max_forward_error=max_error,
                incomplete=incomplete,originals=preserved())
    save(OUT/'audit.json',result);print(json.dumps(result),flush=True);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','main','stress','integration','decision','glm','audit'])
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int)
    parser.add_argument('--model',choices=MODELS)
    args=parser.parse_args()
    with ExclusiveBatch(OUT/'execution.lock'):
        if args.stage=='prepare':prepare()
        elif args.stage=='decision':decision()
        elif args.stage=='audit':audit()
        elif args.stage=='glm':
            if not read(OUT/'local-decision.json')['eligible_for_real']:raise RuntimeError('Local gate failed; no paid calls')
            model=args.model or MODELS[0]
            if model!=MODELS[0]:
                attempts=read(OUT/'api/fallback-authority.json')
                if model not in attempts['allowed_models']:raise RuntimeError('Model switch lacks observed failure evidence')
            probe(model,OUT);batch('glm','glm',model,args.start,args.stop)
        else:batch(args.stage,'integration' if args.stage=='integration' else 'fast',start=args.start,stop=args.stop)


if __name__=='__main__':main()
