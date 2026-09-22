"""Frozen PC2 study after complete diagnosis; GC2 is an explicit mechanism control."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import copy
import hashlib
import importlib.util
import json
import sys
from unittest.mock import patch
import round16_common as c
from round9_lock import ExclusiveBatch

OUT=c.OUT/'pc2'
VARIANTS=('S44','GC2','PC2')
COUNTS={k:v for k,v in c.COUNTS.items() if k!='diagnostic'}
DEPENDENCIES=('workspace/round16_policy.py','workspace/round16_experiment.py','tests/test_round16_policy.py')


def source_files(variant):
    if variant not in VARIANTS:raise ValueError(variant)
    files=c.source_files('S44' if variant=='S44' else 'PC2')
    if variant=='GC2':
        files['starnet_model.py']+=b'\n\nclass ParticipantSquadModel(ParticipantSquadModel):\n    pc2_depth = 0\n'
    compile(files['starnet_model.py'],'starnet_model.py','exec')
    return files


def fingerprint():
    return dict(core=c.fingerprint(DEPENDENCIES),packages={v:{n:hashlib.sha256(b).hexdigest() for n,b in source_files(v).items()} for v in VARIANTS})


def load(variant):
    name='round16_pc2_'+variant
    if name not in sys.modules:
        folder=OUT/'packages'/variant/'team_submission'
        for name_,content in source_files(variant).items():
            p=folder/name_
            if p.exists():
                if p.read_bytes()!=content:raise RuntimeError('PC2 staged source changed')
            else:
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:f.write(content)
        spec=importlib.util.spec_from_file_location(name,folder/'starnet_model.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def prepare():
    diagnosis=c.read(c.OUT/'diagnostic/summary.json')
    if diagnosis['status']!='PASS' or diagnosis['graphs']!=54 or diagnosis['next_branch']!='PC2':
        raise RuntimeError('Complete diagnostic gate does not support PC2')
    c.preserved()
    seeds={pool:[dict(spec=s,sha256=c.digest(c.generate(s))) for s in c.specs(pool)] for pool in COUNTS}
    all_hashes=[e['sha256'] for es in seeds.values() for e in es]
    diagnostic_hashes={e['sha256'] for e in c.read(c.OUT/'diagnostic/protocol.json')['seeds']}
    if len(set(all_hashes))!=len(all_hashes) or set(all_hashes)&diagnostic_hashes:
        raise RuntimeError('New development/confirmation must be independent')
    protocol=dict(code=fingerprint(),counts=COUNTS,seeds=seeds,baseline='S44',candidate='PC2',control='GC2',
        diagnosis_sha256=c.sha(c.OUT/'diagnostic/summary.json'),
        candidate_description='S44 root pair, same4 fixed windows as diagnosis,3 antithetic public-response worlds,2 feedback S44 numerical replans then feedback-greedy continuation. Complete6 branch values within3s or fallback. Margin0.5. No scan/response-prior changes.',
        control_description='GC2 identical worlds, root pair, windows, budgets and scorer; greedy feedback continuation without two S44 numerical replans. Not automatically promoted as a candidate.',
        computational_scope='Optional NumPy finite50 acceleration ONLY inside hypothetical branches; real S44 base unchanged. Floating-point tolerance tested; deadlines/approximate posterior remain limitations.',
        development_gate='Full new54 triples. PC2-S44 mean>0,>=3 wins across>=2 families,nontriggered_different==0,first44 scan prefixes identical,resources pass. without-best-two and negative tail descriptive, not veto.',
        confirm_gate='If development passes, complete fixed216 S44/PC2 pairs plus54 stress pairs. Both means positive/nonnegative required for real validation. 95%paired-t AND family-bootstrap lower>0 for confirmed local gain. No N extension, no tuning.',
        estimand='Graph-level mean delta candidate-S44; GC2 is a mechanism control. All data synthetic, not leaderboard score.',
        workers=2,api=False,upload=False,git_commit=False)
    for v in VARIANTS:load(v)
    c.save(OUT/'protocol.json',protocol,True)
    print(json.dumps(dict(prepared=True,counts=COUNTS,originals=c.preserved())),flush=True)


def check():
    p=c.read(OUT/'protocol.json')
    if p['code']!=fingerprint():raise RuntimeError('Frozen PC2 source changed')
    if p['diagnosis_sha256']!=c.sha(c.OUT/'diagnostic/summary.json'):raise RuntimeError('Diagnostic evidence changed')
    return p


def variants(pool):
    return VARIANTS if pool in ('dev','integration') else ('S44','PC2')


def run_one(pool,index,variant):
    p=check();entry=p['seeds'][pool][index];spec=entry['spec']
    identity=dict(entry=entry,variant=variant,code=p['code'],mode='integration' if pool=='integration' else 'fast')
    path=OUT/'runs'/pool/(spec['case']+'-'+variant+'.json')
    if path.exists():
        row=c.read(path)
        if row.get('status')!='PASS' or row.get('identity')!=identity:raise RuntimeError('Incomplete/different PC2 result; inspect before retry')
        return row
    if (OUT/'STOP.json').exists():raise RuntimeError('Earlier PC2 execution failed')
    c.save(path,dict(status='STARTED',identity=identity))
    try:
        with patch.object(c,'load',load):
            row=c.run_game(spec,variant,mode=identity['mode'])
        row['identity']=identity;c.save(path,row)
    except BaseException as exc:
        c.save(path,dict(status='FAILED',identity=identity,error_type=type(exc).__name__))
        c.save(OUT/'STOP.json',dict(case=spec['case'],variant=variant,error_type=type(exc).__name__))
        raise
    return row


def group(pool,index):
    order=variants(pool)
    if (index//54+index%54)%2:order=order[::-1]
    rows=[run_one(pool,index,v) for v in order]
    return dict(case=rows[0]['spec']['case'],seconds=sum(r['elapsed_seconds'] for r in rows))


def summarize(pool):
    p=check();rows={}
    for entry in p['seeds'][pool]:
        for v in variants(pool):
            row=c.read(OUT/'runs'/pool/(entry['spec']['case']+'-'+v+'.json'))
            if row['status']!='PASS' or row['identity']['code']!=p['code'] or row['identity']['entry']!=entry:
                raise RuntimeError('Incomplete/mixed PC2 cohort')
            rows[entry['spec']['case'],v]=row
    comparisons={}
    for a,b in [('PC2','S44')]+([('GC2','S44'),('PC2','GC2')] if pool in ('dev','integration') else []):
        pairs=[];nontriggered=0;prefix_changes=0;replacements=0;reasons=collections.Counter()
        for entry in p['seeds'][pool]:
            spec=entry['spec'];x,y=rows[spec['case'],a],rows[spec['case'],b]
            records=x['candidate_records'];changed=any(r.get('replaced') for r in records)
            if not changed and (abs(x['final_score']-y['final_score'])>1e-8 or c.bag(x['actions'])!=c.bag(y['actions'])):
                nontriggered+=1
            prefix_changes+=int([r['args'][0] for r in x['actions'] if r['action']=='scan_node']!=[r['args'][0] for r in y['actions'] if r['action']=='scan_node'])
            replacements+=sum(bool(r.get('replaced')) for r in records)
            reasons.update(r['reason'] for r in records)
            pairs.append(dict(case=spec['case'],family=spec['family'],delta=x['final_score']-y['final_score'],changed=changed))
        stats=c.paired_stats(pairs)
        stats.update(pairs=pairs,nontriggered_different=nontriggered,scan_prefix_differences=prefix_changes,
            changed_graphs=sum(r['changed'] for r in pairs),replacements=replacements,reasons=dict(reasons),
            positive_families=len({r['family'] for r in pairs if r['delta']>1e-8}))
        comparisons[a+'-'+b]=stats
    result=dict(pool=pool,graphs=COUNTS[pool],games=len(rows),comparisons=comparisons,
        max_operations=max(r['operations'] for r in rows.values()),max_calls=max(r['logical_calls'] for r in rows.values()),
        max_seconds=max(r['elapsed_seconds'] for r in rows.values()),http_calls=0,originals=c.preserved())
    c.save(OUT/'summaries'/(pool+'.json'),result,True)
    if pool=='dev':
        a=comparisons['PC2-S44']
        eligible=a['mean']>0 and a['wins']>=3 and a['positive_families']>=2 and a['nontriggered_different']==0 and a['scan_prefix_differences']==0
        c.save(OUT/'development-decision.json',dict(eligible=eligible,status='ADVANCE_PC2' if eligible else 'KEEP_S44_PC2_DEV_GATE_FAILED',
            mean=a['mean'],control_delta=comparisons['PC2-GC2']['mean'],no_retuning=True),True)
    print(json.dumps(dict(pool=pool,graphs=COUNTS[pool],comparisons={k:{a:b for a,b in v.items() if a!='pairs'} for k,v in comparisons.items()})),flush=True)
    return result


def batch(pool,start=0,stop=None):
    p=check();c.preserved();stop=COUNTS[pool] if stop is None else stop
    if not 0<=start<stop<=COUNTS[pool]:raise ValueError('Invalid cohort block')
    if pool in ('confirm','stress') and not c.read(OUT/'development-decision.json')['eligible']:
        raise RuntimeError('PC2 development gate failed')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        cursor=start;pending={};count=0
        while cursor<min(stop,start+2):
            f=executor.submit(group,pool,cursor);pending[f]=cursor;cursor+=1
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                pending.pop(f);answer=f.result();count+=1
                print(json.dumps(dict(pool=pool,completed=count,range=[start,stop],**answer)),flush=True)
                if cursor<stop:
                    f=executor.submit(group,pool,cursor);pending[f]=cursor;cursor+=1
    if all((OUT/'runs'/pool/(s['case']+'-'+v+'.json')).exists() for s in c.specs(pool) for v in variants(pool)):
        return summarize(pool)
    return dict(status='FIXED_BLOCK_COMPLETE',range=[start,stop])


def audit():
    check();games=0;max_error=0.;incomplete=[]
    for path in sorted((OUT/'runs').glob('*/*.json')):
        row=c.read(path)
        if row['status']!='PASS':incomplete.append(str(path.relative_to(OUT)));continue
        max_error=max(max_error,c.audit_game(row));games+=1
    result=dict(status='PASS' if not incomplete else 'INCOMPLETE',games=games,max_forward_error=max_error,incomplete=incomplete,originals=c.preserved())
    c.save(OUT/'audit.json',result);print(json.dumps(result),flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','dev','confirm','stress','integration','audit'])
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int);a=p.parse_args()
    with ExclusiveBatch(OUT/'execution.lock'):
        if a.stage=='prepare':prepare()
        elif a.stage=='audit':audit()
        else:batch(a.stage,a.start,a.stop)

if __name__=='__main__':main()
