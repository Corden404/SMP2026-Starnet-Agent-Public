"""One V3 candidate: paired-world valuation, unchanged V2 scan schedule and caps.

Conditional on all32 old-state diagnosis, then a NEW complete54-cell development
cohort. No F1 combination; no tuning on confirmation; no paid calls in this CLI.
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
os.environ['ANONYMIZED_TELEMETRY']='False'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import round13_crossed as core
import round13_stress as stress
import round12_experiment as runner
from round4_seeds import generate as ordinary_generate
from round9_lock import ExclusiveBatch

ROOT=core.ROOT
OUT=ROOT/'agent_notes/experiments/round15-coupled'
VARIANTS=('S44','V3')
COUNTS=dict(dev=54,confirm=108,stress=54,integration=6)
read,save,sha,digest=core.read,core.save,core.sha,core.digest


def specs(pool):
    if pool=='stress':rows=copy.deepcopy(stress.specs())
    elif pool=='integration':rows=copy.deepcopy(core.specs('integration'))
    else:rows=copy.deepcopy(core.specs('main')[:COUNTS[pool]])
    for i,s in enumerate(rows):s.update(case=f'r15-{pool}-{i:03}',pool='r15-'+pool)
    assert len(rows)==COUNTS[pool]
    return rows


def generate(spec):
    return stress.generate(spec) if spec.get('stress_generator_version')==2 else ordinary_generate(spec)


def source_files(variant):
    if variant not in VARIANTS:raise ValueError(variant)
    files=core.source_files('S44')
    if variant=='V3':
        files['starnet_model.py']+=b'\n\n'+(ROOT/'workspace/round12_policy.py').read_bytes()+b'\n\n'+(ROOT/'workspace/round15_policy.py').read_bytes()
    compile(files['starnet_model.py'],'starnet_model.py','exec');return files


def fingerprint():
    return dict(core=core.fingerprint(),files={p:sha(ROOT/p) for p in
        ('workspace/round12_policy.py','workspace/round12_experiment.py','workspace/round13_stress.py',
         'workspace/round15_policy.py','workspace/round15_experiment.py','tests/test_round15.py')},
        packages={v:{p:hashlib.sha256(b).hexdigest() for p,b in source_files(v).items()} for v in VARIANTS})


def load(variant):
    name='round15_'+variant
    if name not in sys.modules:
        folder=OUT/'packages'/variant/'team_submission'
        for n,b in source_files(variant).items():
            p=folder/n
            if p.exists():
                if p.read_bytes()!=b:raise RuntimeError('Staged V3 drift')
            else:
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:f.write(b)
        spec=importlib.util.spec_from_file_location(name,folder/'starnet_model.py')
        m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name]


def prepare():
    diagnostic=read(ROOT/'agent_notes/experiments/round14-branches/summaries/scan.json')
    if not (diagnostic['states']==32 and diagnostic['sign_errors']>0 and diagnostic['positive_states']['total']>0):
        raise RuntimeError('No demonstrated value-estimation failure with beneficial scan opportunities')
    for v in VARIANTS:load(v)
    seeds={p:[dict(spec=s,sha256=digest(generate(s))) for s in specs(p)] for p in COUNTS}
    hashes=[r['sha256'] for es in seeds.values() for r in es]
    old_hashes={r['sha256'] for es in core.read(core.OUT/'protocol.json')['seeds'].values() for r in es}
    if len(set(hashes))!=len(hashes) or set(hashes)&old_hashes:raise RuntimeError('Seed collision')
    protocol=dict(code=fingerprint(),seeds=seeds,counts=COUNTS,baseline='S44',candidate='V3',
        rationale='All28 valid old scan states:13 sign errors,11 positive true suffix differences. Investigate valuation, not retune thresholds.',
        changes='Only valuation: four same latent worlds per branch; empirical whole(w,persona) donors; fixed response latent; exact S44 numerical replan after simulated receipts; no future response in selection.',
        unchanged='First44 scans, windows40/20, margin0.5,8s complete-estimate budget,max2 real extra scans,unknown-edge absence,real response estimator,prompts. No F1 mix.',
        limitation='Empirical four-world prior is not calibrated; assuming unknown-unknown edges absent remains. Computation incomplete => no purchase.',
        development='Fixed54 complete cells. Mean>0 and without-best-two>0 and nontriggered_different==0 => local confirmation; otherwise stop this implementation. Full confirm108+stress54 then positive means and CI lower needed for confirmation. No N extension.',
        controls='Equal information/resources identical branch=>zero; latent not passed to candidate selector; same world reused; real scan receipts only; other sizes bypass.',
        api='No paid requests in this study. Prior GLM6v SSL uncertain failure requires separate service recovery; never silently retries.',
        workers=2,no_upload=True,no_git_commit=True,no_tuning_after_freeze=True,
        diagnostic_sha256=sha(ROOT/'agent_notes/experiments/round14-branches/summaries/scan.json'))
    save(OUT/'protocol.json',protocol,True)
    print(json.dumps(dict(prepared=True,counts=COUNTS,originals=core.preserved())),flush=True)


def check():
    p=read(OUT/'protocol.json')
    if p['code']!=fingerprint():raise RuntimeError('V3 policy/evaluator frozen source drift')
    core.preserved();return p


def configure():
    runner.OUT=OUT;runner.VARIANTS=VARIANTS;runner.COUNTS=COUNTS
    runner.specs=specs;runner.generate=generate;runner.source_files=source_files
    runner.load=load;runner.check=check


def group(pool,index,mode):
    configure();return runner.group(pool,index,mode,None)


def batch(pool):
    check();mode='integration' if pool=='integration' else 'fast';n=COUNTS[pool]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        pending={};cursor=0;done_count=0
        for _ in range(2):
            f=ex.submit(group,pool,cursor,mode);pending[f]=cursor;cursor+=1
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                pending.pop(f)
                try:r=f.result()
                except BaseException:
                    for other in pending:other.cancel()
                    raise
                done_count+=1
                if done_count%6==0 or done_count==n:print(json.dumps(dict(pool=pool,completed=done_count,total=n,**r)),flush=True)
                if cursor<n:
                    new=ex.submit(group,pool,cursor,mode);pending[new]=cursor;cursor+=1
    configure();result=runner.summarize(pool,mode);result['candidate']='V3'
    result['originals']=core.preserved();save(OUT/'summaries'/(pool+'-'+mode+'.json'),result)
    if pool=='dev':
        gate=result['mean']>0 and result['without_best_two']>0 and result['nontriggered_different']==0
        save(OUT/'development-decision.json',dict(eligible=gate,status='V3_DEV_POSITIVE' if gate else 'KEEP_S44_V3_DEV_GATE_FAILED'),True)
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','dev','confirm','stress','integration','audit']);a=p.parse_args()
    with ExclusiveBatch(OUT/'execution.lock'):
        if a.stage=='prepare':prepare()
        elif a.stage=='audit':configure();runner.audit()
        else:
            if a.stage in ('confirm','stress') and not read(OUT/'development-decision.json')['eligible']:
                raise RuntimeError('V3 development gate failed')
            batch(a.stage)

if __name__=='__main__':main()
