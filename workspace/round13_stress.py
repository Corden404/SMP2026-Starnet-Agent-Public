"""Before-outcome stress-design correction; main216 and F1 remain byte-identical.

Original54 stress inputs were never run. This separate protocol removes the
within-mode persona/density alias, and decouples w/r reversal from density.
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import copy
import json
import random
from pathlib import Path
import sys
import round13_crossed as core
from round4_seeds import generate as original_generate,FAMILIES
from round9_lock import ExclusiveBatch

ROOT=core.ROOT
OUT=ROOT/'agent_notes/experiments/round13-balanced-stress'
MAIN_OUT=core.OUT
ORIGINAL_SPECS=core.specs
ORIGINAL_FINGERPRINT=core.fingerprint
ORIGINAL_PRESERVED=core.preserved
read,save,sha,digest=core.read,core.save,core.sha,core.digest


def specs():
    rows=copy.deepcopy(ORIGINAL_SPECS('stress'))
    # Balance separately inside each mode x persona subset (4 or5 cases each).
    # Assignment depends only on fixed design identity, never on outcomes.
    for mode_index,mode in enumerate(('degree_w','degree_r','community','w_r')):
        for persona in range(3):
            group=[s for s in rows if s['mode']==mode and s['persona_mix']==persona]
            rng=random.Random(13002026+mode_index*17+persona*131)
            rng.shuffle(group)
            for j,s in enumerate(group):
                s['ood']=(j+mode_index+persona)%2==0
                # Four joint density/reversal combinations in the first4 members.
                s['stress_reverse']=(j//2+mode_index+persona)%2==0
    for i,s in enumerate(rows):
        s['case']=f'r13-balanced-stress-{i:03d}';s['pool']='r13-balanced-stress'
        s['stress_generator_version']=2
    return rows


def generate(spec):
    seed=original_generate(spec)
    if spec.get('stress_generator_version')!=2:return seed
    if spec['mode'] in ('degree_w','degree_r','w_r'):
        degree=collections.Counter(i for edge in seed['edges'] for i in edge)
        for persona in ('和平','中立','暴力'):
            nodes=[n for n in seed['nodes'] if n['persona']==persona]
            field='w' if spec['mode']=='degree_w' else 'r'
            values=sorted(n[field] for n in nodes)
            order=sorted(nodes,key=lambda n:((n['w'] if spec['mode']=='w_r' else degree[n['id']]),n['id']),
                         reverse=spec['stress_reverse'])
            for n,value in zip(order,values):n[field]=value
    # Node-value multisets and therefore original_total are unchanged.
    return seed


def fingerprint():
    return dict(core=ORIGINAL_FINGERPRINT(),adapter={p:sha(ROOT/p) for p in
        ('workspace/round13_stress.py','tests/test_round13_stress.py')})


def prepare():
    ORIGINAL_PRESERVED()
    if list((MAIN_OUT/'runs/stress-fast').glob('*.json')):
        raise RuntimeError('Old stress has already been observed; cannot silently replace it')
    rows=specs();entries=[dict(spec=s,sha256=digest(generate(s))) for s in rows]
    groups={m:{str(p):dict(collections.Counter(str(s['ood']) for s in rows if s['mode']==m and s['persona_mix']==p)) for p in range(3)}
            for m in ('degree_w','degree_r','community','w_r')}
    for mode in groups.values():
        if any(set(c)!={'False','True'} for c in mode.values()):raise RuntimeError('Density alias remains')
    for mode in ('degree_w','degree_r','w_r'):
        for p in range(3):
            if len({(s['ood'],s['stress_reverse']) for s in rows if s['mode']==mode and s['persona_mix']==p})!=4:
                raise RuntimeError('Reversal still aliases density/persona')
    original=read(MAIN_OUT/'protocol.json')
    hashes={e['sha256'] for es in original['seeds'].values() for e in es}
    if any(e['sha256'] in hashes for e in entries):raise RuntimeError('Old seed collision')
    save(OUT/'original-workspace.json',read(MAIN_OUT/'original-workspace.json'),True)
    p=dict(seeds={'stress':entries},code=fingerprint(),counts={'stress':54},
        supersedes=dict(protocol_sha256=sha(MAIN_OUT/'protocol.json'),pool='stress',old_games_started=0),
        reason='Before stress outcomes: formula aliased OOD with persona within each mode. Correct only stress design, keep main216 sources/inputs/analysis frozen.',
        full_cells=54,mode_persona_ood=groups,workers=2,
        reversal='Within-persona ranks flipped independently of graph density. Test-only generator, no hidden fields enter participant.',
        remaining_limit='54 graphs cannot fully cross every graph/response/persona/mode/density/reversal combination. Core54 cells complete; nuisance factors balanced within mode/persona.',
        gates='Original main216/stress54 mean gates and API plan unchanged. No old stress scores exist or are pooled.')
    save(OUT/'protocol.json',p,True)
    print(json.dumps(dict(prepared=True,counts=p['counts'],density=groups,old_stress_games=0)),flush=True)


def check():
    p=read(OUT/'protocol.json')
    if p['code']!=fingerprint():raise RuntimeError('Balanced stress source drift')
    # ORIGINAL_PRESERVED uses core.OUT dynamically; both snapshots are identical.
    ORIGINAL_PRESERVED()
    return p


def configure():
    core.OUT=OUT
    core.generate=generate
    core.specs=lambda pool:specs() if pool=='stress' else ORIGINAL_SPECS(pool)
    core.check=check


def group(index):
    configure();return core.group('stress',index,'fast',None)


def batch():
    check()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        pending={};cursor=0;count=0
        for _ in range(2):
            f=ex.submit(group,cursor);pending[f]=cursor;cursor+=1
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                pending.pop(f)
                try:r=f.result()
                except BaseException:
                    for other in pending:other.cancel()
                    raise
                count+=1
                if count%6==0 or count==54:print(json.dumps(dict(completed=count,total=54,**r)),flush=True)
                if cursor<54:
                    new=ex.submit(group,cursor);pending[new]=cursor;cursor+=1
    configure();result=core.summarize('stress')
    result['stress_protocol_sha256']=sha(OUT/'protocol.json')
    result['stress_protocol_path']=(OUT/'protocol.json').relative_to(ROOT).as_posix()
    result['superseded_old_stress_games']=0
    save(OUT/'summaries/stress-fast.json',result)
    # Explicit handoff to the unchanged main decision reader, never old runs.
    save(MAIN_OUT/'summaries/stress-fast.json',result,True)
    return result


def audit():
    configure();return core.audit()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run','audit']);args=parser.parse_args()
    with ExclusiveBatch(OUT/'execution.lock'):
        if args.stage=='prepare':prepare()
        elif args.stage=='run':batch()
        else:audit()

if __name__=='__main__':main()
