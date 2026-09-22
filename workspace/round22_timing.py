"""Round22 exact-PC2 repeatability probe on fixed Round21 stress cases."""
from __future__ import annotations
import argparse,collections,concurrent.futures,json,statistics
from pathlib import Path
import round22_state as s
import round21_common as h
from round9_lock import ExclusiveBatch
OUT=s.OUT/'timing'
INDICES=(0,5,10,15,20,25,31,35,40,45,50,53)
DEPS=('workspace/round22_timing.py','tests/test_round22_timing.py','workspace/round22_state.py','workspace/round21_common.py')

def code():return dict(archive=s.ARCHIVE_SHA,deps={x:s.sha(s.ROOT/x) for x in DEPS})

def prepare():
    s.preserved();specs=h.specs('stress');entries=[]
    for i in INDICES:
        spec=specs[i];path=s.ROOT/'agent_notes/experiments/round21/study/runs/stress'/(spec['case']+'-PC2.json')
        row=s.read(path)
        if row.get('status')!='PASS' or s.digest(h.c.generate(spec))!=row['identity']['entry']['sha256']:
            raise RuntimeError('Archived stress identity mismatch '+spec['case'])
        entries.append(dict(index=i,spec=spec,seed_sha=s.digest(h.c.generate(spec)),archived_path=path.relative_to(s.ROOT).as_posix(),archived_sha=s.sha(path)))
    pr=dict(code=code(),entries=entries,repeats=2,new_games=24,workers=2,
            metric='Compare scan prefix, pre-budget20 intervention sequence, full action sequence/bag/final score and first-window PC2 evidence.',
            drift_gate='Intrinsic timing drift confirmed if any fixed case has >1 distinct pre-budget20 prefix across archived+2 repeats.',
            no_api=True,no_upload=True)
    s.write(OUT/'protocol.json',pr,True)
    s.update('R1','RUNNING','12张固定stress图重复性协议已冻结；每图新增2次exact PC2，共24局。',
             next_action='运行固定12图重复探针。',evidence='timing/protocol.json')
    print(json.dumps(dict(prepared=True,cases=len(entries),new_games=24),ensure_ascii=False),flush=True)

def check():
    q=s.read(OUT/'protocol.json')
    if q['code']!=code():raise RuntimeError('Timing protocol source drift')
    return q

def checked(path,identity,fn):
    path=Path(path)
    if path.exists():
        r=s.read(path)
        if r.get('status')!='PASS' or r.get('identity')!=identity:raise RuntimeError('Incomplete/different repeat '+str(path))
        return r
    rec=dict(status='STARTED',identity=identity,pid=__import__('os').getpid(),started_at=s.now());s.write(path,rec,True)
    try:
        got=fn();rec.update(got,status='PASS',identity=identity,completed_at=s.now());s.write(path,rec);return rec
    except BaseException as exc:
        rec.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),failed_at=s.now());s.write(path,rec);raise

def run_case(pos):
    q=check();e=q['entries'][pos];rows=[]
    for rep in (1,2):
        ident=dict(entry=e,repeat=rep,code=q['code'])
        path=OUT/'runs'/(e['spec']['case']+'-repeat%d.json'%rep)
        row=checked(path,ident,lambda:h.run_game(e['spec'],'PC2','fast'));rows.append(row)
    return dict(case=e['spec']['case'],runs=len(rows))

def action_seq(row):
    return [(a['action'],tuple(a['args'])) for a in row['actions']]

def bag(row):
    return sorted(h.bag(row).items())

def pre20(row):
    op=next((x['operation'] for x in row['candidate_records'] if 'budget20' in x.get('tags',[])),len(row['actions']))
    return [(a['action'],tuple(a['args'])) for a in row['actions'][44:op]]

def summarize():
    q=check();cases=[];drift=0
    for e in q['entries']:
        archived=s.read(s.ROOT/e['archived_path'])
        if s.sha(s.ROOT/e['archived_path'])!=e['archived_sha']:raise RuntimeError('Archived repeat source changed')
        rows=[archived]+[s.read(OUT/'runs'/(e['spec']['case']+'-repeat%d.json'%r)) for r in (1,2)]
        pre=[pre20(x) for x in rows];seq=[action_seq(x) for x in rows];bags=[bag(x) for x in rows];scores=[x['final_score'] for x in rows]
        scan=[h.scan_prefix(x) for x in rows]
        distinct_pre=len({s.digest(x) for x in pre});distinct_seq=len({s.digest(x) for x in seq});distinct_bag=len({s.digest(x) for x in bags})
        d=distinct_pre>1;drift+=d
        first=[x['candidate_records'][0] if x['candidate_records'] else None for x in rows]
        cases.append(dict(case=e['spec']['case'],index=e['index'],pre20_drift=d,distinct_pre20=distinct_pre,distinct_full_sequence=distinct_seq,
                          distinct_bag=distinct_bag,score_min=min(scores),score_max=max(scores),score_range=max(scores)-min(scores),
                          scan_identical=len({s.digest(x) for x in scan})==1,first_window=first,
                          scores=scores,pre20_signatures=[s.digest(x) for x in pre]))
    stress31=next(x for x in cases if x['index']==31)
    result=dict(status='PASS',cases=12,new_games=24,pre20_drift_cases=drift,intrinsic_timing_drift=drift>0,
                stress031_pre20_drift=stress31['pre20_drift'],stress031_score_range=stress31['score_range'],
                full_sequence_drift_cases=sum(x['distinct_full_sequence']>1 for x in cases),
                bag_drift_cases=sum(x['distinct_bag']>1 for x in cases),max_score_range=max(x['score_range'] for x in cases),details=cases)
    s.write(OUT/'summary.json',result,True)
    s.update('R1','COMPLETE','12图PC2重复探针：budget20前漂移%d/12；全序列漂移%d/12；stress031复现=%s；最大score range %.6f。'%(
        drift,result['full_sequence_drift_cases'],stress31['pre20_drift'],result['max_score_range']),
        next_action='执行R2：28张触发图同前缀baseline/bundle各3次反事实分支。',evidence='timing/summary.json')
    print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False),flush=True);return result

def run():
    q=check();s.preserved()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        fut={ex.submit(run_case,i):i for i in range(len(q['entries']))}
        for f in concurrent.futures.as_completed(fut):
            info=f.result();done=sum(all((OUT/'runs'/(e['spec']['case']+'-repeat%d.json'%r)).exists() and s.read(OUT/'runs'/(e['spec']['case']+'-repeat%d.json'%r)).get('status')=='PASS' for r in (1,2)) for e in q['entries'])
            if done%2==0 or done==12:s.update('R1','RUNNING','PC2重复探针已持久化%d/12图；最近%s。'%(done,info['case']),next_action='继续固定12图，不按中途漂移调整。',evidence='timing/runs')
            print(json.dumps(dict(completed=done,**info),ensure_ascii=False),flush=True)
    return summarize()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary']);a=ap.parse_args()
    with ExclusiveBatch(s.OUT/'execution.lock'):
        {'prepare':prepare,'run':run,'summary':summarize}[a.stage]()
