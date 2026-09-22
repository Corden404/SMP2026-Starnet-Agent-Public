"""Round22 same-prefix causal branches for the earliest selected B2B event per case."""
from __future__ import annotations
import argparse,collections,concurrent.futures,copy,json,math,os,statistics
from pathlib import Path
import round22_state as s
import round21_common as h
from round9_lock import ExclusiveBatch

OUT=s.OUT/'causal'
R21=s.ROOT/'agent_notes/experiments/round21/study'
POOLS=('dev','confirm','stress')
COST={'comm':2.,'cut':3.,'shield':5.}
DEPS=('workspace/round22_causal.py','tests/test_round22_causal.py','workspace/round22_state.py',
      'workspace/round21_common.py','workspace/round21_policy.py')

def code():
    return dict(archive=s.ARCHIVE_SHA,deps={x:s.sha(s.ROOT/x) for x in DEPS})

def key_action(key):
    kind,u,v=key
    if kind not in COST:raise ValueError('Unknown action key '+str(key))
    return dict(action=kind,target_node_1=u,target_node_2=v,prompt_id=1 if kind=='comm' else None,
                cost=COST[kind],gain=0.,roi=0.)

def pc2_seen(row,operation):
    return sorted({t for e in row.get('candidate_records',[]) if e.get('operation',10**9)<=operation for t in e.get('tags',[])})

def calls_before(row,operation):
    return sum(c['operation']<operation for c in row.get('chains',[]))

def inventory():
    found=[]
    for pool in POOLS:
        for path in sorted((R21/'runs'/pool).glob('*-B2B.json')):
            row=s.read(path)
            selected=[x for x in row.get('extra_records',[]) if x.get('kind')=='bundle' and x.get('reason')=='selected' and x.get('selected')]
            if not selected:continue
            event=min(selected,key=lambda x:x['operation'])
            op=event['operation']
            if row['actions'][op]['action']=='scan_node':raise RuntimeError('Selected event points at scan')
            inc=list(event['incumbent']);prefix=[list(x) for x in event['selected']['prefix']]
            if not prefix:raise RuntimeError('Empty selected bundle')
            found.append(dict(pool=pool,case=row['spec']['case'],family=row['spec']['family'],source_path=path.relative_to(s.ROOT).as_posix(),
                source_sha=s.sha(path),operation=op,budget=event['budget'],incumbent=inc,bundle=prefix,label=event['selected']['label'],
                calls=calls_before(row,op),pc2_seen=pc2_seen(row,op),prefix_sha=s.digest(row['actions'][:op])))
    found.sort(key=lambda x:(POOLS.index(x['pool']),x['case']))
    if len(found)!=28 or len({x['case'] for x in found})!=28:raise RuntimeError('Expected28 earliest selected-event cases')
    counts=collections.Counter(x['pool'] for x in found)
    if counts!=collections.Counter(dev=4,confirm=23,stress=1):raise RuntimeError('Selected pool counts changed: '+repr(counts))
    if any(x['case']=='r21-dev-029' and x['operation']!=72 for x in found):raise RuntimeError('dev029 earliest event changed')
    return found

def prepare():
    s.preserved();events=inventory()
    q=dict(code=code(),events=events,repeats=3,branches_per_event=6,total_new_branches=168,workers=2,
       branch='Restore exact B2B public prefix into exact PC2; consume PC2 tags through event operation; force incumbent one action or frozen selected structural prefix; exact PC2 thereafter.',
       ordering='Within each event/repeat alternate baseline/bundle order; two events may run concurrently. Successful immutable identities are reused.',
       metric='Per event use mean of 3 bundle finals minus mean of 3 baseline finals; 28 cases equal weight. Also SD, wins/ties/losses, without-best-two, pool strata.',
       gate='Diagnostic support for R3 iff mean>0, without-best-two>0, positive cases>=5 and positives>negatives, prefix/resource/audit all pass.',
       limitations='Post-selected Round21 trigger events; not fresh validation and not deployable score evidence. For 2-action bundles, forced consecutive actions isolate intervention effect and intentionally omit between-action CPU selector timing.',
       no_api=True,no_upload=True)
    s.write(OUT/'protocol.json',q,True)
    s.update('R2','RUNNING','28张最早B2B触发事件协议已冻结（dev4/confirm23/stress1）；baseline/bundle各3次，共168晚局分支。',
             next_action='运行同前缀局部因果分支；不按中途delta调整。',evidence='causal/protocol.json')
    print(json.dumps(dict(prepared=True,events=28,new_branches=168),ensure_ascii=False),flush=True)

def check():
    q=s.read(OUT/'protocol.json')
    if q['code']!=code():raise RuntimeError('Causal protocol source drift')
    return q

def checked(path,identity,fn):
    path=Path(path)
    if path.exists():
        r=s.read(path)
        if r.get('status')!='PASS' or r.get('identity')!=identity:raise RuntimeError('Incomplete/different causal record '+str(path))
        return r
    rec=dict(status='STARTED',identity=identity,pid=os.getpid(),started_at=s.now());s.write(path,rec,True)
    try:
        got=fn();rec.update(got,status='PASS',identity=identity,completed_at=s.now());s.write(path,rec);return rec
    except BaseException as exc:
        rec.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),failed_at=s.now());s.write(path,rec);raise

def restore(entry,namespace):
    path=s.ROOT/entry['source_path']
    if s.sha(path)!=entry['source_sha']:raise RuntimeError('Round21 source drift')
    row=s.read(path);module=h.load('PC2',namespace);h.clear(module)
    with h.c.offline():
        _,raw,logged,model=h.c.restore(row,entry['operation'],module)
    model.pc2_records=copy.deepcopy([x for x in row.get('candidate_records',[]) if x.get('operation',10**9)<=entry['operation']])
    model._pc2_seen=set(entry['pc2_seen'])
    if abs(raw.get_remaining_budget()-entry['budget'])>1e-8:raise RuntimeError('Restored budget mismatch')
    if model.llm_counter.call_count!=entry['calls']:raise RuntimeError('Restored call count mismatch')
    if s.digest(row['actions'][:entry['operation']])!=entry['prefix_sha']:raise RuntimeError('Prefix hash mismatch')
    return row,module,raw,logged,model

def branch(entry,which,repeat):
    row,module,raw,logged,model=restore(entry,'r22-causal-%s-%d'%(which,repeat))
    keys=[entry['incumbent']] if which=='baseline' else entry['bundle']
    actions=[key_action(x) for x in keys]
    with h.c.offline():
        for action in actions:h.c.force(model,raw,action)
        result=h.c.finish(raw,logged,model)
    result.update(spec=row['spec'],variant='PC2-forced-'+which,mode='round22-causal',
                  start_operation=entry['operation'],forced_actions=actions,prefix_sha256=entry['prefix_sha'],
                  causal_case=entry['case'],causal_label=entry['label'],repeat=repeat)
    err=h.c.audit_game(result);result['audit_error']=err;h.clear(module)
    return result

def run_event(index):
    q=check();e=q['events'][index];records=[]
    for rep in (1,2,3):
        order=('baseline','bundle') if (index+rep)%2==0 else ('bundle','baseline')
        for which in order:
            ident=dict(event=e,which=which,repeat=rep,code=q['code'])
            path=OUT/'runs'/(e['case']+'-%s-r%d.json'%(which,rep))
            records.append(checked(path,ident,lambda e=e,which=which,rep=rep:branch(e,which,rep)))
    return dict(case=e['case'],branches=len(records))

def pstdev(xs):return statistics.pstdev(xs) if len(xs)>1 else 0.

def summarize():
    q=check();details=[];maxerr=0.;prefix_errors=resource_errors=0
    for e in q['events']:
        vals={}
        for which in ('baseline','bundle'):
            rows=[s.read(OUT/'runs'/(e['case']+'-%s-r%d.json'%(which,r))) for r in (1,2,3)]
            for r in rows:
                if r.get('status')!='PASS' or r['identity']['event']!=e:raise RuntimeError('Mixed/incomplete causal result')
                maxerr=max(maxerr,r.get('audit_error',h.c.audit_game(r)))
                prefix_errors+=int(r['prefix_sha256']!=e['prefix_sha'] or r['start_operation']!=e['operation'])
                resource_errors+=int(r['operations']>=120 or r['logical_calls']>=120)
            scores=[r['final_score'] for r in rows]
            vals[which]=dict(scores=scores,mean=statistics.mean(scores),sd=pstdev(scores),spread=max(scores)-min(scores))
        delta=vals['bundle']['mean']-vals['baseline']['mean']
        details.append(dict(case=e['case'],pool=e['pool'],family=e['family'],label=e['label'],operation=e['operation'],
                            budget=e['budget'],delta=delta,baseline=vals['baseline'],bundle=vals['bundle']))
    deltas=[x['delta'] for x in details];ordered=sorted(deltas,reverse=True)
    without_best_two=statistics.mean(sorted(deltas,reverse=True)[2:]) if len(deltas)>2 else statistics.mean(deltas)
    eps=1e-8;wins=sum(x>eps for x in deltas);losses=sum(x<-eps for x in deltas);ties=len(deltas)-wins-losses
    strata={}
    for pool in POOLS:
        xs=[x['delta'] for x in details if x['pool']==pool]
        strata[pool]=dict(n=len(xs),mean=statistics.mean(xs) if xs else None,wins=sum(x>eps for x in xs),
                          ties=sum(abs(x)<=eps for x in xs),losses=sum(x<-eps for x in xs))
    mean=statistics.mean(deltas);median=statistics.median(deltas)
    gate=(mean>0 and without_best_two>0 and wins>=5 and wins>losses and prefix_errors==0 and resource_errors==0 and maxerr<=1e-8)
    result=dict(status='PASS',events=28,new_branches=168,mean_delta=mean,median_delta=median,without_best_two=without_best_two,
                wins=wins,ties=ties,losses=losses,pool_strata=strata,positive_cases=wins,negative_cases=losses,
                max_baseline_spread=max(x['baseline']['spread'] for x in details),max_bundle_spread=max(x['bundle']['spread'] for x in details),
                mean_baseline_spread=statistics.mean(x['baseline']['spread'] for x in details),
                mean_bundle_spread=statistics.mean(x['bundle']['spread'] for x in details),
                prefix_errors=prefix_errors,resource_errors=resource_errors,max_forward_error=maxerr,eligible_r3=gate,details=details)
    s.write(OUT/'summary.json',result,True)
    s.update('R2','COMPLETE','28同前缀事件：mean %+.6f；median %+.6f；去最好2 %+.6f；胜平负%d/%d/%d；R3资格%s。'%(
        mean,median,without_best_two,wins,ties,losses,gate),
        next_action='进入R3等价加速/稳定化研究。' if gate else '局部因果未过门槛；跳过R3/R4并完成回归审计。',
        evidence='causal/summary.json')
    if not gate:
        s.update('R3','SKIPPED_GATE','R2同前缀局部因果未通过冻结门槛。',evidence='causal/summary.json')
        s.update('R4','SKIPPED_GATE','R2同前缀局部因果未通过冻结门槛。',evidence='causal/summary.json')
    print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False),flush=True);return result

def run():
    q=check();s.preserved()
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        fut={ex.submit(run_event,i):i for i in range(len(q['events']))}
        for f in concurrent.futures.as_completed(fut):
            info=f.result()
            done=sum(all((OUT/'runs'/(e['case']+'-%s-r%d.json'%(w,r))).exists() and s.read(OUT/'runs'/(e['case']+'-%s-r%d.json'%(w,r))).get('status')=='PASS' for w in ('baseline','bundle') for r in (1,2,3)) for e in q['events'])
            if done%3==0 or done==28:s.update('R2','RUNNING','同前缀因果已持久化%d/28事件；最近%s。'%(done,info['case']),
                next_action='继续固定28事件；PASS复用，失败先检查。',evidence='causal/runs')
            print(json.dumps(dict(completed=done,**info),ensure_ascii=False),flush=True)
    return summarize()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary']);a=ap.parse_args()
    with ExclusiveBatch(s.OUT/'execution.lock'):
        {'prepare':prepare,'run':run,'summary':summarize}[a.stage]()
