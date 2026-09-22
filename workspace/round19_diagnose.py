"""Complete historical135-state error map; paired forced branches and equal-kernel shadows."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import copy
import json
import statistics
import time
import round19_state as s
import round19_common as h
from round9_lock import ExclusiveBatch

OUT=s.OUT/'diagnostic'
OLD=s.ROOT/'agent_notes/experiments/round18'
MODES=('depth2','full_s44','full_pc2')
DEPS=('workspace/round19_diagnose.py','tests/test_round19_numeric.py','tests/test_round19_diagnose.py')


def key(record):
    names={'communicate':'comm','shield_node':'shield','cut_link':'cut'}
    kind=names[record['action']];args=record['args']
    return (kind,args[0],args[1] if kind=='cut' else None)


def inventory():
    entries=[]
    for i,path in enumerate(sorted((OLD/'study/runs/dev').glob('*-PC2.json'))):
        row=s.read(path);cutrow=s.read(path.with_name(path.name.replace('-PC2','-Cut1')))
        for tag in ('budget20','budget10'):
            ev=next(e for e in row['candidate_records'] if tag in e['tags']);snap=h.snapshot(row,ev)
            entries.append(dict(id='dev-%03d-%s'%(i,tag),case=row['spec']['case'],family=row['spec']['family'],primary=True,
                source_path=path.relative_to(s.ROOT).as_posix(),source_sha256=s.sha(path),snapshot=snap,
                incumbent=h.action(key(row['actions'][snap['operation']])),reuse=None))
            cev=next(e for e in cutrow['extra_records'] if tag in e['tags'])
            def prefix(r,op):return [(a['action'],a['args']) for a in r['actions'][:op]]
            if prefix(row,snap['operation'])!=prefix(cutrow,cev['operation']):
                cp=path.with_name(path.name.replace('-PC2','-Cut1'))
                entries.append(dict(id='supp-%03d-%s'%(i,tag),case=row['spec']['case'],family=row['spec']['family'],primary=False,
                    source_path=cp.relative_to(s.ROOT).as_posix(),source_sha256=s.sha(cp),snapshot=h.snapshot(cutrow,cev),
                    incumbent=h.action(cev['incumbent']),reuse=None))
    for i,path in enumerate(sorted((OLD/'diagnostic/graphs').glob('*.json'))):
        row=s.read(path);bp=OLD/row['base_path']
        for j,item in enumerate(row['states']):
            entries.append(dict(id='diag-%03d-%d'%(i,j),case=row['spec']['case'],family=row['spec']['family'],primary=True,
                source_path=bp.relative_to(s.ROOT).as_posix(),source_sha256=s.sha(bp),snapshot=item['snapshot'],
                incumbent=item['snapshot']['chosen'],reuse={k:(OLD/v).relative_to(s.ROOT).as_posix() for k,v in item.get('branch_paths',{}).items()}))
    if len(entries)!=135 or sum(e['primary'] for e in entries)!=132:raise RuntimeError('Historical inventory differs from approved plan')
    if len({e['id'] for e in entries})!=135:raise RuntimeError('Duplicate state identity')
    return entries


def gates(metrics,coverage,risk_cases):
    old=metrics['depth2'];new=metrics['full_pc2']
    mae_improved=new['graph_mae']<old['graph_mae']-1e-8
    decisions_improved=new['graph_selected_realized']>old['graph_selected_realized']+1e-8
    nondominated=(new['graph_mae']<=old['graph_mae']+1e-8 and
                  new['graph_selected_realized']>=old['graph_selected_realized']-1e-8)
    eligible=coverage>=.90 and nondominated and (mae_improved or decisions_improved)
    risk_graphs={x['case'] for x in risk_cases}
    risk_supported=eligible and len(risk_graphs)>=2 and any(x['actual_delta']<0 for x in risk_cases)
    return dict(eligible_ct1=eligible,eligible_ct1r=risk_supported,coverage=coverage,mae_improved=mae_improved,
                decisions_improved=decisions_improved,nondominated=nondominated,risk_graphs=len(risk_graphs))


def prepare():
    if not s.read(s.OUT/'numeric-tests.json')['success']:raise RuntimeError('Numerical tests must pass')
    s.preserved();entries=inventory()
    sources={e['source_path']:e['source_sha256'] for e in entries}
    for e in entries:
        for path in (e['reuse'] or {}).values():sources[path]=s.sha(s.ROOT/path)
    protocol=dict(code=h.fingerprint(DEPS),entries=entries,sources=sources,states=135,primary_states=132,supplementary_states=3,
        historical_graphs=66,new_independent_graphs=0,workers=2,max_new_branches=222,
        forecasts=list(MODES),forecast_seconds=8.,same_private_kernel=True,
        continuation='Forced action then exact archived PC2 via public interface. Restore consumed windows, operation/call counts. No latent access by policy.',
        reuse='48 original diagnostic branches audited, not rerun or counted as new.',
        metrics='On complete COMMON primary states: within-graph means then equal graph mean of prediction absolute error and realized selected-branch delta. Additional3 prefixes reported separately. Missing comparisons retained and coverage reported.',
        gate='Before seeing the135-state result: coverage>=90%; full_pc2 graph MAE<=depth2 AND realized selected delta>=depth2, with one strict improvement>1e-8. Otherwise stop CT1. No retuning or extension.',
        risk_gate='Only if CT1 gate passes: at least2 distinct primary graphs with mean>=0.5,min-world<0 and differential unobserved-response actions, including at least1 realized loss. Risk variant remains an unvalidated prospective candidate.',
        limitations='Realized selection uses diagnostic labels and is not deployable gain. Two outcomes can differ from posterior expectation even with correct model. Wall clocks and accelerated kernel limit exact policy equivalence.',
        no_api=True,no_upload=True)
    h.load('Numeric')
    s.write(OUT/'protocol.json',protocol,True)
    s.update('R1','RUNNING','135状态清单与同核8秒影子对照已冻结；132主状态、3补充状态，旧48分支复用。',
             next_action='运行 round19_diagnose.py run --start 0 --stop 135；完成状态即时同步。',evidence='diagnostic/protocol.json')
    print(json.dumps(dict(prepared=True,states=135,primary=132,new_branches_max=222)),flush=True)


def check():
    p=s.read(OUT/'protocol.json')
    if p['code']!=h.fingerprint(DEPS):raise RuntimeError('Frozen diagnostic source drift')
    return p


def unknown_difference(state,evidence):
    unknown=set(state[0])-set(state[2]);result=set()
    for world in evidence.get('worlds',[]):
        branches=world['branches']
        a=branches.get('0',branches.get(0));b=branches.get('1',branches.get(1))
        ac=collections.Counter(x[1] for x in a['actions'] if x[0]=='comm')
        bc=collections.Counter(x[1] for x in b['actions'] if x[0]=='comm')
        result.update(i for i in set(ac)|set(bc) if ac[i]!=bc[i] and i in unknown)
    return sorted(result)


def shadow_record(identity,mode,evidence):
    return dict(evidence,identity=identity,mode=mode)


def run_state(index):
    p=check();entry=p['entries'][index];ident=dict(entry=entry,code=p['code'])
    path=OUT/'states'/(entry['id']+'.json')
    def execute():
        row=s.read(s.ROOT/entry['source_path'])
        if s.sha(s.ROOT/entry['source_path'])!=entry['source_sha256']:raise RuntimeError('Historical source changed')
        snap=entry['snapshot'];state,eligible,ctx=h.public_at(row,snap)
        module=h.load('Numeric');h.clear(module)
        with h.old.c.offline():cut=module.r18_best_cut(state,snap['budget'],eligible,time.monotonic()+8.)
        same=cut is not None and module.BudgetRollout.key(cut)==module.BudgetRollout.key(entry['incumbent'])
        result=dict(entry_id=entry['id'],case=entry['case'],family=entry['family'],primary=entry['primary'],snapshot=snap,
                    reason='NO_CUT' if cut is None else 'SAME_ROOT' if same else 'COMPARED',incumbent=entry['incumbent'],cut=cut)
        if cut is None or same:return result
        branch_rows={};paths={};actions={'incumbent':entry['incumbent'],'cut':cut}
        for which in (('incumbent','cut') if index%2==0 else ('cut','incumbent')):
            if entry['reuse'] and which in entry['reuse']:
                rp=s.ROOT/entry['reuse'][which]
                if s.sha(rp)!=p['sources'][entry['reuse'][which]]:raise RuntimeError('Old branch changed')
                br=s.read(rp);h.old.c.audit_game(br)
                if br['start_operation']!=snap['operation'] or br['prefix_sha256']!=s.digest(row['actions'][:snap['operation']]):raise RuntimeError('Reused prefix mismatch')
                if module.BudgetRollout.key(br['forced_action'])!=module.BudgetRollout.key(actions[which]):raise RuntimeError('Reused forced action differs')
                paths[which]=dict(path=entry['reuse'][which],reused=True)
            else:
                rp=OUT/'branches'/(entry['id']+'-'+which+'.json')
                br=h.checked(rp,dict(ident,which=which,action=actions[which]),lambda which=which:h.branch(row,snap,actions[which]))
                paths[which]=dict(path=rp.relative_to(s.ROOT).as_posix(),reused=False)
            branch_rows[which]=br
        result.update(actual_delta=branch_rows['cut']['final_score']-branch_rows['incumbent']['final_score'],
                      baseline_suffix_drift=branch_rows['incumbent']['final_score']-row['final_score'],branch_paths=paths,forecasts={})
        for mode in (MODES if index%2==0 else MODES[::-1]):
            h.clear(module);start=time.monotonic()
            with h.old.c.offline():
                _,ev=module.r19_compare(state,snap['budget'],eligible,ctx,entry['incumbent'],cut,seconds=8.,mode=mode)
            ev['seconds']=time.monotonic()-start
            ev['differential_unobserved_responses']=unknown_difference(state,ev)
            if ev.get('completed_worlds')==3:
                delta=result['actual_delta'];ev['abs_error']=abs(ev['predicted_delta']-delta)
                ev['selected_realized']=delta if ev['predicted_delta']>=.5 else 0.
                ev['regret']=max(0.,delta)-ev['selected_realized']
                ev['risk_selected_realized']=delta if ev['predicted_delta']>=.5 and ev['min_delta']>=0 else 0.
            result['forecasts'][mode]=ev
            s.write(OUT/'shadows'/(entry['id']+'-'+mode+'.json'),shadow_record(ident,mode,ev),True)
        h.clear(module)
        return result
    result=h.checked(path,ident,execute)
    return dict(id=entry['id'],primary=entry['primary'],reason=result['reason'])


def summarize():
    p=check();rows=[]
    for e in p['entries']:
        row=s.read(OUT/'states'/(e['id']+'.json'))
        if row['status']!='PASS' or row['identity']!=dict(entry=e,code=p['code']):raise RuntimeError('Incomplete/mixed135-state diagnosis')
        rows.append(row)
    primary=[r for r in rows if r['primary'] and r['reason']=='COMPARED']
    common=[r for r in primary if all(r['forecasts'][mode].get('completed_worlds')==3 for mode in MODES)]
    metrics={}
    for mode in MODES:
        families=collections.defaultdict(list)
        for row in common:families[row['case']].append(row['forecasts'][mode])
        metrics[mode]=dict(graphs=len(families),states=len(common),
            graph_mae=statistics.mean(statistics.mean(x['abs_error'] for x in v) for v in families.values()) if families else 0.,
            graph_selected_realized=statistics.mean(statistics.mean(x['selected_realized'] for x in v) for v in families.values()) if families else 0.,
            graph_regret=statistics.mean(statistics.mean(x['regret'] for x in v) for v in families.values()) if families else 0.,
            graph_risk_selected_realized=statistics.mean(statistics.mean(x['risk_selected_realized'] for x in v) for v in families.values()) if families else 0.,
            selected_states=sum(r['forecasts'][mode]['predicted_delta']>=.5 for r in common),
            total_complete=sum(r['forecasts'][mode].get('completed_worlds')==3 for r in primary),
            timeouts=sum(r['forecasts'][mode].get('completed_worlds')!=3 for r in primary))
    risk_cases=[]
    for row in common:
        f=row['forecasts']['full_pc2']
        if f['predicted_delta']>=.5 and f['min_delta']<0 and f['differential_unobserved_responses']:
            risk_cases.append(dict(id=row['entry_id'],case=row['case'],actual_delta=row['actual_delta'],
                predicted=f['predicted_delta'],min_delta=f['min_delta'],unknown=f['differential_unobserved_responses']))
    coverage=len(common)/max(1,len(primary));decision=gates(metrics,coverage,risk_cases)
    result=dict(status='PASS',states=len(rows),primary_states=132,supplementary_states=3,historical_graphs=66,
                common_states=len(common),metrics=metrics,decision=decision,risk_cases=risk_cases,
                baseline_suffix_drift_states=sum(abs(r.get('baseline_suffix_drift',0))>1e-8 for r in rows),
                new_branches=sum(1 for _ in (OUT/'branches').glob('*.json')),
                reused_branches=sum(x['reused'] for r in rows for x in r.get('branch_paths',{}).values()),
                no_margin_and_supplementary_included=True,new_independent_graphs=0,api_requests=0,
                limitations=p['limitations'],code=p['code'])
    s.write(OUT/'summary.json',result,True)
    s.update('R1','COMPLETE','135状态完成；主状态共同完整%d；CT1资格%s，CT1-R资格%s；新增%d分支、复用%d。'%
             (len(common),decision['eligible_ct1'],decision['eligible_ct1r'],result['new_branches'],result['reused_branches']),
             next_action='按诊断门槛实现并冻结新候选。' if decision['eligible_ct1'] else '诊断未支持候选；按计划停止后续策略队列，执行全量回归和审计。',evidence='diagnostic/summary.json')
    if not decision['eligible_ct1']:
        for step in ('R2','R3','R4'):s.update(step,'SKIPPED_GATE','完整R1诊断未达到事前冻结的后缀改善标准，不新增或调参候选。',evidence='diagnostic/summary.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ('code','risk_cases')},ensure_ascii=False),flush=True)
    return result


def run(start,stop):
    p=check();s.preserved()
    if not 0<=start<stop<=135:raise ValueError('Invalid fixed block')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        pending={};cursor=start
        while cursor<min(stop,start+2):pending[pool.submit(run_state,cursor)]=cursor;cursor+=1
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                pending.pop(future);info=future.result()
                complete=sum(1 for path in (OUT/'states').glob('*.json') if s.read(path).get('status')=='PASS')
                s.update('R1','RUNNING','已持久化%d/135状态；最近%s。'%(complete,info['id']),
                    next_action='继续同一固定块；PASS身份匹配复用，STARTED/FAILED先检查。',evidence='diagnostic/states/'+info['id']+'.json')
                print(json.dumps(dict(completed=complete,**info)),flush=True)
                if cursor<stop:pending[pool.submit(run_state,cursor)]=cursor;cursor+=1
    if all((OUT/'states'/(e['id']+'.json')).exists() for e in p['entries']):return summarize()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=135);a=ap.parse_args()
    try:
        with ExclusiveBatch(s.OUT/'execution.lock'):
            if a.stage=='prepare':prepare()
            elif a.stage=='run':run(a.start,a.stop)
            else:summarize()
    except BaseException as exc:
        s.update('R1','BLOCKED',type(exc).__name__+': '+str(exc),overall='BLOCKED',next_action='保留失败状态，检查后再显式恢复；不重跑成功记录。')
        raise
