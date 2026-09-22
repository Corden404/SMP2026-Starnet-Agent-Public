"""Pre-registered PC2/Cut1/Fast study; durable full fixed cohorts, no score-driven extension."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
import round18_state as s
import round18_common as h
from round9_lock import ExclusiveBatch

OUT=s.OUT/'study'
DEPS=('workspace/round18_study.py','workspace/round18_cut1.py','workspace/round18_fast.py',
      'tests/test_round18_policies.py','tests/test_round18_study.py')
POOLS=('dev','confirm','stress','integration','real')


def package_hashes(variants=('PC2','Cut1','Fast')):
    return {v:{n:hashlib.sha256(b).hexdigest() for n,b in h.source_files(v).items()} for v in variants}


def code():
    return dict(core=h.fingerprint(DEPS),packages=package_hashes())


def benchmark_prepare():
    if not s.read(s.OUT/'preflight-policies.json')['success']:
        raise RuntimeError('Policy safety preflight not passed')
    cases=[row['spec']['case'] for row in h.entries('diagnostic')[:6]]
    paths={case:'diagnostic/baseline/'+case+'.json' for case in cases}
    p=dict(code=code(),cases=cases,baseline_hashes={case:s.sha(s.OUT/path) for case,path in paths.items()},
        blocks=3,enumerations_per_block=8,order='Alternating variant order by graph and block; cold-clear before each block.',
        gate='All candidate dictionaries exactly equal; aggregate full workload Fast time lower than PC2. Fixed workload, no retuning.',
        scope='Timing-only six fixed public snapshots, not strategy gain')
    s.write(OUT/'benchmark-protocol.json',p,True)
    return p


def benchmark():
    p=benchmark_prepare()
    def workload(case):
        base=s.read(s.OUT/'diagnostic/baseline'/(case+'.json'))
        snap=next(x for x in base['states'] if 'budget20' in x['tags'])
        _,raw,logged,model=h.restore(base,snap,'bench-state')
        state=model.completed_state();eligible=set(model.local_nodes)
        slots=min(model.operation_limit-1-model.operation_count,model.llm_counter.safe_limit-model.llm_counter.call_count)
        return state,eligible,slots,snap['budget']
    rows=[]
    for i,case in enumerate(p['cases']):
        state,eligible,slots,budget=workload(case)
        for block in range(p['blocks']):
            outcomes={}
            for variant in (('PC2','Fast') if (i+block)%2==0 else ('Fast','PC2')):
                module=h.load(variant,'benchmark');h.clear(module)
                engine=module.PC2Rollout(*state,module.SEARCH,slots,eligible=eligible)
                durations=[];actions=[]
                for rep in range(p['enumerations_per_block']):
                    belief=module.BudgetRollout.clone(state)
                    for node,d in belief[0].items():d['w']+=(rep*.001 if node in eligible else 0.)
                    start=time.perf_counter();candidates=engine.candidates(belief,budget)
                    durations.append(time.perf_counter()-start);actions.append(candidates)
                outcomes[variant]=dict(durations=durations,actions=actions)
                h.clear(module)
            equal=outcomes['PC2']['actions']==outcomes['Fast']['actions']
            rows.append(dict(case=case,block=block,exact_equal=equal,
                seconds={v:sum(outcomes[v]['durations']) for v in outcomes},
                warm_seconds={v:sum(outcomes[v]['durations'][1:]) for v in outcomes}))
    total={v:sum(row['seconds'][v] for row in rows) for v in ('PC2','Fast')}
    warm={v:sum(row['warm_seconds'][v] for row in rows) for v in ('PC2','Fast')}
    result=dict(status='PASS',code=p['code'],rows=rows,seconds=total,warm_seconds=warm,
        speedup=total['PC2']/total['Fast'],warm_speedup=warm['PC2']/warm['Fast'],
        exact_equal=all(row['exact_equal'] for row in rows))
    result['eligible_fast']=result['exact_equal'] and total['Fast']<total['PC2']
    s.write(OUT/'benchmark.json',result,True)
    s.update('P2B','COMPLETE' if result['eligible_fast'] else 'SKIPPED_GATE',
        '固定六状态三轮对照：候选逐字段一致=%s；完整工作量加速 %.3f 倍，热缓存 %.3f 倍。'%
        (result['exact_equal'],result['speedup'],result['warm_speedup']),
        next_action='冻结候选并运行 54 图配对开发；性能提升不当作提分。',evidence='study/benchmark.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','code')},ensure_ascii=False),flush=True)
    return result


def prepare():
    s.preserved()
    diagnosis=s.read(s.OUT/'diagnostic/summary.json');bench=s.read(OUT/'benchmark.json')
    if bench['code']!=code():raise RuntimeError('Benchmark source changed')
    variants=['PC2']
    if diagnosis['eligible_cut1']:variants.append('Cut1')
    if bench['eligible_fast']:variants.append('Fast')
    if len(variants)==1:raise RuntimeError('No candidate passed implementation gate')
    seeds={pool:h.entries(pool) for pool in POOLS}
    fresh=[e['sha256'] for pool,entries in seeds.items() for e in entries]
    old={e['sha256'] for e in h.entries('diagnostic')}
    if len(set(fresh))!=len(fresh) or set(fresh)&old:raise RuntimeError('Cohort overlap')
    p=dict(code=code(),variants=variants,seeds=seeds,workers=2,
        baseline='PC2',counts={pool:len(entries) for pool,entries in seeds.items()},
        benchmark_sha256=s.sha(OUT/'benchmark.json'),diagnosis_sha256=s.sha(s.OUT/'diagnostic/summary.json'),
        design='Same deterministic44 scan prefix; complete6x3x3 development. Unique216 confirmation and54 balanced stress. Integration6 and real24 have distinct identities.',
        development_gate='Mean>0,>=3 wins spanning>=2 families,prefix drift=0,unexplained first divergence=0,nontriggered Cut1 score/bag drift=0. Fast completed-check rate>=PC2 and benchmark gate passed. Tail/without-best-two descriptive.',
        selection='Select one eligible candidate by development mean; exact tie prefers Fast. Never combine candidates.',
        local_gate='Fixed216 confirmation mean>0 AND54 stress mean>=0 plus same safety/trace conditions. Both paired-t and family-bootstrap lower>0 required for confirmed-local label only.',
        real_design='Only if local gate passes:24 new graphs x2 repeats x2 variants; same real model; graph-first aggregation. No mid-cohort model switch or sample extension.',
        runtime_scope='Soft deadlines remain hardware-dependent; report first divergence explanations, no claim of bit-identical runtimes.',
        no_upload=True,no_git_commit=True)
    for variant in variants:h.stage(variant)
    s.write(OUT/'protocol.json',p,True)
    if 'Cut1' in variants:s.update('P2A','COMPLETE','Cut1 实现与公开边界/完整比较/缓存隔离测试通过，源码已冻结；尚未确认提分。',evidence='preflight-policies.json; study/protocol.json')
    s.update('P3','RUNNING','54 图开发协议已冻结，策略：'+', '.join(variants)+'。',
        next_action='运行 workspace/round18_study.py run --pool dev --start 0 --stop 54',evidence='study/protocol.json')
    print(json.dumps(dict(prepared=True,variants=variants,counts=p['counts'])),flush=True)


def check():
    p=s.read(OUT/'protocol.json')
    if p['code']!=code():raise RuntimeError('Frozen study source drift')
    if p['benchmark_sha256']!=s.sha(OUT/'benchmark.json') or p['diagnosis_sha256']!=s.sha(s.OUT/'diagnostic/summary.json'):
        raise RuntimeError('Frozen prior evidence changed')
    return p


def variants(pool):
    p=check()
    if pool in ('dev','integration'):return p['variants']
    decision=s.read(OUT/'development-decision.json')
    if not decision['selected']:raise RuntimeError('Development did not admit any candidate')
    if pool=='real' and not s.read(OUT/'local-decision.json')['eligible_real']:
        raise RuntimeError('Local gate forbids paid real stage')
    return ['PC2',decision['selected']]


def run_one(pool,index,variant):
    p=check();entry=p['seeds'][pool][index];spec=entry['spec']
    if h.c.digest(h.c.generate(spec))!=entry['sha256']:raise RuntimeError('Seed identity changed')
    mode='integration' if pool=='integration' else 'fast'
    identity=dict(entry=entry,variant=variant,code=p['code'],mode=mode)
    return h.checked_record(OUT/'runs'/pool/(spec['case']+'-'+variant+'.json'),identity,
        lambda:h.run_game(spec,variant,mode=mode))


def group(pool,index):
    order=variants(pool);shift=index%len(order);order=order[shift:]+order[:shift]
    if (index//len(order))%2:order=order[::-1]
    rows=[run_one(pool,index,v) for v in order]
    return dict(case=rows[0]['spec']['case'],games=len(rows),seconds=sum(r['elapsed_seconds'] for r in rows))


def action_key(record):
    names={'communicate':'comm','cut_link':'cut','shield_node':'shield','scan_node':'scan'}
    name=names[record['action']];args=record['args']
    return [name,args[0],args[1] if name=='cut' else None]


def plan_key(plan):
    a=plan['first'];return [a['action'],a['target_node_1'],a['target_node_2']]


def divergence(base,candidate,variant):
    """Classify only the first changed action from an otherwise identical public prefix."""
    if h.scan_prefix(base)!=h.scan_prefix(candidate):return dict(reason='SCAN_DRIFT',explained=False)
    pairs=list(zip(base['actions'],candidate['actions']))
    index=next((i for i,(a,b) in enumerate(pairs) if a['action']!=b['action'] or a['args']!=b['args']),None)
    if index is None:
        if len(base['actions'])==len(candidate['actions']):return dict(reason='IDENTICAL',explained=True)
        return dict(reason='UNEQUAL_SUFFIX_LENGTH',explained=False)
    info=dict(operation=index,baseline_action=action_key(base['actions'][index]),candidate_action=action_key(candidate['actions'][index]))
    if variant=='Cut1':
        event=next((r for r in candidate['extra_records'] if r['operation']==index and r.get('replaced')),None)
        if event and list(event['incumbent'])==info['baseline_action'] and list(event['chosen'])==info['candidate_action']:
            return dict(info,reason='CUT1_REPLACEMENT',explained=True)
    roots=[]
    for row in (base,candidate):
        event=next((r for r in row['search_records'] if r['operation']==index),None)
        roots.append(plan_key(event['plans'][0]) if event and event.get('plans') else None)
    if roots[0]!=roots[1]:return dict(info,reason='ROOT_SEARCH_DRIFT',explained=False,root_keys=roots)
    if variant=='Fast':
        events=[next((r for r in row['candidate_records'] if r['operation']==index),None) for row in (base,candidate)]
        if all(events):
            complete=[r.get('completed_worlds',0)==3 for r in events]
            if complete[0]!=complete[1]:
                return dict(info,reason='PRIVATE_COMPARISON_COMPLETION',explained=True,completed=complete)
        return dict(info,reason='UNEXPLAINED_PC2_DECISION',explained=False,events=events)
    return dict(info,reason='UNEXPLAINED_DECISION',explained=False)


def statistics_for(pool,variant):
    p=check();pairs=[];details=[];max_error=0.;prefix=0;nontriggered=0;unexplained=0
    completed=collections.Counter();checks=collections.Counter();max_ops=max_calls=0
    for entry in p['seeds'][pool]:
        case=entry['spec']['case'];rows={}
        for v in ('PC2',variant):
            row=s.read(OUT/'runs'/pool/(case+'-'+v+'.json'))
            if row['status']!='PASS' or row['identity']['entry']!=entry or row['identity']['code']!=p['code'] or row['variant']!=v:
                raise RuntimeError('Incomplete/mixed study cohort')
            max_error=max(max_error,h.c.audit_game(row));rows[v]=row
            completed[v]+=sum(e.get('completed_worlds',0)==3 for e in row['candidate_records'])
            checks[v]+=len(row['candidate_records']);max_ops=max(max_ops,row['operations']);max_calls=max(max_calls,row['logical_calls'])
        a,b=rows['PC2'],rows[variant];delta=b['final_score']-a['final_score']
        explanation=divergence(a,b,variant)
        changed_score_or_bag=abs(delta)>1e-8 or h.bag(a)!=h.bag(b)
        trigger=any(r.get('replaced') for r in b['extra_records']) if variant=='Cut1' else None
        prefix+=int(h.scan_prefix(a)!=h.scan_prefix(b))
        nontriggered+=int(variant=='Cut1' and not trigger and changed_score_or_bag)
        unexplained+=int(not explanation['explained'] and changed_score_or_bag)
        pairs.append(dict(case=case,family=entry['spec']['family'],delta=delta))
        details.append(dict(case=case,delta=delta,triggered=trigger,explanation=explanation,
            same_action_bag=h.bag(a)==h.bag(b),extra_replacements=sum(r.get('replaced',False) for r in b['extra_records'])))
    stat=h.c.paired_stats(pairs)
    stat.update(pairs=pairs,details=details,prefix_drift=prefix,nontriggered_different=nontriggered,unexplained_divergences=unexplained,
        complete_comparisons=dict(completed),comparison_checks=dict(checks),max_operations=max_ops,max_calls=max_calls,max_forward_error=max_error,
        complete_rates={v:completed[v]/max(1,checks[v]) for v in ('PC2',variant)})
    return stat


def passes_safety(stat,variant):
    return (stat['prefix_drift']==0 and stat['nontriggered_different']==0 and stat['unexplained_divergences']==0
        and stat['max_operations']<120 and stat['max_calls']<120 and stat['max_forward_error']<=1e-8
        and (variant!='Fast' or stat['complete_rates']['Fast']>=stat['complete_rates']['PC2']))


def select_development(stats):
    eligible=[]
    for variant,stat in stats.items():
        families={r['family'] for r in stat['pairs'] if r['delta']>1e-8}
        if stat['mean']>0 and stat['wins']>=3 and len(families)>=2 and passes_safety(stat,variant):eligible.append(variant)
    eligible.sort(key=lambda v:(-stats[v]['mean'],0 if v=='Fast' else 1))
    return eligible[0] if eligible else None


def summarize(pool):
    stats={v:statistics_for(pool,v) for v in variants(pool) if v!='PC2'}
    result=dict(pool=pool,status='PASS',graphs=len(check()['seeds'][pool]),comparisons=stats)
    s.write(OUT/'summaries'/(pool+'.json'),result,True)
    if pool=='dev':
        selected=select_development(stats)
        decision=dict(selected=selected,stats_sha256=s.sha(OUT/'summaries/dev.json'),code=check()['code'],
            candidates={v:dict(mean=x['mean'],wins=x['wins'],safety=passes_safety(x,v),unexplained=x['unexplained_divergences'],
                prefix_drift=x['prefix_drift'],nontriggered_different=x['nontriggered_different'],complete_rates=x['complete_rates']) for v,x in stats.items()})
        s.write(OUT/'development-decision.json',decision,True)
        summary='；'.join('%s 平均差 %+.6f，胜平负 %d/%d/%d'%(v,x['mean'],x['wins'],x['ties'],x['losses']) for v,x in stats.items())
        s.update('P3','COMPLETE','54 图开发完成：'+summary+'；入选：'+str(selected)+'。',
            next_action='完成独立 CaseVO 集成；入选候选进入固定确认。' if selected else '按门槛跳过确认/真实API，完成独立集成与最终审计。',evidence='study/summaries/dev.json; study/development-decision.json')
        if not selected:
            for step in ('P4','P5'):s.update(step,'SKIPPED_GATE','无候选通过预先冻结的开发门槛；不扩大样本、不付费追分。',evidence='study/development-decision.json')
    elif pool in ('confirm','stress'):
        paths=[OUT/'summaries'/('confirm.json'),OUT/'summaries'/('stress.json')]
        if all(path.exists() for path in paths):
            selected=s.read(OUT/'development-decision.json')['selected']
            confirm=s.read(paths[0])['comparisons'][selected];stress=s.read(paths[1])['comparisons'][selected]
            eligible=confirm['mean']>0 and stress['mean']>=0 and passes_safety(confirm,selected) and passes_safety(stress,selected)
            decision=dict(candidate=selected,eligible_real=eligible,
                confirmed_local=eligible and confirm['ci95_t'][0]>0 and confirm['ci95_family_bootstrap'][0]>0,
                confirm_mean=confirm['mean'],stress_mean=stress['mean'],code=check()['code'])
            s.write(OUT/'local-decision.json',decision,True)
            s.update('P4','COMPLETE','216 确认/54 压力完成；均差 %+.6f / %+.6f；真实验收资格 %s。'%
                (confirm['mean'],stress['mean'],eligible),next_action='准备同模型真实 LLM 队列。' if eligible else '跳过真实 API，完成最终审计。',evidence='study/local-decision.json')
            if not eligible:s.update('P5','SKIPPED_GATE','完整确认/压力不满足原门槛，不调用付费模型。',evidence='study/local-decision.json')
    print(json.dumps(dict(pool=pool,graphs=result['graphs'],comparisons={v:{k:x[k] for k in ('mean','wins','ties','losses','ci95_t','prefix_drift','nontriggered_different','unexplained_divergences','complete_rates')} for v,x in stats.items()}),ensure_ascii=False),flush=True)
    return result


def run(pool,start,stop):
    p=check();allowed=variants(pool)
    if pool=='real':raise RuntimeError('Use the separately frozen real-model runner')
    if not 0<=start<stop<=len(p['seeds'][pool]):raise ValueError('Invalid fixed block')
    if pool in ('confirm','stress') and not (OUT/'summaries/integration.json').exists():raise RuntimeError('Integration cohort must pass first')
    step='P3' if pool in ('dev','integration') else 'P4'
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        pending={};cursor=start
        while cursor<min(stop,start+2):pending[executor.submit(group,pool,cursor)]=cursor;cursor+=1
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                pending.pop(future);item=future.result()
                complete=sum(all((OUT/'runs'/pool/(e['spec']['case']+'-'+v+'.json')).exists() and
                    s.read(OUT/'runs'/pool/(e['spec']['case']+'-'+v+'.json'))['status']=='PASS' for v in allowed) for e in p['seeds'][pool])
                s.update(step,'RUNNING','%s 已持久化 %d/%d 完整配对图；最近 %s。'%(pool,complete,len(p['seeds'][pool]),item['case']),
                    next_action='同一命令可复用完整局；未完成/失败记录先检查，不盲重跑。',evidence='study/runs/'+pool)
                print(json.dumps(dict(pool=pool,completed=complete,**item)),flush=True)
                if cursor<stop:pending[executor.submit(group,pool,cursor)]=cursor;cursor+=1
    if all((OUT/'runs'/pool/(e['spec']['case']+'-'+v+'.json')).exists() for e in p['seeds'][pool] for v in allowed):return summarize(pool)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['benchmark','prepare','run','summary'])
    ap.add_argument('--pool',choices=POOLS,default='dev');ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int)
    a=ap.parse_args()
    try:
        with ExclusiveBatch(s.OUT/'execution.lock'):
            if a.stage=='benchmark':benchmark()
            elif a.stage=='prepare':prepare()
            elif a.stage=='summary':summarize(a.pool)
            else:run(a.pool,a.start,a.stop if a.stop is not None else len(check()['seeds'][a.pool]))
    except BaseException as exc:
        s.update('P3' if a.pool in ('dev','integration') else 'P4','BLOCKED',type(exc).__name__+': '+str(exc),
            overall='BLOCKED',next_action='检查失败记录及存活进程；不删除失败记录或重跑完成局。')
        raise

if __name__=='__main__':main()
