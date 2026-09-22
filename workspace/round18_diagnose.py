"""Frozen 12-graph PC2 cut-coverage diagnosis, public-prefix paired branches and profiling."""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import copy
import cProfile
import io
import json
import pstats
import statistics
import time
import round18_state as s
import round18_common as h
from round9_lock import ExclusiveBatch

OUT=s.OUT/'diagnostic'
DEPS=('workspace/round18_diagnose.py','tests/test_round18_core.py')


def prepare():
    s.preserved();h.stage('PC2')
    rows=h.entries('diagnostic')
    if len({r['sha256'] for r in rows})!=12: raise RuntimeError('Diagnostic seed collision')
    protocol=dict(code=h.fingerprint(DEPS),seeds=rows,graphs=12,workers=2,
        snapshots=['first budget<=20','first budget<=10'],
        candidates='PC2 chosen action and best existing legal positive-gain cut. Same candidates: record no alternative.',
        continuation='Force root once; same PC2 suffix with consumed original window tags restored. All branches start same public prefix/resources. No hidden attributes reach model.',
        cut1_gate='>=3 states with cut suffix gain>0.5, spanning>=2 graphs. No sample extension or tuning.',
        profiling='First six cases (one per family), first budget<=20 snapshot. Eight PC2 private candidate enumerations; public copied weights perturbed by a fixed tiny offset. Runtime profile only, not score evidence.',
        interpretation='Correlated states, diagnostic realized-world opportunity, not deployable average gain. Soft deadlines can cause suffix drift; report it, never filter cases.',
        no_api=True,no_upload=True)
    s.write(OUT/'protocol.json',protocol,True)
    s.update('P1','RUNNING','12 图诊断协议已冻结；每个整局/分支独立落盘。',
        next_action='运行 .\\.venv\\python.exe -X utf8 -B workspace/round18_diagnose.py run --start 0 --stop 12',
        evidence='diagnostic/protocol.json')
    print(json.dumps(dict(prepared=True,graphs=12)))


def check():
    p=s.read(OUT/'protocol.json')
    if p['code']!=h.fingerprint(DEPS): raise RuntimeError('Frozen diagnostic source changed')
    return p


def profile(base,snapshot):
    module,raw,logged,model=h.restore(base,snapshot,'profile')
    state=model.completed_state();slots=min(model.operation_limit-1-model.operation_count,
        model.llm_counter.safe_limit-model.llm_counter.call_count)
    engine=module.PC2Rollout(*state,module.SEARCH,slots,eligible=model.local_nodes)
    timer=cProfile.Profile();durations=[];counts=[]
    timer.enable()
    for rep in range(8):
        belief=module.BudgetRollout.clone(state)
        for i,d in belief[0].items(): d['w'] += (rep*.001 if i in model.local_nodes else 0.)
        start=time.perf_counter();actions=engine.candidates(belief,snapshot['budget'])
        durations.append(time.perf_counter()-start);counts.append(len(actions))
    timer.disable();stream=io.StringIO()
    pstats.Stats(timer,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(25)
    result=dict(status='PASS',case=base['spec']['case'],durations=durations,candidate_counts=counts,
        profiler=stream.getvalue(),cache_info=module.pc2_coefficients.cache_info()._asdict(),
        scope='Timing-only public-state benchmark, no policy gain claim')
    h.clear(module)
    return result


def run_graph(index):
    p=check();entry=p['seeds'][index];spec=entry['spec']
    if h.c.digest(h.c.generate(spec))!=entry['sha256']: raise RuntimeError('Seed mismatch')
    identity=dict(entry=entry,code=p['code'])
    base=h.checked_record(OUT/'baseline'/(spec['case']+'.json'),identity,lambda:h.run_game(spec,capture=True))
    snapshots=[r for r in base['states'] if any(t in r['tags'] for t in ('budget20','budget10'))]
    rows=[]
    for j,snapshot in enumerate(snapshots):
        with h.c.offline():
            module,raw,logged,model=h.restore(base,snapshot,'candidate')
            state=model.completed_state()
            slots=min(model.operation_limit-1-model.operation_count,model.llm_counter.safe_limit-model.llm_counter.call_count)
            engine=module.RestrictedRollout(*state,module.SEARCH,slots,eligible=model.local_nodes)
            cut=next((a for a in engine.candidates(state,snapshot['budget']) if a['action']=='cut'),None)
            same=cut is not None and module.BudgetRollout.key(cut)==module.BudgetRollout.key(snapshot['chosen'])
            h.clear(module)
        item=dict(snapshot=snapshot,cut=cut,reason='NO_LEGAL_CUT' if cut is None else 'SAME_ROOT' if same else 'COMPARED')
        if cut is not None and not same:
            branches={}
            actions={'incumbent':snapshot['chosen'],'cut':cut}
            for kind in (('incumbent','cut') if (index+j)%2==0 else ('cut','incumbent')):
                path=OUT/'branches'/('%s-%d-%s.json'%(spec['case'],j,kind))
                ident=dict(identity,snapshot=snapshot,kind=kind,action=actions[kind])
                branches[kind]=h.checked_record(path,ident,lambda kind=kind:h.branch(base,snapshot,actions[kind]))
            item.update(delta=branches['cut']['final_score']-branches['incumbent']['final_score'],
                baseline_suffix_drift=branches['incumbent']['final_score']-base['final_score'],
                branch_paths={kind:('diagnostic/branches/%s-%d-%s.json'%(spec['case'],j,kind)) for kind in branches})
        rows.append(item)
        s.write(OUT/'states'/('%s-%d.json'%(spec['case'],j)),dict(status='PASS',identity=identity,**item),True)
    if index<6 and snapshots:
        h.checked_record(OUT/'profiles'/(spec['case']+'.json'),identity,lambda:profile(base,snapshots[0]))
    result=dict(status='PASS',identity=identity,spec=spec,states=rows,base_path='diagnostic/baseline/'+spec['case']+'.json')
    s.write(OUT/'graphs'/(spec['case']+'.json'),result,True)
    return dict(case=spec['case'],states=len(rows),compared=sum(r['reason']=='COMPARED' for r in rows))


def summarize():
    p=check();rows=[s.read(OUT/'graphs'/(e['spec']['case']+'.json')) for e in p['seeds']]
    states=[(row,item) for row in rows for item in row['states']]
    if any(row['status']!='PASS' for row in rows): raise RuntimeError('Incomplete diagnosis')
    wins=[(row,item) for row,item in states if item.get('delta',0.)>.5]
    graph_wins={row['spec']['case'] for row,item in wins}
    result=dict(status='PASS',graphs=len(rows),states=len(states),
        compared=sum(item['reason']=='COMPARED' for row,item in states),
        missing_reasons=dict(collections.Counter(item['reason'] for row,item in states)),
        opportunity_states=len(wins),opportunity_graphs=len(graph_wins),
        eligible_cut1=len(wins)>=3 and len(graph_wins)>=2,
        graph_mean_opportunity=statistics.mean(statistics.mean(max(0.,item.get('delta',0.)) for item in row['states']) if row['states'] else 0. for row in rows),
        suffix_drift_states=sum(abs(item.get('baseline_suffix_drift',0.))>1e-8 for row,item in states),
        graph_details=[dict(case=row['spec']['case'],family=row['spec']['family'],
            deltas=[item.get('delta') for item in row['states']]) for row in rows],
        scope='Forced-branch realized-world diagnostic; not implementable expected improvement or new confirmation.')
    s.write(OUT/'summary.json',result,True)
    s.update('P1','COMPLETE','12 图诊断完成：%d/%d 状态存在 >0.5 断边机会，跨 %d 图；原后缀漂移 %d。'%
        (result['opportunity_states'],result['states'],result['opportunity_graphs'],result['suffix_drift_states']),
        next_action='根据冻结门槛实施 Cut1（若允许）；分析固定六状态剖析并实现独立 Fast。',evidence='diagnostic/summary.json; diagnostic/profiles/')
    s.update('P2A','PENDING' if result['eligible_cut1'] else 'SKIPPED_GATE',
        '断边诊断达到预先门槛，待实现。' if result['eligible_cut1'] else '断边诊断未达到预先门槛，不追改阈值、不扩大本轮样本。',evidence='diagnostic/summary.json')
    print(json.dumps({k:v for k,v in result.items() if k!='graph_details'},ensure_ascii=False),flush=True)
    return result


def run(start,stop):
    p=check()
    if not 0<=start<stop<=12: raise ValueError('Invalid block')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        pending={};cursor=start
        while cursor<min(stop,start+2):
            pending[executor.submit(run_graph,cursor)]=cursor;cursor+=1
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                pending.pop(future);item=future.result()
                complete=sum((OUT/'graphs'/(e['spec']['case']+'.json')).exists() for e in p['seeds'])
                s.update('P1','RUNNING','诊断已持久化 %d/12 图；最近完成 %s。'%(complete,item['case']),
                    next_action='同一 run 命令可复用完成图；异常记录先检查。',evidence='diagnostic/graphs/'+item['case']+'.json')
                print(json.dumps(dict(completed=complete,**item)),flush=True)
                if cursor<stop:
                    pending[executor.submit(run_graph,cursor)]=cursor;cursor+=1
    if all((OUT/'graphs'/(e['spec']['case']+'.json')).exists() for e in p['seeds']): return summarize()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary'])
    ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=12);a=ap.parse_args()
    try:
        with ExclusiveBatch(s.OUT/'execution.lock'):
            if a.stage=='prepare':prepare()
            elif a.stage=='run':run(a.start,a.stop)
            else:summarize()
    except BaseException as exc:
        s.update('P1','BLOCKED',type(exc).__name__+': '+str(exc),overall='BLOCKED',
            next_action='检查失败记录和运行进程，保留已完成结果；不要盲重跑。')
        raise

if __name__=='__main__':main()
