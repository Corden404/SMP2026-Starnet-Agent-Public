"""Round20 Tail Bundle historical diagnostic on the frozen Round19 primary states."""
from __future__ import annotations
import argparse, collections, concurrent.futures, copy, json, os, statistics, time
from pathlib import Path
import round20_state as s
import round19_common as h
from round9_lock import ExclusiveBatch

OUT=s.OUT/'bundle-diagnostic'
OLD=s.ROOT/'agent_notes/experiments/round19/diagnostic/states'
DEPS=('workspace/round20_bundle_diagnose.py','tests/test_round20_bundle.py',
      'workspace/round20_state.py','workspace/round19_common.py','workspace/round19_numeric.py')
BASE_SELECTED=0.08384218455260183
BASE_REGRET=0.6258904507285521

def fingerprint():
    return dict(archive=s.ARCHIVE_SHA,dependencies={x:s.sha(s.ROOT/x) for x in DEPS})

def action_key(a): return (a['action'],a['target_node_1'],a['target_node_2'])

def checked(path,identity,operation):
    path=Path(path)
    if path.exists():
        row=s.read(path)
        if row.get('status')!='PASS' or row.get('identity')!=identity:
            raise RuntimeError('Existing incomplete/different result requires inspection: '+str(path))
        return row
    row=dict(status='STARTED',identity=identity,pid=os.getpid(),started_at=s.now())
    s.write(path,row,True)
    try:
        result=operation();row.update(result,status='PASS',identity=identity,completed_at=s.now());s.write(path,row);return row
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),failed_at=s.now());s.write(path,row);raise

def inventory():
    rows=[]
    for path in sorted(OLD.glob('*.json')):
        row=s.read(path)
        if row.get('status')!='PASS' or not row.get('primary') or row.get('reason')!='COMPARED':continue
        entry=row['identity']['entry']
        if not entry.get('primary'):continue
        inc_path=s.ROOT/row['branch_paths']['incumbent']['path']
        src=s.ROOT/entry['source_path']
        rows.append(dict(id=row['entry_id'],case=row['case'],family=row['family'],
          state_path=path.relative_to(s.ROOT).as_posix(),state_sha256=s.sha(path),
          source_path=entry['source_path'],source_sha256=s.sha(src),
          incumbent_path=inc_path.relative_to(s.ROOT).as_posix(),incumbent_sha256=s.sha(inc_path),
          snapshot=row['snapshot'],incumbent=row['incumbent'],old_best_cut_delta=row['actual_delta']))
    if len(rows)!=132 or len({x['id'] for x in rows})!=132:raise RuntimeError('Expected132 frozen primary states')
    return rows

def prepare():
    s.preserved();entries=inventory()
    protocol=dict(code=fingerprint(),entries=entries,states=132,historical_graphs=66,workers=2,
      candidate_labels=['C1','C2','S1','S2','C1-C','C1-S','S1-C'],max_prefix=2,
      rule='Structural immediate gain>0.25; top2 per type by gain. Full-S44 3-world continuation. Whole-state soft deadline12s. Select only if every generated plan and baseline complete and best mean delta>=0.5.',
      realization='At most one selected structural prefix is forced on the frozen historical environment, then exact archived PC2 continues. Incumbent branch is reused from Round19.',
      gate=dict(coverage_min=.95,graph_realized_gt=BASE_SELECTED,graph_regret_lt=BASE_REGRET,positive_graphs_min=3,positive_families_min=2),
      baselines=dict(round19_depth2_selected=BASE_SELECTED,round19_depth2_regret=BASE_REGRET),
      no_api=True,no_training=True,no_upload=True)
    s.write(OUT/'protocol.json',protocol,True)
    s.update('R1B','RUNNING','132个历史primary状态 Tail Bundle 协议已冻结；每状态最多7个公开belief前缀、最多1条新realized分支。',
      next_action='运行 round20_bundle_diagnose.py run --start 0 --stop 132。',evidence='bundle-diagnostic/protocol.json')
    print(json.dumps(dict(prepared=True,states=132,max_realized_branches=132),ensure_ascii=False),flush=True)

def check():
    p=s.read(OUT/'protocol.json')
    if p['code']!=fingerprint():raise RuntimeError('Bundle diagnostic source drift')
    return p

def top_structural(module,state,budget,eligible,kind,n=2):
    slots=999
    engine=module.R19Kernel(*state,module.SEARCH,slots,eligible=eligible)
    xs=[a for a in engine.candidates(state,budget) if a['action']==kind and a['gain']>.25]
    xs.sort(key=lambda a:(-a['gain'],-a['roi'],a['target_node_1'],a['target_node_2'] or -1))
    return xs[:n]

def structural_apply(module,state,action,eligible):
    engine=module.R19Kernel(*state,module.SEARCH,999,eligible=eligible)
    return engine.apply(state,action,{},False)

def bundle_candidates(module,state,budget,eligible):
    cuts=top_structural(module,state,budget,eligible,'cut',2)
    shields=top_structural(module,state,budget,eligible,'shield',2)
    plans=[]
    if cuts:plans.append(('C1',[cuts[0]]))
    if len(cuts)>1:plans.append(('C2',[cuts[1]]))
    if shields:plans.append(('S1',[shields[0]]))
    if len(shields)>1:plans.append(('S2',[shields[1]]))
    if cuts:
        first=cuts[0];post=structural_apply(module,state,first,eligible);remain=budget-first['cost']
        nxt=top_structural(module,post,remain,eligible,'cut',1)
        if nxt:plans.append(('C1-C',[first,nxt[0]]))
        nxt=top_structural(module,post,remain,eligible,'shield',1)
        if nxt:plans.append(('C1-S',[first,nxt[0]]))
    if shields:
        first=shields[0];post=structural_apply(module,state,first,eligible);remain=budget-first['cost']
        nxt=top_structural(module,post,remain,eligible,'cut',1)
        if nxt:plans.append(('S1-C',[first,nxt[0]]))
    out=[];seen=set()
    for label,prefix in plans:
        key=tuple(action_key(a) for a in prefix)
        if key not in seen and sum(a['cost'] for a in prefix)<=budget:
            seen.add(key);out.append((label,prefix))
    if len(out)>7:raise RuntimeError('Bundle candidate cap exceeded')
    return out

def prefix_continue(module,state,budget,eligible,ctx,prefix,world,deadline):
    st=module.BudgetRollout.clone(state);c=dict(ctx,seen=set(ctx['seen']))
    engine=module.R19Kernel(*st,module.SEARCH,module.r19_slots(c),eligible=eligible)
    actions=[]
    for a in prefix:
        module.r19_check(deadline);module.r19_legal(st,a,budget,module.r19_slots(c),eligible)
        st=engine.apply(st,a,world,True);budget-=a['cost'];c['operation']+=1;c['calls']+=1
        actions.append(module.BudgetRollout.key(a))
    if budget<2 or module.r19_slots(c)<=0:
        return dict(score=module.r18_cut_score(st[0],st[1])[0],actions=actions,budget=budget)
    first=module.r19_s44(st,budget,module.r19_slots(c),eligible,deadline,strict=False)
    if first is None:return dict(score=module.r18_cut_score(st[0],st[1])[0],actions=actions,budget=budget)
    suffix=module.r19_continue(st,budget,eligible,c,first,world,deadline,mode='full_s44')
    suffix['actions']=actions+suffix['actions'];return suffix

def forecast(module,state,budget,eligible,ctx,incumbent,plans,seconds=12.):
    deadline=time.monotonic()+seconds;worlds=module.pc2_worlds(state)
    result=dict(baseline=None,plans=[],all_complete=True)
    try:
        base=[]
        for world in worlds:
            base.append(module.r19_continue(state,budget,eligible,ctx,incumbent,world,deadline,mode='full_s44')['score'])
        result['baseline']=base
    except TimeoutError:
        result.update(all_complete=False,reason='baseline_timeout');return result
    for label,prefix in plans:
        values=[]
        try:
            for world,base in zip(worlds,result['baseline']):
                got=prefix_continue(module,state,budget,eligible,ctx,prefix,world,deadline)
                values.append(dict(score=got['score'],delta=got['score']-base,actions=got['actions']))
            mean=sum(x['delta'] for x in values)/3
            result['plans'].append(dict(label=label,prefix=[list(action_key(a)) for a in prefix],
                complete=True,mean_delta=mean,min_delta=min(x['delta'] for x in values),worlds=values))
        except TimeoutError:
            result['plans'].append(dict(label=label,prefix=[list(action_key(a)) for a in prefix],complete=False))
            result['all_complete']=False
            break
    if not result['all_complete']:
        result['reason']='candidate_timeout';return result
    best=max(result['plans'],key=lambda x:x['mean_delta'],default=None)
    if best is None or best['mean_delta']<.5:
        result.update(reason='no_margin',selected=None);return result
    result.update(reason='selected',selected=best);return result

def force_bundle(row,snap,prefix):
    module=h.load('PC2','r20-force');h.clear(module)
    with h.old.c.offline():
        _,raw,logged,model=h.old.c.restore(row,snap['operation'],module)
        model._pc2_init();model._pc2_seen=set(snap['pc2_seen'])
        if raw.get_remaining_budget()!=snap['budget']:raise RuntimeError('Bundle restore budget mismatch')
        for a in prefix:h.old.c.force(model,raw,a)
        result=h.old.c.finish(raw,logged,model)
    result.update(spec=row['spec'],variant='PC2-forced-bundle',mode='diagnostic',start_operation=snap['operation'],
      forced_actions=copy.deepcopy(prefix),prefix_sha256=s.digest(row['actions'][:snap['operation']]))
    h.old.c.audit_game(result);h.clear(module);return result

def run_state(index):
    p=check();entry=p['entries'][index];ident=dict(entry=entry,code=p['code'])
    path=OUT/'states'/(entry['id']+'.json')
    def execute():
        if s.sha(s.ROOT/entry['state_path'])!=entry['state_sha256'] or s.sha(s.ROOT/entry['source_path'])!=entry['source_sha256']:
            raise RuntimeError('Historical state/source drift')
        old_state=s.read(s.ROOT/entry['state_path']);row=s.read(s.ROOT/entry['source_path']);snap=entry['snapshot']
        state,eligible,ctx=h.public_at(row,snap);module=h.load('Numeric','r20-bundle');h.clear(module)
        plans=bundle_candidates(module,state,snap['budget'],eligible)
        with h.old.c.offline():pred=forecast(module,state,snap['budget'],eligible,ctx,entry['incumbent'],plans,12.)
        out=dict(entry_id=entry['id'],case=entry['case'],family=entry['family'],snapshot=snap,
          incumbent=entry['incumbent'],candidate_count=len(plans),forecast=pred,old_best_cut_delta=entry['old_best_cut_delta'])
        if pred.get('selected') is not None:
            lookup={label:prefix for label,prefix in plans};label=pred['selected']['label'];prefix=lookup[label]
            rp=OUT/'branches'/(entry['id']+'-'+label+'.json')
            branch=checked(rp,dict(ident,label=label,prefix=[list(action_key(a)) for a in prefix]),lambda:force_bundle(row,snap,prefix))
            if s.sha(s.ROOT/entry['incumbent_path'])!=entry['incumbent_sha256']:raise RuntimeError('Incumbent branch drift')
            inc=s.read(s.ROOT/entry['incumbent_path']);h.old.c.audit_game(inc)
            realized=branch['final_score']-inc['final_score']
            out.update(realized_delta=realized,branch_path=rp.relative_to(s.ROOT).as_posix(),
              incumbent_final=inc['final_score'],bundle_final=branch['final_score'])
        h.clear(module);return out
    row=checked(path,ident,execute);return dict(id=entry['id'],selected=row['forecast'].get('selected') is not None)

def summarize():
    p=check();rows=[]
    for e in p['entries']:
        r=s.read(OUT/'states'/(e['id']+'.json'))
        if r.get('status')!='PASS' or r.get('identity')!=dict(entry=e,code=p['code']):raise RuntimeError('Incomplete/mixed bundle state')
        rows.append(r)
    complete=[r for r in rows if r['forecast'].get('all_complete')]
    bycase=collections.defaultdict(list)
    for r in rows:bycase[r['case']].append(r)
    graph_selected=[];graph_regret=[];positive_graphs=[];families=set()
    for case,rs in bycase.items():
        vals=[r.get('realized_delta',0.) for r in rs]
        selected=statistics.mean(vals)
        regrets=[max(0.,r['old_best_cut_delta'],r.get('realized_delta',0.))-r.get('realized_delta',0.) for r in rs]
        graph_selected.append(selected);graph_regret.append(statistics.mean(regrets))
        if selected>1e-8:
            positive_graphs.append(case);families.add(rs[0]['family'])
    graph_mean=statistics.mean(graph_selected);regret=statistics.mean(graph_regret)
    coverage=len(complete)/len(rows);selected_states=sum(r['forecast'].get('selected') is not None for r in rows)
    labels=collections.Counter(r['forecast']['selected']['label'] for r in rows if r['forecast'].get('selected'))
    gate=coverage>=p['gate']['coverage_min'] and graph_mean>p['gate']['graph_realized_gt'] and regret<p['gate']['graph_regret_lt'] and len(positive_graphs)>=p['gate']['positive_graphs_min'] and len(families)>=p['gate']['positive_families_min']
    result=dict(status='PASS',states=len(rows),graphs=len(bycase),complete_states=len(complete),coverage=coverage,
      selected_states=selected_states,selected_labels=dict(labels),graph_realized_mean=graph_mean,graph_regret=regret,
      positive_graphs=len(positive_graphs),positive_families=len(families),eligible_b1=gate,
      baselines=p['baselines'],new_realized_branches=sum(1 for _ in (OUT/'branches').glob('*.json')),
      no_api=True,new_independent_graphs=0)
    s.write(OUT/'summary.json',result,True)
    s.update('R1B','COMPLETE','132状态完成；coverage %.3f；selected %d；图均realized %+.6f；regret %.6f；B1资格%s。'%(
      coverage,selected_states,graph_mean,regret,gate),
      next_action='实现并冻结B1。' if gate else '历史bundle诊断未过冻结门槛；跳过R2-R5并完成全量回归/审计。',evidence='bundle-diagnostic/summary.json')
    if not gate:
        for step in ('R2','R3','R4','R5'):s.update(step,'SKIPPED_GATE','R1B Tail Bundle 历史诊断未达到冻结门槛，不追改参数或扩样。',evidence='bundle-diagnostic/summary.json')
    print(json.dumps(result,ensure_ascii=False),flush=True);return result

def run(start,stop):
    p=check();s.preserved()
    if not 0<=start<stop<=132:raise ValueError('Invalid block')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        pending={};cur=start
        while cur<min(stop,start+2):pending[ex.submit(run_state,cur)]=cur;cur+=1
        done_count=0
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                pending.pop(f);info=f.result();done_count+=1
                complete=sum(1 for x in (OUT/'states').glob('*.json') if s.read(x).get('status')=='PASS')
                if complete%5==0 or complete==132:
                    s.update('R1B','RUNNING','Tail Bundle 已持久化 %d/132 状态；最近 %s。'%(complete,info['id']),
                      next_action='继续固定132状态；PASS复用，失败先检查。',evidence='bundle-diagnostic/states')
                print(json.dumps(dict(completed=complete,**info)),flush=True)
                if cur<stop:pending[ex.submit(run_state,cur)]=cur;cur+=1
    if all((OUT/'states'/(e['id']+'.json')).exists() for e in p['entries']):return summarize()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=132);a=ap.parse_args()
    try:
        with ExclusiveBatch(s.OUT/'execution.lock'):
            if a.stage=='prepare':prepare()
            elif a.stage=='run':run(a.start,a.stop)
            else:summarize()
    except BaseException as exc:
        s.update('R1B','BLOCKED',type(exc).__name__+': '+str(exc),overall='BLOCKED',next_action='保留失败记录，检查后显式恢复；不要覆盖PASS。')
        raise
if __name__=='__main__':main()
