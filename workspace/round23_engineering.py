"""Round23 fixed R1 engineering protocol: original vs SF1 vs E1/E2."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json,statistics,sys,time
from pathlib import Path
import round23_state as s
import round21_common as h
from round9_lock import ExclusiveBatch
OUT=s.OUT/'engineering'
VARIANTS=('ORIG','SF1','E1','E2')
DEPS=('workspace/round23_engineering.py','workspace/round23_e1.py','workspace/round23_e2.py',
      'tests/test_round23_engineering.py','workspace/round22_stablefast.py',
      'workspace/round22_timing.py','workspace/round22_causal.py','workspace/round21_common.py')
EXTRA_STRESS=(1,7,13,19,26,34,42,52)
ORDERS=(('ORIG','SF1','E1','E2'),('E2','E1','SF1','ORIG'),('SF1','E2','ORIG','E1'),
        ('E1','ORIG','E2','SF1'),('ORIG','E1','SF1','E2'))

def source_files(variant):
    files=s.archive_files()
    if variant=='SF1':
        files['starnet_model.py']+=b'\n\n'+(s.ROOT/'workspace/round22_stablefast.py').read_bytes()
    elif variant=='E1':
        files['starnet_model.py']+=b'\n\n'+(s.ROOT/'workspace/round23_e1.py').read_bytes()
    elif variant=='E2':
        for name in ('workspace/round23_e1.py','workspace/round23_e2.py'):
            files['starnet_model.py']+=b'\n\n'+(s.ROOT/name).read_bytes()
    elif variant!='ORIG':raise ValueError(variant)
    compile(files['starnet_model.py'],'starnet_model.py','exec');return files

def load(variant,namespace='default'):
    name='r23_'+namespace+'_'+variant
    if name not in sys.modules:
        folder=OUT/'packages'/variant/'team_submission'
        for n,b in source_files(variant).items():
            p=folder/n;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                if p.read_bytes()!=b:raise RuntimeError('Staged source drift '+str(p))
            else:p.write_bytes(b)
        spec=importlib.util.spec_from_file_location(name,folder/'starnet_model.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name]

def clear(m):
    for n in ('coefficients','pc2_coefficients','r22_root_rows','r23_e1_plan','r23_e2_plan'):
        if hasattr(m,n):getattr(m,n).cache_clear()

def code():
    return dict(archive=s.ARCHIVE_SHA,deps={x:s.sha(s.ROOT/x) for x in DEPS},
        packages={v:{n:hashlib.sha256(b).hexdigest() for n,b in source_files(v).items()} for v in VARIANTS})

def restore_state(row,operation,module):
    clear(module)
    with h.c.offline():
        _,raw,logged,model=h.c.restore(row,operation,module)
        state=model.completed_state();eligible=sorted(model.local_nodes)
    budget=raw.get_remaining_budget()
    slots=min(model.operation_limit-1-model.operation_count,model.llm_counter.safe_limit-model.llm_counter.call_count)
    clear(module);return state,eligible,budget,slots

def build_fixtures():
    m=load('ORIG','fixture');rows=[]
    tq=s.read(s.ROOT/'agent_notes/experiments/round22/timing/protocol.json')
    for e in tq['entries']:
        row=s.read(s.ROOT/e['archived_path']);state,eligible,budget,slots=restore_state(row,44,m)
        rows.append(dict(name=e['spec']['case']+'-op44',source=e['archived_path'],source_sha=s.sha(s.ROOT/e['archived_path']),
                         operation=44,state=state,eligible=eligible,budget=budget,slots=slots,kind='timing'))
    cq=s.read(s.ROOT/'agent_notes/experiments/round22/causal/protocol.json')
    for e in cq['events']:
        row=s.read(s.ROOT/e['source_path']);state,eligible,budget,slots=restore_state(row,e['operation'],m)
        rows.append(dict(name=e['case']+'-event',source=e['source_path'],source_sha=s.sha(s.ROOT/e['source_path']),
                         operation=e['operation'],state=state,eligible=eligible,budget=budget,slots=slots,kind='causal'))
    stress=h.specs('stress')
    for i in EXTRA_STRESS:
        spec=stress[i];path=s.ROOT/'agent_notes/experiments/round21/study/runs/stress'/(spec['case']+'-PC2.json');row=s.read(path)
        rec=next((x for x in row['candidate_records'] if 'budget20' in x.get('tags',[])),None)
        if rec is None:raise RuntimeError('Missing budget20 record '+spec['case'])
        state,eligible,budget,slots=restore_state(row,rec['operation'],m)
        rows.append(dict(name=spec['case']+'-late',source=path.relative_to(s.ROOT).as_posix(),source_sha=s.sha(path),
                         operation=rec['operation'],state=state,eligible=eligible,budget=budget,slots=slots,kind='stress-extra'))
    clear(m)
    if len(rows)!=48 or len({x['name'] for x in rows})!=48:raise RuntimeError('Expected48 distinct fixtures')
    return rows

def prepare():
    s.preserved();fixtures=build_fixtures()
    refs=[{k:v for k,v in f.items() if k not in ('state','eligible','budget','slots')} for f in fixtures]
    protocol=dict(code=code(),fixtures=refs,variants=list(VARIANTS),cycles=5,orders=[list(x) for x in ORDERS],
      perturbations=8,exactness=dict(negative=[False,True],max_gain_error=1e-12,max_roi_error=1e-12,order_exact=True),
      performance_gate=dict(median_vs_orig=1.60,min_cycle_vs_orig=1.50,median_vs_sf1=1.08),
      selection='Eligible E1/E2 with highest median vs ORIG; <=1e-8 tie prefers E1.',
      no_api=True,no_policy_games=True)
    s.write(OUT/'protocol.json',protocol,True)
    s.update('R1','RUNNING','48状态工程协议冻结：E1/E2 exactness + 5-cycle性能；R1期间API=0。',
             next_action='运行E1/E2等价性与性能门槛。',evidence='engineering/protocol.json')
    print(json.dumps(dict(prepared=True,fixtures=48,cycles=5),ensure_ascii=False),flush=True)

def check():
    p=s.read(OUT/'protocol.json')
    if p['code']!=code():raise RuntimeError('Engineering source drift')
    for f in p['fixtures']:
        if s.sha(s.ROOT/f['source'])!=f['source_sha']:raise RuntimeError('Fixture source drift '+f['name'])
    return p

def materialize():
    p=check();m=load('ORIG','materialize');rows=[]
    for ref in p['fixtures']:
        row=s.read(s.ROOT/ref['source'])
        state,eligible,budget,slots=restore_state(row,ref['operation'],m)
        rows.append(dict(ref,state=state,eligible=eligible,budget=budget,slots=slots))
    clear(m)
    if len(rows)!=48:raise RuntimeError('Materialized fixture count')
    return rows

def keys(xs):return [(x['action'],x['target_node_1'],x['target_node_2']) for x in xs]

def exactness():
    check();fixtures=materialize();orig=load('ORIG','exact-orig');mods={v:load(v,'exact-'+v.lower()) for v in ('E1','E2')}
    out={}
    for v,m in mods.items():
        mg=mr=0.;count=0
        for f in fixtures:
            for negative in (False,True):
                a=orig.RestrictedRollout(*copy.deepcopy(f['state']),orig.SEARCH,f['slots'],eligible=set(f['eligible']))
                b=m.RestrictedRollout(*copy.deepcopy(f['state']),m.SEARCH,f['slots'],eligible=set(f['eligible']))
                xa=a.candidates(a.initial,f['budget'],negative);xb=b.candidates(b.initial,f['budget'],negative)
                if keys(xa)!=keys(xb):raise RuntimeError(v+' candidate key/order differs '+f['name'])
                if len(xa)!=len(xb):raise RuntimeError(v+' candidate count differs')
                for x,y in zip(xa,xb):
                    if x['cost']!=y['cost'] or x.get('prompt_id')!=y.get('prompt_id'):raise RuntimeError(v+' metadata differs')
                    mg=max(mg,abs(x['gain']-y['gain']));mr=max(mr,abs(x['roi']-y['roi']));count+=1
        out[v]=dict(candidate_rows=count,max_gain_error=mg,max_roi_error=mr,order_exact=True,
                    eligible=mg<=1e-12 and mr<=1e-12)
    return out

def mutate(state,r):
    st=copy.deepcopy(state);ids=sorted(st[0])
    for j,i in enumerate(ids):st[0][i]['w']+=((j%7)-3)*1e-4*(r+1)
    return st

def run_variant(v):
    m=load(v,'bench-'+v.lower());fixtures=materialize();clear(m);start=time.perf_counter();rows=0
    for f in fixtures:
        for r in range(8):
            st=mutate(f['state'],r);engine=m.RestrictedRollout(*st,m.SEARCH,f['slots'],eligible=set(f['eligible']))
            rows+=len(engine.candidates(engine.initial,f['budget']))
    sec=time.perf_counter()-start;clear(m);return dict(seconds=sec,rows=rows)

def performance():
    cycles=[]
    for ci,order in enumerate(ORDERS):
        got={}
        for v in order:got[v]=run_variant(v)
        base=got['ORIG']['seconds'];sf1=got['SF1']['seconds']
        ratios={v:base/got[v]['seconds'] for v in ('SF1','E1','E2')}
        sf1_ratios={v:sf1/got[v]['seconds'] for v in ('E1','E2')}
        cycles.append(dict(cycle=ci,order=list(order),runs=got,vs_orig=ratios,vs_sf1=sf1_ratios))
        print(json.dumps(dict(cycle=ci,vs_orig=ratios,vs_sf1=sf1_ratios),ensure_ascii=False),flush=True)
    summary={}
    for v in ('E1','E2'):
        vo=[x['vs_orig'][v] for x in cycles];vs=[x['vs_sf1'][v] for x in cycles]
        summary[v]=dict(median_vs_orig=statistics.median(vo),min_vs_orig=min(vo),median_vs_sf1=statistics.median(vs),
                        per_cycle_vs_orig=vo,per_cycle_vs_sf1=vs)
    return dict(cycles=cycles,summary=summary,rows=cycles[0]['runs']['ORIG']['rows'])

def summarize():
    ex=exactness();perf=performance();gate=check()['performance_gate'];eligible=[]
    for v in ('E1','E2'):
        x=perf['summary'][v]
        if ex[v]['eligible'] and x['median_vs_orig']>=gate['median_vs_orig'] and x['min_vs_orig']>=gate['min_cycle_vs_orig'] and x['median_vs_sf1']>=gate['median_vs_sf1']:
            eligible.append(v)
    eligible.sort(key=lambda v:(-perf['summary'][v]['median_vs_orig'],0 if v=='E1' else 1))
    selected=eligible[0] if eligible else None
    rec=dict(status='PASS',exactness=ex,performance=perf,eligible=eligible,selected=selected,gate=gate,no_api=True)
    s.write(OUT/'summary.json',rec,True)
    detail='；'.join('%s exact=%s medOrig=%.3fx minOrig=%.3fx medSF1=%.3fx'%(v,ex[v]['eligible'],perf['summary'][v]['median_vs_orig'],perf['summary'][v]['min_vs_orig'],perf['summary'][v]['median_vs_sf1']) for v in ('E1','E2'))
    if selected:
        s.update('R1','COMPLETE',detail+'；选中 '+selected+'。',next_action='进入R2冻结SF2/B2B-SF2包并做边界测试。',evidence='engineering/summary.json')
    else:
        s.update('R1','STOPPED_GATE',detail+'；无候选达到冻结性能门槛。',next_action='跳过R2-R5，进入R6回归审计。',evidence='engineering/summary.json')
        for k in ('R2','R3','R4','R5'):s.update(k,'SKIPPED_GATE','R1工程性能门槛未通过。',evidence='engineering/summary.json')
    print(json.dumps(dict(selected=selected,eligible=eligible,exactness=ex,performance=perf['summary']),ensure_ascii=False),flush=True);return rec

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run']);a=ap.parse_args()
    with ExclusiveBatch(s.OUT/'execution.lock'):
        prepare() if a.stage=='prepare' else summarize()
