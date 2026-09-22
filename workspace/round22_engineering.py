"""Round22 StableFast exactness/performance gate."""
from __future__ import annotations
import copy,hashlib,importlib.util,json,statistics,sys,time
from pathlib import Path
import round22_state as s
import round21_common as h
import round22_causal as causal
import round22_timing as timing
from round9_lock import ExclusiveBatch
OUT=s.OUT/'stablefast'
DEPS=('workspace/round22_stablefast.py','workspace/round22_engineering.py','tests/test_round22_stablefast.py')
def source_files():
    files=s.archive_files();files['starnet_model.py']+=b'\n\n'+(s.ROOT/'workspace/round22_stablefast.py').read_bytes()
    compile(files['starnet_model.py'],'starnet_model.py','exec');return files
def load(namespace='default'):
    name='r22_sf_'+namespace
    if name not in sys.modules:
        folder=s.OUT/'packages/SF/team_submission'
        for n,b in source_files().items():
            p=folder/n;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                if p.read_bytes()!=b:raise RuntimeError('SF staged source drift')
            else:p.write_bytes(b)
        spec=importlib.util.spec_from_file_location(name,folder/'starnet_model.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name]
def code():
    return dict(archive=s.ARCHIVE_SHA,deps={x:s.sha(s.ROOT/x) for x in DEPS},
                sf={n:hashlib.sha256(b).hexdigest() for n,b in source_files().items()})
def public_state(row,operation,module):
    h.clear(module)
    with h.c.offline():
        _,raw,logged,model=h.c.restore(row,operation,module)
        state=model.completed_state();eligible=set(model.local_nodes)
    return state,eligible,raw.get_remaining_budget(),min(model.operation_limit-1-model.operation_count,model.llm_counter.safe_limit-model.llm_counter.call_count)
def fixtures():
    old=h.load('PC2','r22-fixture')
    result=[]
    tq=timing.check()
    for e in tq['entries'][:12]:
        row=s.read(s.ROOT/e['archived_path']);state,eligible,budget,slots=public_state(row,44,old)
        result.append(dict(name=e['spec']['case']+'-op44',state=state,eligible=sorted(eligible),budget=budget,slots=slots))
    cq=causal.check()
    for e in cq['events'][:12]:
        row=s.read(s.ROOT/e['source_path']);state,eligible,budget,slots=public_state(row,e['operation'],old)
        result.append(dict(name=e['case']+'-event',state=state,eligible=sorted(eligible),budget=budget,slots=slots))
    h.clear(old)
    if len(result)!=24:raise RuntimeError('Expected24 engineering fixtures')
    return result
def keys(xs):return [(x['action'],x['target_node_1'],x['target_node_2']) for x in xs]
def exactness():
    m=load('exact');fs=fixtures();max_gain=max_roi=0.;checked=0
    for f in fs:
        for negative in (False,True):
            a=m._R22OriginalRestrictedRollout(*copy.deepcopy(f['state']),m.SEARCH,f['slots'],eligible=set(f['eligible']))
            b=m.R22StableRestricted(*copy.deepcopy(f['state']),m.SEARCH,f['slots'],eligible=set(f['eligible']))
            xa=a.candidates(a.initial,f['budget'],negative);xb=b.candidates(b.initial,f['budget'],negative)
            if keys(xa)!=keys(xb):raise RuntimeError('Candidate order/key differs '+f['name'])
            if len(xa)!=len(xb):raise RuntimeError('Candidate count differs')
            for x,y in zip(xa,xb):
                max_gain=max(max_gain,abs(x['gain']-y['gain']));max_roi=max(max_roi,abs(x['roi']-y['roi']))
                if abs(x['cost']-y['cost'])>0 or x.get('prompt_id')!=y.get('prompt_id'):raise RuntimeError('Candidate metadata differs')
            checked+=len(xa)
    return dict(fixtures=len(fs),candidate_rows=checked,max_gain_error=max_gain,max_roi_error=max_roi,
                order_exact=True,eligible=max_gain<=1e-10 and max_roi<=1e-10)
def mutate(state,round_index):
    st=copy.deepcopy(state)
    ids=sorted(st[0])
    for j,i in enumerate(ids):
        st[0][i]['w'] += ((j%7)-3)*1e-4*(round_index+1)
    return st
def run_benchmark(module,fast,rounds=8):
    if hasattr(module,'coefficients'):module.coefficients.cache_clear()
    if hasattr(module,'r22_root_rows'):module.r22_root_rows.cache_clear()
    cls=module.R22StableRestricted if fast else module.RestrictedRollout
    total=0.;rows=0;start=time.perf_counter()
    for f in fixtures():
        for r in range(rounds):
            st=mutate(f['state'],r);engine=cls(*st,module.SEARCH,f['slots'],eligible=set(f['eligible']))
            rows+=len(engine.candidates(engine.initial,f['budget']))
    total=time.perf_counter()-start
    return dict(seconds=total,rows=rows)
def benchmark():
    old=h.load('PC2','r22-bench-old');sf=load('bench')
    # Separate modules/caches; alternating repetitions reduce one-off process effects.
    samples=[]
    for cycle in range(3):
        order=('old','fast') if cycle%2==0 else ('fast','old')
        got={}
        for kind in order:
            got[kind]=run_benchmark(old,False) if kind=='old' else run_benchmark(sf,True)
        samples.append(got)
    old_secs=statistics.median(x['old']['seconds'] for x in samples)
    fast_secs=statistics.median(x['fast']['seconds'] for x in samples)
    return dict(cycles=samples,old_seconds=old_secs,fast_seconds=fast_secs,speedup=old_secs/fast_secs if fast_secs else float('inf'),
                rows=samples[0]['old']['rows'])
def main():
    s.preserved()
    ex=exactness();perf=benchmark();eligible=ex['eligible'] and perf['speedup']>=1.5
    rec=dict(status='PASS',code=code(),exactness=ex,performance=perf,eligible_sf=eligible,
             gate='max gain/ROI error<=1e-10, candidate key/order exact, median fixed benchmark speedup>=1.5x',
             no_policy_cohort=True,no_api=True)
    s.write(OUT/'summary.json',rec,True)
    s.update('R3','COMPLETE' if eligible else 'STOPPED_GATE','StableFast等价：order=%s gainerr %.3g；固定基准 %.3fx；工程资格%s。'%(
        ex['order_exact'],ex['max_gain_error'],perf['speedup'],eligible),
        next_action='冻结SF与B2B-SF，进入全新54图开发。' if eligible else '等价/性能门槛未过；跳过R4并完成回归审计。',
        evidence='stablefast/summary.json')
    if not eligible:s.update('R4','SKIPPED_GATE','StableFast等价/性能门槛未通过。',evidence='stablefast/summary.json')
    print(json.dumps(rec,ensure_ascii=False),flush=True)
if __name__=='__main__':
    with ExclusiveBatch(s.OUT/'execution.lock'):main()
