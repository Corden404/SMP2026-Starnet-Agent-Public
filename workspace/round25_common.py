"""Round25 package loading and offline runner."""
from __future__ import annotations
import copy,hashlib,importlib.util,sys,time
from pathlib import Path
import round25_state as s
import round21_common as r21
import round18_common as r18
import round16_common as c
from experiment_round5 import FiniteEnv
from benchmark import LoggedEnv
ROOT=s.ROOT;FAMILIES=c.FAMILIES;SCALES=c.SCALES;VARIANTS=('PC2','B2B','B2R')
DEPS=('workspace/round25_common.py','workspace/round25_policy.py','workspace/round25_state.py','workspace/round21_policy.py','workspace/round19_numeric.py','workspace/round18_cut1.py')
def source_files(variant):
    if variant=='PC2':files=s.archive_files()
    elif variant in ('B2B','B2R'):
        files=r21.source_files('B2B')
        if variant=='B2R':files['starnet_model.py']+=b'\n\n'+(ROOT/'workspace/round25_policy.py').read_bytes()
    else:raise ValueError(variant)
    compile(files['starnet_model.py'],'starnet_model.py','exec');return files
def fingerprint(extra=()):
    return dict(archive=s.ARCHIVE_SHA,dependencies={p:s.sha(ROOT/p) for p in sorted(set(DEPS+tuple(extra)))},
        packages={v:{n:hashlib.sha256(b).hexdigest() for n,b in source_files(v).items()} for v in VARIANTS})
def load(variant,namespace='default'):
    name='r25_'+namespace+'_'+variant
    if name not in sys.modules:
        folder=s.OUT/'packages'/variant/'team_submission'
        for n,b in source_files(variant).items():
            p=folder/n;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():
                if p.read_bytes()!=b:raise RuntimeError('Staged package drift '+str(p))
            else:p.write_bytes(b)
        spec=importlib.util.spec_from_file_location(name,folder/'starnet_model.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return sys.modules[name]
def clear(m):
    for n in ('coefficients','pc2_coefficients','r19_structure','r19_rows'):
        if hasattr(m,n):
            try:getattr(m,n).cache_clear()
            except AttributeError:pass
    if hasattr(m,'r21_clear_private'):m.r21_clear_private()
def run_game(spec,variant,mode='fast'):
    m=load(variant);clear(m);raw=FiniteEnv(c.generate(spec));logged=LoggedEnv(raw);chains=[];start=time.monotonic()
    with c.offline():
        model=r18.initialize(m,logged,n=spec['n'],mode=mode)
        for role in ('scout','commander'):
            agent=getattr(model,role);orig=agent._run_chain
            def trace(name,data,orig=orig,role=role):
                before=copy.deepcopy(data);ans=orig(name,data);chains.append(dict(role=role,chain=name,operation=model.operation_count,input=before,output=copy.deepcopy(ans)));return ans
            agent._run_chain=trace
        result=c.finish(raw,logged,model,n=spec['n'])
    result.update(status='PASS',spec=spec,variant=variant,mode=mode,chains=chains,
        candidate_records=copy.deepcopy(getattr(model,'pc2_records',[])),extra_records=copy.deepcopy(getattr(model,'r21_records',[])),
        scan_count=len(raw._scanned),elapsed_seconds=time.monotonic()-start)
    expected=44 if spec['n']==50 else spec['n']
    if result['scan_count']!=expected:raise RuntimeError('Scan count mismatch')
    c.audit_game(result);clear(m);return result
def checked(path,identity,fn):
    path=Path(path)
    if path.exists():
        r=s.read(path)
        if r.get('status')!='PASS' or r.get('identity')!=identity:raise RuntimeError('Existing incomplete/different '+str(path))
        return r
    rec=dict(status='STARTED',identity=identity,pid=__import__('os').getpid(),started_at=s.now());s.write(path,rec,True)
    try:
        got=fn();rec.update(got,status='PASS',identity=identity,completed_at=s.now());s.write(path,rec);return rec
    except BaseException as exc:
        rec.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),failed_at=s.now());s.write(path,rec);raise
def scan_prefix(r):return [a['args'][0] for a in r['actions'] if a['action']=='scan_node']
def bag(r):return c.bag(r['actions'])
def vetoes(r):return [x for x in r.get('extra_records',[]) if x.get('kind')=='bundle' and x.get('reason')=='risk_veto']
def selections(r):return [x for x in r.get('extra_records',[]) if x.get('kind')=='bundle' and x.get('reason')=='selected']
