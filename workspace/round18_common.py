"""PC2-derived, public-state-only evaluation; all outputs are additive round18 files."""
from __future__ import annotations
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
import round18_state as s
import round16_common as c
from experiment_round5 import FiniteEnv, PublicEnv
from benchmark import LoggedEnv

FAMILIES = c.FAMILIES
SCALES = c.SCALES
EXTENSIONS = {'Cut1':'workspace/round18_cut1.py', 'Fast':'workspace/round18_fast.py'}
CORE = tuple(c.CORE_DEPENDENCIES) + ('workspace/round18_state.py','workspace/round18_common.py')


def fingerprint(extra=()):
    return dict(archive=s.ARCHIVE_SHA, dependencies={p:s.sha(s.ROOT/p) for p in sorted(set(CORE+tuple(extra)))})


def source_files(variant='PC2'):
    files=s.archive_files()
    if variant!='PC2':
        if variant not in EXTENSIONS: raise ValueError(variant)
        files['starnet_model.py'] += b'\n\n' + (s.ROOT/EXTENSIONS[variant]).read_bytes()
    compile(files['starnet_model.py'],'starnet_model.py','exec')
    return files


def stage(variant):
    folder=s.OUT/'packages'/variant/'team_submission'
    for name,content in source_files(variant).items():
        path=folder/name
        if path.exists():
            if path.read_bytes()!=content: raise RuntimeError('Staged source drift: '+str(path))
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream: stream.write(content)
    return folder


def load(variant='PC2', namespace='default'):
    name='r18_'+namespace+'_'+variant
    if name not in sys.modules:
        folder=stage(variant)
        spec=importlib.util.spec_from_file_location(name,folder/'starnet_model.py')
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name]


def clear(module):
    for name in ('coefficients','pc2_coefficients'):
        getattr(module,name).cache_clear()
    if hasattr(module,'r18_clear_private'): module.r18_clear_private()


def specs(pool):
    if pool in ('diagnostic','integration','real'):
        count={'diagnostic':12,'integration':6,'real':24}[pool]
        rows=[dict(family=FAMILIES[i%6],scale=SCALES[(i%6+i//6)%3],
            persona_mix=(2*(i%6)+i//6)%3,rep=i//6,mode='independent',n=50,ood=False,generator_version=1)
            for i in range(count)]
    elif pool in ('dev','confirm'):
        rows=[dict(family=f,scale=sc,persona_mix=p,rep=rep,mode='independent',n=50,ood=False,generator_version=1)
            for rep in range(1 if pool=='dev' else 4) for f in FAMILIES for sc in SCALES for p in range(3)]
    elif pool=='stress':
        rows=copy.deepcopy(c.stress_design.specs())
    else: raise ValueError(pool)
    for i,row in enumerate(rows):row.update(case='r18-'+pool+'-%03d'%i,pool='r18-'+pool)
    return rows


def entries(pool):
    return [dict(spec=spec,sha256=s.digest(c.generate(spec))) for spec in specs(pool)]


def initialize(module, env, n=50, mode='fast', llm=None):
    # The archived PC2 and previous S44 person configurations are byte-identical.
    if json.loads(source_files('PC2')['config.json'])['person'] != c.old.people():
        raise RuntimeError('Inherited initialization config mismatch')
    model=c.old.initialize(module,env,n=n,mode=mode,llm=llm)
    if not hasattr(model,'pc2_records'): model._pc2_init()
    return model


def run_game(spec,variant='PC2',mode='fast',capture=False):
    module=load(variant);clear(module)
    raw=FiniteEnv(c.generate(spec));logged=LoggedEnv(raw)
    states=[];chains=[];started=time.monotonic()
    with c.offline():
        model=initialize(module,logged,n=spec['n'],mode=mode)
        for role in ('scout','commander'):
            agent=getattr(model,role);original=agent._run_chain
            def trace(name,data,original=original,role=role):
                before=copy.deepcopy(data);answer=original(name,data)
                chains.append(dict(role=role,chain=name,operation=model.operation_count,input=before,output=copy.deepcopy(answer)))
                return answer
            agent._run_chain=trace
        if capture:
            original=model._learned_candidates;seen=set()
            def select(budget):
                options=original(budget)
                tags=(['first'] if model.operation_count==44 else [])+['budget'+str(b) for b in (40,20,10) if budget<=b]
                new=[tag for tag in tags if tag not in seen]
                if new and options:
                    seen.update(new)
                    states.append(dict(operation=model.operation_count,calls=model.llm_counter.call_count,budget=budget,
                        tags=new,chosen=copy.deepcopy(options[0]),pc2_seen=sorted(model._pc2_seen)))
                return options
            model._learned_candidates=select
        result=c.finish(raw,logged,model,n=spec['n'])
        result['extra_records']=copy.deepcopy(getattr(model,'r18_records',[]))
    expected=44 if spec['n']==50 else spec['n']
    if len(raw._scanned)!=expected: raise RuntimeError('Unexpected scan count')
    result.update(status='PASS',spec=spec,variant=variant,mode=mode,chains=chains,states=states,
        scan_count=len(raw._scanned),elapsed_seconds=time.monotonic()-started)
    c.audit_game(result);clear(module)
    return result


def restore(row,snapshot,namespace='branch'):
    module=load('PC2',namespace);clear(module)
    _,raw,logged,model=c.restore(row,snapshot['operation'],module)
    model.pc2_records=copy.deepcopy([r for r in row['candidate_records'] if r['operation']<=snapshot['operation']])
    model._pc2_seen=set(snapshot['pc2_seen'])
    if raw.get_remaining_budget()!=snapshot['budget'] or model.llm_counter.call_count!=snapshot['calls']:
        raise RuntimeError('Restored resources differ from public snapshot')
    return module,raw,logged,model


def checked_record(path,identity,operation):
    """Reuse only immutable successful identities. Never retry incomplete work implicitly."""
    path=Path(path)
    if path.exists():
        old=s.read(path)
        if old.get('identity')!=identity or old.get('status')!='PASS':
            raise RuntimeError('Existing incomplete/different record requires inspection: '+str(path))
        return old
    row=dict(status='STARTED',identity=identity,pid=os.getpid(),started_at=s.now())
    s.write(path,row,True)
    try:
        result=operation()
        row.update(result,status='PASS',identity=identity,completed_at=s.now())
        s.write(path,row)
        return row
    except BaseException as exc:
        row.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),failed_at=s.now())
        s.write(path,row)
        raise


def branch(row,snapshot,action):
    with c.offline():
        module,raw,logged,model=restore(row,snapshot)
        c.force(model,raw,action)
        result=c.finish(raw,logged,model)
    result.update(spec=row['spec'],variant='PC2-forced-branch',mode='diagnostic',
        start_operation=snapshot['operation'],forced_action=action,
        prefix_sha256=s.digest(row['actions'][:snapshot['operation']]))
    c.audit_game(result);clear(module)
    return result


def scan_prefix(row):
    return [a['args'][0] for a in row['actions'] if a['action']=='scan_node']


def bag(row):
    return c.bag(row['actions'])
