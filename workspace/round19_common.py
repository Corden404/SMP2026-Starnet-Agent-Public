"""Round19 isolated package loading and durable records; never edits old studies."""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import time
import round19_state as s
import round18_common as old
from experiment_round5 import FiniteEnv,PublicEnv
from benchmark import LoggedEnv

ROOT=s.ROOT
CORE=tuple(old.CORE)+('workspace/round19_state.py','workspace/round19_common.py','workspace/round19_numeric.py',
                      'workspace/round18_cut1.py','workspace/round16_policy.py')


def source_files(variant='PC2'):
    files=s.archive_files()
    additions=[]
    if variant in ('Cut1','Numeric','CT1','CT1R'):additions.append('workspace/round18_cut1.py')
    if variant in ('Numeric','CT1','CT1R'):additions.append('workspace/round19_numeric.py')
    if variant in ('CT1','CT1R'):additions.append('workspace/round19_policy.py')
    if variant not in ('PC2','Cut1','Numeric','CT1','CT1R'):raise ValueError(variant)
    for path in additions:files['starnet_model.py']+=b'\n\n'+(ROOT/path).read_bytes()
    if variant=='CT1R':files['starnet_model.py']+=b'\n\nclass ParticipantSquadModel(ParticipantSquadModel):\n    r19_risk = True\n'
    compile(files['starnet_model.py'],'starnet_model.py','exec')
    return files


def load(variant='PC2',namespace='default'):
    name='round19_'+namespace+'_'+variant
    if name not in sys.modules:
        folder=s.OUT/'packages'/variant/'team_submission'
        for key,data in source_files(variant).items():
            p=folder/key
            if p.exists():
                if p.read_bytes()!=data:raise RuntimeError('Staged source drift: '+str(p))
            else:
                p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:f.write(data)
        spec=importlib.util.spec_from_file_location(name,folder/'starnet_model.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


def clear(module):
    for name in ('coefficients','pc2_coefficients'):
        if hasattr(module,name):getattr(module,name).cache_clear()
    if hasattr(module,'r19_clear_private'):module.r19_clear_private()
    elif hasattr(module,'r18_clear_private'):module.r18_clear_private()


def fingerprint(extra=()):
    return dict(archive=s.ARCHIVE_SHA,dependencies={p:s.sha(ROOT/p) for p in sorted(set(CORE+tuple(extra)))})


def snapshot(row,event):
    op=event['operation']
    return dict(operation=op,calls=sum(c['operation']<op for c in row['chains']),budget=event['budget'],
                pc2_seen=sorted({t for e in row['candidate_records'] if e['operation']<=op for t in e['tags']}))


def context(model):
    return dict(operation=model.operation_count,calls=model.llm_counter.call_count,operation_limit=model.operation_limit,
                safe_limit=model.llm_counter.safe_limit,seen=set(model._pc2_seen))


def public_at(row,snap):
    module=load('PC2','restore');clear(module)
    with old.c.offline():
        _,raw,logged,model=old.c.restore(row,snap['operation'],module)
        model._pc2_init();model._pc2_seen=set(snap['pc2_seen'])
        if raw.get_remaining_budget()!=snap['budget'] or model.llm_counter.call_count!=snap['calls']:
            raise RuntimeError('Public snapshot resource mismatch')
        state=model.completed_state();eligible=set(model.local_nodes);ctx=context(model)
    clear(module)
    return state,eligible,ctx


def checked(path,identity,operation):
    path=Path(path)
    if path.exists():
        row=s.read(path)
        if row.get('status')!='PASS' or row.get('identity')!=identity:
            raise RuntimeError('Existing incomplete/different result requires inspection: '+str(path))
        return row
    record=dict(status='STARTED',identity=identity,pid=os.getpid(),started_at=s.now())
    s.write(path,record,True)
    try:
        result=operation();record.update(result,status='PASS',identity=identity,completed_at=s.now());s.write(path,record)
        return record
    except BaseException as exc:
        record.update(status='FAILED',error_type=type(exc).__name__,error=str(exc),failed_at=s.now());s.write(path,record)
        raise


def branch(row,snap,action):
    module=load('PC2','branch');clear(module)
    with old.c.offline():
        _,raw,logged,model=old.c.restore(row,snap['operation'],module)
        model._pc2_init();model._pc2_seen=set(snap['pc2_seen'])
        if raw.get_remaining_budget()!=snap['budget'] or model.llm_counter.call_count!=snap['calls']:
            raise RuntimeError('Branch resource mismatch')
        old.c.force(model,raw,action);result=old.c.finish(raw,logged,model)
    result.update(spec=row['spec'],variant='PC2-forced-branch',mode='diagnostic',start_operation=snap['operation'],
                  forced_action=action,prefix_sha256=s.digest(row['actions'][:snap['operation']]))
    old.c.audit_game(result);clear(module);return result


def action(key):
    kind,u,v=key
    return dict(action=kind,target_node_1=u,target_node_2=v,prompt_id=1 if kind=='comm' else None,
                cost={'comm':2.,'cut':3.,'shield':5.}[kind],gain=0.,roi=0.)


def specs(pool):
    rows=copy.deepcopy(old.specs(pool))
    for i,row in enumerate(rows):row.update(case='r19-'+pool+'-%03d'%i,pool='r19-'+pool)
    return rows


def entries(pool):
    return [dict(spec=spec,sha256=s.digest(old.c.generate(spec))) for spec in specs(pool)]


def run_game(spec,variant,mode='fast'):
    module=load(variant);clear(module)
    raw=FiniteEnv(old.c.generate(spec));logged=LoggedEnv(raw);chains=[];start=time.monotonic()
    with old.c.offline():
        model=old.initialize(module,logged,n=spec['n'],mode=mode)
        for role in ('scout','commander'):
            agent=getattr(model,role);original=agent._run_chain
            def trace(name,data,original=original,role=role):
                before=copy.deepcopy(data);answer=original(name,data)
                chains.append(dict(role=role,chain=name,operation=model.operation_count,input=before,output=copy.deepcopy(answer)))
                return answer
            agent._run_chain=trace
        result=old.c.finish(raw,logged,model,n=spec['n'])
        result['extra_records']=copy.deepcopy(getattr(model,'r19_records',getattr(model,'r18_records',[])))
    result.update(status='PASS',spec=spec,variant=variant,mode=mode,chains=chains,scan_count=len(raw._scanned),
                  elapsed_seconds=time.monotonic()-start)
    if result['scan_count']!=(44 if spec['n']==50 else spec['n']):raise RuntimeError('Scan prefix size')
    old.c.audit_game(result);clear(module);return result
