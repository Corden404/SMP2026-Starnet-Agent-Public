"""Verify the actual ZIP through CaseVO; mirror public sandbox actions independently."""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
import sys
import socket
import time
import zipfile
from pathlib import Path
from round8_common import ROOT,OUT,load_s48,people,file_hash,save
from experiment_round8 import specs,real_llm,classes,initialize
from experiment_round3 import TopChoiceLLM,SERVER,fast_model
from experiment_round5 import FiniteEnv,PublicEnv
from round4_seeds import generate
from benchmark import LoggedEnv

class ServiceAbort(SystemExit):pass

class MirroredEnv:
    """Only the public wrapper is passed to the participant, never the mirror seed."""
    def __init__(self,remote,mirror):self.remote=remote;self.mirror=mirror
    def get_remaining_budget(self):
        try:r=self.remote.get_remaining_budget()
        except Exception as e:raise ServiceAbort(type(e).__name__)
        if abs(r-self.mirror.get_remaining_budget())>1e-8:raise ServiceAbort('Budget mismatch')
        return r
    def _call(self,name,*args):
        try:r=getattr(self.remote,name)(*args)
        except Exception as e:raise ServiceAbort(type(e).__name__)
        expected=getattr(self.mirror,name)(*args)
        if name=='scan_node':
            if r is None or any(r[k]!=expected[k] for k in ('persona','comm_left')) or abs(r['w']-expected['w'])>1e-8 or set(r['neighbors'])!=set(expected['neighbors']):
                raise ServiceAbort('Scan mismatch')
        elif name=='communicate':
            if r.get('status')!='success' or abs(r['new_w']-expected['new_w'])>1e-8:raise ServiceAbort('Response mismatch')
        elif r!=expected:raise ServiceAbort('Action mismatch')
        return r
    def scan_node(self,i):return self._call('scan_node',i)
    def communicate(self,i,p):return self._call('communicate',i,p)
    def cut_link(self,u,v):return self._call('cut_link',u,v)
    def shield_node(self,i):return self._call('shield_node',i)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',default='Q48');ap.add_argument('--study',default='')
    ap.add_argument('--remote',action='store_true');ap.add_argument('--glm',action='store_true');ap.add_argument('--sizes',action='store_true')
    ap.add_argument('--glm-model',choices=['glm-4-plus','glm-4.5-air','glm-4.6v'],default='glm-4-plus')
    args=ap.parse_args();out=OUT/args.study if args.study else OUT
    if not args.remote and not args.glm:
        def deny(*a,**k):raise ServiceAbort('Network disabled during offline ZIP acceptance')
        socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    stage=out/('submission-'+args.variant)
    manifest=json.loads((stage/'manifest.json').read_text(encoding='utf-8'))
    archive=Path(manifest['archive']);assert file_hash(archive)==manifest['sha256']
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(manifest['submission_files'])
        for name,h in manifest['submission_files'].items():
            assert file_hash(stage/'team_submission'/name)==h
            assert z.read(name)==(stage/'team_submission'/name).read_bytes()
    name='round8_actual_package_'+args.variant
    spec=importlib.util.spec_from_file_location(name,stage/'team_submission/starnet_model.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    variants={'S48':load_s48().ParticipantSquadModel,args.variant:module.ParticipantSquadModel}
    cases=specs('glm-check')[:6]
    if args.sizes:
        cases=[]
        for n in (4,100):
            s=dict(specs('glm-check')[5]);s.update(case=f'r8-size-{n}',n=n);cases.append(s)
    folder=out/('size-acceptance' if args.sizes else ('remote-' if args.remote else 'package-')+('glm' if args.glm else 'fake'))
    if args.glm and args.glm_model!='glm-4-plus':folder=folder.parent/(folder.name+'-'+args.glm_model)
    save(folder/'protocol.json',dict(cases=[s['case'] for s in cases],variants=list(variants),archive_sha256=manifest['sha256'],
        purpose='Actual ZIP and public API consistency, not candidate selection',remote=args.remote,glm=args.glm,glm_model=args.glm_model if args.glm else None))
    completed=[]
    for s in cases:
        seed=generate(s);limit=seed['global_setting']['max_api_calls']
        for variant,cls in variants.items():
            path=folder/(s['case']+'-'+variant+'.json')
            if path.exists():
                old=json.loads(path.read_text(encoding='utf-8'))
                if old['status']!='PASS' or old['archive_sha256']!=manifest['sha256']:raise RuntimeError('Prior incomplete or mismatched acceptance')
                completed.append(old);continue
            result=dict(case=s['case'],variant=variant,status='STARTED',archive_sha256=manifest['sha256'])
            save(path,result);start=time.monotonic();mirror=FiniteEnv(seed);llm=None;logged=None
            try:
                if args.remote:
                    from api_client import RemoteStarNetEnv
                    remote=RemoteStarNetEnv(SERVER,seed,timeout=20.)
                    result['session_id']=remote.session_id;save(path,result)
                    env=MirroredEnv(remote,mirror)
                else:env=mirror
                logged=LoggedEnv(env);llm=real_llm(path.with_suffix('.http.jsonl')) if args.glm else TopChoiceLLM()
                if args.glm and args.glm_model!='glm-4-plus':
                    llm.model_config.update(model=args.glm_model,thinking={'type':'disabled'})
                ps=copy.deepcopy(people())
                with contextlib.redirect_stdout(io.StringIO()):
                    model=cls(PublicEnv(logged),ps,llm)
                    for _ in range(limit):
                        if model.step():break
                    else:raise RuntimeError('Fuse reached')
                prediction=mirror.evaluate();actual=remote.trigger_eval() if args.remote else prediction
                if abs(actual-prediction)>(.0051 if args.remote else 1e-8):raise RuntimeError('Final score mismatch')
                assert model.operation_count<limit and model.llm_counter.call_count<limit
                result.update(status='PASS',final_score=actual,expected_score=prediction,score_error=actual-prediction,
                    actions=logged.actions,operation_count=model.operation_count,logical_calls=model.llm_counter.call_count,
                    http_calls=llm.http_request_count,metadata=getattr(llm,'response_metadata',[]),
                    remaining_budget=mirror.get_remaining_budget(),scan_count=len(mirror._scanned),
                    stop_reason=model.stop_reason,elapsed_seconds=time.monotonic()-start)
                if not args.glm:
                    reference_env=FiniteEnv(seed);reference_log=LoggedEnv(reference_env)
                    reference=initialize(fast_model(PublicEnv(reference_log),copy.deepcopy(people()),classes(variant)))
                    with contextlib.redirect_stdout(io.StringIO()):
                        for _ in range(limit):
                            if reference.step():break
                        else:raise RuntimeError('Reference fuse reached')
                    key=lambda xs:[(a['action'],tuple(a['args'])) for a in xs]
                    result['same_actions_as_frozen_experiment']=key(logged.actions)==key(reference_log.actions)
                    result['reference_experiment_score']=reference_env.evaluate()
                    if not result['same_actions_as_frozen_experiment']:raise RuntimeError('Packaged entry trajectory mismatch')
                    if abs(prediction-reference_env.evaluate())>1e-8:raise RuntimeError('Packaged entry score mismatch')
                if args.sizes:assert len(mirror._scanned)==s['n']
                save(path,result);completed.append(result)
                print(json.dumps({k:result[k] for k in ('case','variant','status','final_score','score_error')},ensure_ascii=True),flush=True)
            except BaseException as e:
                result.update(status='FAILED',error_type=type(e).__name__,actions=logged.actions if logged else [],elapsed_seconds=time.monotonic()-start)
                save(path,result);raise
    if args.sizes:
        for s in cases:
            rows=[r for r in completed if r['case']==s['case']]
            assert abs(rows[0]['final_score']-rows[1]['final_score'])<1e-8
    save(folder/'summary.json',dict(status='PASS',games=len(completed),max_error=max(abs(r['score_error']) for r in completed),
        max_operations=max(r['operation_count'] for r in completed),max_calls=max(r['logical_calls'] for r in completed),
        glm_http_calls=sum(r['http_calls'] for r in completed)))

if __name__=='__main__':main()
