"""Prospective paired experiments from the exact S48 ZIP, with real GLM option."""
import argparse
import concurrent.futures
import contextlib
import copy
import io
import json
import os
import socket
import time
from pathlib import Path

from round8_common import ROOT, OUT, ARCHIVE_SHA, prepare_baseline, load_s48, people, file_hash, digest, save
from round4_seeds import generate, FAMILIES
from experiment_round3 import fast_model, TopChoiceLLM
from experiment_round5 import FiniteEnv, PublicEnv
from benchmark import LoggedEnv

COUNTS = {'dev': 48, 'confirm': 384, 'stress': 96, 'glm-base': 6, 'glm-check': 12}

def specs(pool):
    result = []
    for i in range(COUNTS[pool]):
        f, rep = i % 6, i // 6
        result.append(dict(case=f'r8-{pool}-{i:03}', pool=f'r8-{pool}', family=FAMILIES[f],
            scale=(.55, 1., 1.5)[(rep+f)%3], persona_mix=(rep//3+f)%3, rep=rep,
            mode=('degree_w','degree_r','community','w_r')[rep%4] if pool=='stress' else 'independent',
            n=50, ood=pool=='stress', generator_version=1))
    return result

def prepare():
    prepare_baseline()
    protocol = dict(baseline='S48', archive_sha256=ARCHIVE_SHA,
        initial_candidates=['Q48','S46','S44'],
        development='Q48 vs S48 on 48 dev; S46/S44 start with first 24, expand only on positive signal',
        confirm='Freeze ONE candidate before 384 new paired cases; no retuning on confirmation',
        confirm_gate='mean > 0, stratified bootstrap95 lower > 0, mean without best two > 0, resource valid',
        stress_gate='mean nonnegative, no implementation/resource failures; report tails without arbitrary veto',
        glm='6 baseline seeds x 3 repeats; candidate check 12 seeds x 3 repeats x 2 variants',
        sealed_old_pools='Do not read round4 final108 or round5 final36',
        generator_sha256=file_hash(ROOT/'workspace/round4_seeds.py'),
        seeds={pool:[dict(spec=s, seed_sha256=digest(generate(s))) for s in specs(pool)] for pool in COUNTS})
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != protocol:
        raise RuntimeError('Protocol drift')
    save(path, protocol)
    return protocol

def classes(variant):
    from round8_scan import CLASSES
    return CLASSES[variant]

def fingerprint(variant):
    names=['experiment_round8.py','round8_common.py','experiment_round3.py','experiment_round5.py',
           'mechanism_round5.py','round4_seeds.py','benchmark.py','zhipu.py','team_submission/starnet_model.py']
    if variant != 'S48': names.append('round8_scan.py')
    return dict(archive=ARCHIVE_SHA, files={name:file_hash(ROOT/'workspace'/name) for name in names})

def initialize(model):
    model.search_records=[]; model.scan_diagnostic_records=[]; model.scan_stop_reached=False
    if model.node_count != 50: model.scan_target=model.node_count
    return model

class StopService(SystemExit):
    pass

def real_llm(logpath):
    from zhipu import ZhipuLLM, ensure_credit_available
    import requests
    key=os.environ.get('ZHIPUAI_API_KEY','').strip()
    if not key: raise RuntimeError('Missing ZHIPUAI_API_KEY')
    class JournalLLM(ZhipuLLM):
        def _post_once(self,url,data):
            # No credentials are persisted. Uncertain paid requests stop this batch.
            def append(value):
                with logpath.open('a',encoding='utf-8') as f:
                    f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            self.http_request_count+=1
            append(dict(event='request',number=self.http_request_count,data=data))
            try:
                response=requests.post(url,headers=self.headers,json=data,timeout=(10,45))
            except requests.RequestException as e:
                append(dict(event='uncertain',number=self.http_request_count,error=type(e).__name__))
                raise StopService('Uncertain GLM request; no automatic retry')
            append(dict(event='response',number=self.http_request_count,status=response.status_code,body=response.text))
            ensure_credit_available(response)
            if response.status_code != 200: raise StopService('GLM service status '+str(response.status_code))
            return response
    return JournalLLM(key,5)

def run_one(spec, variant, mode='fast', repeat=0):
    identity=dict(spec=spec,seed_sha256=digest(generate(spec)),variant=variant,
        mode=mode,repeat=repeat,code=fingerprint(variant))
    folder=OUT/(spec['pool']+'-'+mode)
    path=folder/(spec['case']+'-'+variant+f'-r{repeat}.json')
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old['identity']!=identity: raise RuntimeError('Result identity drift: '+str(path))
        if old['status']!='PASS': raise RuntimeError('Prior incomplete run requires explicit recovery: '+str(path))
        return {k:old[k] for k in ('case','variant','mode','repeat','final_score','elapsed_seconds')}
    attempts=[]
    if mode!='glm':
        def deny(*a,**k):
            attempts.append('blocked');raise RuntimeError('Network disabled in local experiment')
        socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny
    seed=generate(spec);raw=FiniteEnv(seed);logged=LoggedEnv(raw);cls=classes(variant)
    result=dict(identity=identity,case=spec['case'],spec=spec,variant=variant,mode=mode,repeat=repeat,status='STARTED')
    save(path,result);started=time.monotonic();states=[];chains=[];model=None;llm=None
    try:
        llm=real_llm(path.with_suffix('.http.jsonl')) if mode=='glm' else TopChoiceLLM()
        with contextlib.redirect_stdout(io.StringIO()):
            model=initialize(fast_model(PublicEnv(logged),copy.deepcopy(people()),cls) if mode=='fast'
                             else cls(PublicEnv(logged),copy.deepcopy(people()),llm))
            if mode!='fast':
                for role in ('scout','commander'):
                    agent=getattr(model,role); original=agent._run_chain
                    def observed(name,data,original=original,role=role):
                        answer=original(name,data)
                        chains.append(dict(role=role,chain=name,operation=model.operation_count,input=data,output=answer))
                        return answer
                    agent._run_chain=observed
            for step in range(120):
                budget=raw.get_remaining_budget()
                if len(logged.actions)==model.scan_target or (budget<=20 and not any(s['budget']<=20 for s in states)):
                    states.append(dict(nodes=copy.deepcopy(model.local_nodes),edges=sorted(model.local_edges),
                        effects=dict(model.comm_effects),observations=copy.deepcopy(model.persona_observations),
                        budget=budget,operation=model.operation_count))
                if model.step():break
            else: raise RuntimeError('Resource fuse reached')
        costs={'scan_node':.5,'communicate':2.,'cut_link':3.,'shield_node':5.}
        used=sum(costs[a['action']] for a in logged.actions)
        if abs(100-used-raw.get_remaining_budget())>1e-8: raise RuntimeError('Budget mismatch')
        if model.operation_count>=120 or model.llm_counter.call_count>=120 or attempts:
            raise RuntimeError('Resource/network guard failed')
        if len(raw._scanned)!=model.scan_target: raise RuntimeError('Unexpected scan coverage')
        result.update(status='PASS',final_score=raw.evaluate(),remaining_budget=raw.get_remaining_budget(),
            operation_count=model.operation_count,logical_calls=model.llm_counter.call_count,
            http_calls=llm.http_request_count,metadata=getattr(llm,'response_metadata',[]),
            stop_reason=model.stop_reason,actions=logged.actions,states=states,chains=chains,
            search_records=model.search_records,scan_diagnostic_records=model.scan_diagnostic_records,
            omitted=sorted(set(range(1,51))-raw._scanned),network_attempts=len(attempts),elapsed_seconds=time.monotonic()-started)
        save(path,result)
    except BaseException as exc:
        result.update(status='FAILED',error_type=type(exc).__name__,elapsed_seconds=time.monotonic()-started,
            actions=logged.actions,chains=chains,http_calls=getattr(llm,'http_request_count',0))
        save(path,result);raise
    return {k:result[k] for k in ('case','variant','mode','repeat','final_score','elapsed_seconds')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--pool',choices=list(COUNTS),default='dev');ap.add_argument('--variants',nargs='+',default=['S48','Q48'])
    ap.add_argument('--mode',choices=['fast','integration','glm'],default='fast')
    ap.add_argument('--limit',type=int);ap.add_argument('--repeats',type=int,default=1);ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args();prepare()
    if args.prepare: print('Frozen S48 and prospective manifest prepared');return
    if args.pool in ('confirm','stress'):
        selection=json.loads((OUT/'candidate-freeze.json').read_text(encoding='utf-8'))
        for v in args.variants:
            if v!='S48' and (v!=selection['variant'] or fingerprint(v)!=selection['code']):
                raise RuntimeError('Only frozen candidate may access confirmation/stress')
    jobs=[(s,v,args.mode,r) for s in specs(args.pool)[:args.limit] for r in range(args.repeats) for v in args.variants]
    if args.mode=='glm':
        # Serial paid calls; any credit or service error immediately stops dispatch.
        for job in jobs: print(json.dumps(run_one(*job),ensure_ascii=True),flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures=[ex.submit(run_one,*job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):print(json.dumps(future.result(),ensure_ascii=True),flush=True)

if __name__=='__main__': main()
