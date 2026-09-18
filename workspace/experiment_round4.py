"""Round 4 runs. Remote execution requires --remote; final seeds require a freeze."""
import argparse
import contextlib
import copy
import io
import json
import os
import time

from api_client import RemoteStarNetEnv
from benchmark import LoggedEnv
from experiment_round3 import ProxyEnv, TopChoiceLLM, fast_model
from round4_common import OUT, ROOT, SOURCE, TraceLLM, attach_traces, cached, digest, identity, save
from round4_seeds import load_case
from team_submission.starnet_model import ParticipantSquadModel
from zhipu import ZhipuLLM, APICreditExhausted

class AuditedEnv(LoggedEnv):
    def _action(self,name,*args):
        event={'action':name,'args':args,'started':time.time()}
        self.actions.append(event)
        try:
            result=getattr(self.delegate,name)(*args)
            event['result']=result
            event['status']='returned'
            return result
        except BaseException as exc:
            event.update(status='exception',error_type=type(exc).__name__,http_status=getattr(getattr(exc,'response',None),'status_code',None))
            raise
        finally:event['elapsed_seconds']=time.time()-event['started']

SERVER='http://8.222.218.162:5000'
VARIANTS={
    'B0':{'strategy':'planner','planner':{'decision_mode':'roi','roi_floor':1.0}},
    'R1':{'strategy':'planner','planner':{'decision_mode':'rollout','rollout_combinations':False,'rollout_learning':False}},
    'R2':{'strategy':'planner','planner':{'decision_mode':'rollout','rollout_combinations':True,'rollout_learning':False}},
    'I1':{'strategy':'planner','planner':{'decision_mode':'information','rollout_combinations':False,'rollout_learning':True}},
    'C1':{'strategy':'planner','planner':{'decision_mode':'rollout','rollout_combinations':True,'rollout_learning':True}},
}


def run(case,variant,batch,remote=False,repeat=0,attempt=0,model_config=None):
    if case.startswith('final-'):
        freeze=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
        if variant not in ('B0',freeze['variant']):raise ValueError('Unselected final variant')
        from round4_common import source_files
        if freeze['submission_files'] != source_files():raise ValueError('Final source drift')
    spec,seed=load_case(case,allow_final=case.startswith('final-'))
    people=json.loads((SOURCE/'config.json').read_text(encoding='utf-8-sig'))['person']
    limit=seed['global_setting']['max_api_calls']
    runtime={'node_count':spec['n'],'operation_limit':limit,'llm_limit':limit,'prompt_ids':[1,2,3],**copy.deepcopy(VARIANTS[variant])}
    people[0]['runtime']=runtime
    llm=ZhipuLLM('placeholder',5,model_config) if remote else TopChoiceLLM()
    configuration=llm.model_config if remote else {'provider':'local','model':'deterministic-first-choice','endpoint':''}
    env_identity={'type':'official-sandbox' if remote else 'proxy','endpoint':SERVER if remote else '',
                  'version':'unknown','calibration_batch':batch}
    ident=identity(seed,ParticipantSquadModel._load_settings(people,2*spec['n']),configuration,batch,env_identity,repeat,attempt)
    if case.startswith('final-'):
        if case not in freeze['final_cases'] or repeat>=freeze['repeats']:
            raise ValueError('Final case/repeat outside frozen protocol')
        if ident['definition']['harness_files']!=freeze['harness_files'] or configuration!=freeze['model_config']:
            raise ValueError('Final harness/model drift')
        if variant!='B0' and VARIANTS[variant]!=freeze['runtime']:
            raise ValueError('Final runtime drift')
    path=OUT/batch/(case+'-'+variant+f'-r{repeat}-a{attempt}.json')
    old=cached(path,ident)
    if old:
        print(json.dumps({'cache_hit':str(path),'run_id':ident['run_id']}),flush=True)
        return old
    result={'identity':ident,'case':case,'spec':spec,'variant':variant,'repeat':repeat,'attempt':attempt,
            'status':'running','evaluation':env_identity,'model_config':configuration,'started':time.time()}
    save(path,result)
    trace=None;model=None;env=None
    try:
        if remote:
            key=os.environ.get('ZHIPUAI_API_KEY','').strip()
            if not key:raise ValueError('ZHIPUAI_API_KEY is required')
            llm=ZhipuLLM(key,5,model_config)
            natural_path=OUT/batch/'natural'/(case+'.json')
            natural_identity={'seed_sha256':digest(seed),'environment':env_identity}
            if natural_path.exists():
                natural_data=json.loads(natural_path.read_text(encoding='utf-8'))
                if natural_data['identity']!=natural_identity:raise ValueError('Natural baseline drift')
                natural=natural_data['score']
            else:
                natural=RemoteStarNetEnv(SERVER,seed,timeout=30).trigger_eval()
                save(natural_path,{'identity':natural_identity,'score':natural,'time':time.time()})
            env=AuditedEnv(RemoteStarNetEnv(SERVER,seed,timeout=30))
        else:
            raw=ProxyEnv(seed);natural=raw.evaluate();env=AuditedEnv(raw)
        trace=TraceLLM(llm)
        with contextlib.redirect_stdout(io.StringIO()):
            model=ParticipantSquadModel(env,people,trace) if remote else fast_model(env,people)
        events=attach_traces(model,trace) if remote else []
        states=[]
        for step in range(limit):
            before=len(env.actions)
            states.append({'step':step,'budget':env.get_remaining_budget(),
                           'nodes':copy.deepcopy(model.local_nodes),'edges':sorted(model.local_edges),
                           'comm_effects':dict(model.comm_effects),
                           'persona_observations':copy.deepcopy(model.persona_observations),
                           'first_action':before,'llm_calls':model.llm_counter.call_count})
            with contextlib.redirect_stdout(io.StringIO()):status=model.step()
            if remote and (step%5==4 or status):
                checkpoint={**result,'status':'checkpoint','checkpoint_step':step+1,
                            'checkpoint_time':time.time(),'counts_are_lower_bounds_if_interrupted':True,
                            'llm_calls':model.llm_counter.call_count,'http_llm_calls':llm.http_request_count,
                            'actions':env.actions,'requests':trace.requests,'chains':events,'states':states,
                            'decisions':model.decision_history,'stop_reason':model.stop_reason}
                save(path.parent/'checkpoints'/path.name,checkpoint)
            if status:break
        else:raise RuntimeError('Runner reached fuse')
        result.update(natural_score=natural,steps=step+1,llm_calls=model.llm_counter.call_count,
                      http_llm_calls=llm.http_request_count if remote else None,
                      remaining_budget=env.get_remaining_budget(),stop_reason=model.stop_reason,
                      states=states,actions=env.actions,chains=events,requests=trace.requests,
                      decisions=model.decision_history,response_metadata=getattr(llm,'response_metadata',[]))
        failure_words=('uncertain','不可用','不确定','无效','Three operations','连续三轮')
        if any(w.lower() in model.stop_reason.lower() for w in failure_words):
            raise RuntimeError('Uncertain execution; see stop_reason')
        result['status']='awaiting_evaluation';save(path,result)
        result['final_score']=env.delegate.trigger_eval() if remote else env.delegate.evaluate()
        result.update(status='completed',elapsed_seconds=time.time()-result['started'])
        save(path,result)
        print(json.dumps({k:result[k] for k in ('case','variant','status','final_score','elapsed_seconds')},ensure_ascii=False),flush=True)
        return result
    except BaseException as exc:
        result.update(status='failed',error_type=type(exc).__name__,error_http_status=getattr(getattr(exc,'response',None),'status_code',None),elapsed_seconds=time.time()-result['started'])
        if env is not None:result['actions']=env.actions
        if trace is not None:result['requests']=trace.requests
        if model is not None:result['stop_reason']=model.stop_reason
        save(path,result)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--case',required=True);p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--batch',required=True);p.add_argument('--remote',action='store_true')
    p.add_argument('--repeat',type=int,default=0);p.add_argument('--attempt',type=int,default=0)
    args=p.parse_args()
    try:run(args.case,args.variant,args.batch,args.remote,args.repeat,args.attempt)
    except APICreditExhausted:
        print('CREDIT_EXHAUSTED: batch must stop',flush=True)
        raise SystemExit(13)
