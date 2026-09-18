"""Persist the actual model HTTP request/response before CaseVO parses it."""
import json
import requests

from recovery_store import RecoveryStop,canonical,encode
from zhipu import ZhipuLLM,ensure_credit_available


FIELDS=('strategy','planner_options','persona_observations','scan_closed','decision_history',
        'scan_queue','comm_effects','action_history','node_count','operation_limit','llm_limit',
        'prompt_ids','operation_count','no_progress_count','stopped','stop_reason',
        'local_nodes','local_edges','dead_nodes')


def model_state(model,env):
    if model is None:return {'initializing':True,'budget':env.last_budget if env else None}
    value={k:getattr(model,k,None) for k in FIELDS}
    value.update(logical_calls=model.llm_counter.call_count,budget=env.last_budget,
                 schedule={'steps':model.schedule.steps,'time':model.schedule.time})
    # Current submission random sampling uses an explicit seed derived from operation_count.
    # Framework RNG is not read by these planner/ROI paths; no fabricated RNG equivalence.
    return canonical(value)


class DurableLLM(ZhipuLLM):
    def __init__(self,key,journal,gate,model_config=None,post=None):
        super().__init__(key,5,model_config)
        self.journal=journal;self.gate=gate;self.post=post or requests.post
        self.state=lambda:None;self.role_context=None

    def _post_once(self,url,data):
        request={'url':url,'data':data,'role_context':self.role_context}
        old=self.journal.enter('llm',request,self.state())
        if old and old['status']=='RESPONSE_DURABLE':
            packet=json.loads(old['response']);self.journal.cache_hits+=1
            self.journal.store.put('llm_cache_hits',self.journal.store.get('llm_cache_hits',0)+1)
        else:
            if old and old['status']=='IN_FLIGHT':raise RecoveryStop('LLM_RESULT_UNKNOWN','Unconfirmed prior request at event '+str(self.journal.seq))
            self.gate.check(paid=True)
            self.journal.inflight()
            wid=self.journal.store.begin_wire(self.journal.aid,'llm',{'url':url,'data':data})
            self.http_request_count+=1
            try:
                response=self.post(url,headers=self.headers,json=data,timeout=(10,30))
                packet={'status':response.status_code,'text':response.text,
                        'headers':{k:v for k,v in response.headers.items() if k.lower() in ('x-request-id','content-type')}}
                self.journal.store.durable_model_response(wid,self.journal.tid,self.journal.seq,packet)
            except requests.RequestException as e:
                self.journal.store.end_wire(wid,'LLM_RESULT_UNKNOWN',{'error_type':type(e).__name__})
                raise RecoveryStop('LLM_RESULT_UNKNOWN',type(e).__name__)
        result=requests.Response();result.status_code=packet['status'];result._content=packet['text'].encode();result.encoding='utf-8';result.headers.update(packet['headers'])
        ensure_credit_available(result)
        if result.status_code!=200:raise RecoveryStop('LLM_SERVICE_FAILURE','HTTP '+str(result.status_code))
        return result


def attach_observers(model,llm,journal,env):
    """Local-only observers: preserve original prompts, chain and action semantics."""
    llm.state=lambda:model_state(model,env)
    for role in ('commander','scout','executor'):
        agent=getattr(model,role);original=agent._run_chain
        def run(name,data,original=original,role=role):
            journal.observe('chain_input',{'role':role,'chain':name,'input':data,'state':model_state(model,env)})
            llm.role_context={'role':role,'chain':name,'operation':model.operation_count}
            output=original(name,data)
            journal.observe('chain_output',{'role':role,'chain':name,'output':output})
            llm.role_context=None
            return output
        agent._run_chain=run
    original_plan=model._rollout_plans
    def plans(budget):
        result=original_plan(budget)
        # The old prompt includes local search elapsed time. Verify every semantic
        # field exactly, then replay ONLY the recorded diagnostic seconds field.
        comparable=[{k:v for k,v in p.items() if k!='seconds'} for p in result]
        old=journal.enter('planning_result',{'plans':comparable},model_state(model,env))
        if old and old['response'] is not None:
            previous=json.loads(old['response'])
            for new,prior in zip(result,previous):
                if 'seconds' in prior:new['seconds']=prior['seconds']
        else:journal.response(result)
        return result
    model._rollout_plans=plans
    original_action=model._execute_planned_action
    def action(chosen,budget,before):
        journal.observe('resolved_action',{'action':chosen,'budget':budget,'state':model_state(model,env)})
        status=original_action(chosen,budget,before)
        journal.observe('applied_locally',{'status':status,'state':model_state(model,env)})
        return status
    model._execute_planned_action=action
