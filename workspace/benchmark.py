"""Paired official-sandbox experiments. Only synthetic/public seeds are sent."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import networkx as nx
from api_client import RemoteStarNetEnv
from team_submission.starnet_model import ParticipantSquadModel
from zhipu import APICreditExhausted, ZhipuLLM

ROOT=Path(__file__).resolve().parents[1]


def make_seed(case):
    if case=='official':
        return json.loads((ROOT/'workspace/custom_seeds/my_test_network.json').read_text(encoding='utf-8'))
    specs={'develop10a':(10,8123,'ba'),'develop10b':(10,8327,'ws'),
           'validate50a':(50,9301,'ba'),'validate50b':(50,9413,'ws')}
    n,number,kind=specs[case]
    rng=random.Random(number)
    graph=nx.barabasi_albert_graph(n,2,seed=number) if kind=='ba' else nx.watts_strogatz_graph(n,4,.25,seed=number)
    nodes=[]
    for i in range(n):
        persona=rng.choices(['和平','中立','暴力'],[.35,.35,.3])[0]
        wrange={'和平':(5,25),'中立':(-10,10),'暴力':(-45,-15)}[persona]
        rrange={'和平':(1.1,1.6),'中立':(.8,1.2),'暴力':(.1,.4)}[persona]
        nodes.append({'id':i+1,'w':round(rng.uniform(*wrange),5),'r':round(rng.uniform(*rrange),5),
                      'persona':persona,'comm_left':3})
    return {'global_setting':{'max_budget':n*2,'max_api_calls':120},
            'original_total':sum(node['w'] for node in nodes),'nodes':nodes,
            'edges':[[u+1,v+1] for u,v in graph.edges()],
            'prompts':{'1':15.0,'2':10.0,'3':-5.0}}


class LoggedEnv:
    def __init__(self,delegate):
        self.delegate=delegate
        self.actions=[]
    def get_remaining_budget(self):return self.delegate.get_remaining_budget()
    def _action(self,name,*args):
        result=getattr(self.delegate,name)(*args)
        self.actions.append({'action':name,'args':args,'result':result})
        return result
    def scan_node(self,i):return self._action('scan_node',i)
    def communicate(self,i,p):return self._action('communicate',i,p)
    def cut_link(self,u,v):return self._action('cut_link',u,v)
    def shield_node(self,i):return self._action('shield_node',i)


def run(case,strategy):
    key=os.environ.get('ZHIPUAI_API_KEY','').strip()
    if not key:raise RuntimeError('Missing ZHIPUAI_API_KEY')
    seed=make_seed(case)
    folder=ROOT/'agent_notes/experiments/round2'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/(case+'-seed.json')).write_text(json.dumps(seed,ensure_ascii=False,indent=2),encoding='utf-8')
    server='http://8.222.218.162:5000'
    base=RemoteStarNetEnv(server,seed)
    natural=base.trigger_eval()
    env=LoggedEnv(RemoteStarNetEnv(server,seed))
    people=json.loads((ROOT/'workspace/team_submission/config.json').read_text(encoding='utf-8'))['person']
    people=copy.deepcopy(people)
    limit=seed['global_setting']['max_api_calls']
    people[0]['runtime']={'strategy':strategy,'node_count':len(seed['nodes']),
                          'operation_limit':limit,'llm_limit':limit,'prompt_ids':[1,2,3]}
    llm=ZhipuLLM(key,tar_len=5)
    start=time.monotonic()
    model=ParticipantSquadModel(env,people,llm)
    step_calls=0
    print(json.dumps({'case':case,'strategy':strategy,'natural_score':natural},ensure_ascii=False),flush=True)
    try:
        while step_calls<limit:
            status=model.step()
            step_calls+=1
            if step_calls%10==0 or status:
                print(json.dumps({'step':step_calls,'budget':env.get_remaining_budget(),
                                  'llm_calls':model.llm_counter.call_count,'stop':bool(status)},ensure_ascii=False),flush=True)
            if status:break
        else:raise RuntimeError('Runner reached fuse without stop')
    except APICreditExhausted:
        print('CREDIT_EXHAUSTED: 立即停止任务，请充值。',flush=True)
        raise
    remaining_budget=env.get_remaining_budget()
    # Official evaluate consumes the session: gather all state before it.
    result={'case':case,'strategy':strategy,'natural_score':natural,
            'remaining_budget':remaining_budget,
            'steps':step_calls,'llm_calls':model.llm_counter.call_count,
            'http_llm_calls':llm.http_request_count,'stop_reason':model.stop_reason,
            'elapsed_seconds':round(time.monotonic()-start,3),'actions':env.actions,
            'seed_sha256':hashlib.sha256(json.dumps(seed,sort_keys=True).encode()).hexdigest(),
            'model_sha256':hashlib.sha256((ROOT/'workspace/team_submission/starnet_model.py').read_bytes()).hexdigest()}
    path=folder/(case+'-'+strategy+'.json')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    result['final_score']=env.delegate.trigger_eval()
    result['gain']=result['final_score']-natural
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='actions'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=['official','develop10a','develop10b','validate50a','validate50b'],required=True)
    parser.add_argument('--strategy',choices=['baseline','score'],required=True)
    args=parser.parse_args()
    run(args.case,args.strategy)
