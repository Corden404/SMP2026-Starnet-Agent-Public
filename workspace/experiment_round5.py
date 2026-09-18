"""Experimental finite50 models. Does not change the submitted/default strategy."""
import contextlib
import copy
from functools import lru_cache
import importlib.util
import io
import json
import time

from round4_common import ROOT, save, digest, file_hash
from experiment_round3 import ProxyEnv, fast_model
from benchmark import LoggedEnv
from round4_seeds import load_case
from team_submission.starnet_model import BudgetRollout

OUT=ROOT/'agent_notes/experiments/round5'
MODEL_VERSION='finite50-sparse-transpose-v1'


@lru_cache(maxsize=2048)
def coefficients(ids, edges, mode=MODEL_VERSION):
    adj={i:{i} for i in ids}
    for u,v in edges:
        if u in adj and v in adj:adj[u].add(v);adj[v].add(u)
    if mode=='stationary':
        remaining=set(ids);c={}
        while remaining:
            stack=[remaining.pop()];component=[]
            while stack:
                i=stack.pop();component.append(i)
                for j in adj[i]&remaining:remaining.remove(j);stack.append(j)
            denominator=sum(len(adj[i]) for i in component)
            for i in component:c[i]=len(component)*len(adj[i])/denominator
        return tuple(c[i] for i in ids)
    if mode!=MODEL_VERSION:raise ValueError(mode)
    c={i:1. for i in ids}
    for _ in range(50):
        nxt={i:0. for i in ids}
        for i,neighbors in adj.items():
            weight=c[i]/len(neighbors)
            for j in neighbors:nxt[j]+=weight
        c=nxt
    return tuple(c[i] for i in ids)


def score(nodes, edges, mode='finite50'):
    ids=tuple(sorted(nodes));effective=tuple(sorted({tuple(sorted((u,v))) for u,v in edges if u in nodes and v in nodes and u!=v}))
    c=coefficients(ids,effective,MODEL_VERSION if mode=='finite50' else mode)
    return sum(nodes[i]['w']*value for i,value in zip(ids,c)),dict(zip(ids,c))


def baseline_class():
    spec=importlib.util.spec_from_file_location('round5_frozen_baseline',OUT/'baseline/starnet_model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.ParticipantSquadModel


B0=baseline_class()


class F0(B0):
    estimate_score=staticmethod(score)
    _fast_score=staticmethod(score)


class FiniteEnv(ProxyEnv):
    def evaluate(self):
        # Independent forward implementation, not the planner's cached transpose.
        from mechanism_round5 import score as direct
        return direct({'nodes':list(self._nodes.values()),'edges':list(self._edges)})


class PublicEnv:
    """Expose only public methods, never seed, latent response, or environment state."""
    __slots__=('get_remaining_budget','scan_node','communicate','cut_link','shield_node')
    def __init__(self,env):
        for name in self.__slots__:setattr(self,name,getattr(env,name))


class FiniteRollout(BudgetRollout):
    @staticmethod
    def metrics(nodes,edges):
        total,influence=score(nodes,edges)
        adj={i:set() for i in nodes}
        for u,v in edges:
            if u in adj and v in adj:adj[u].add(v);adj[v].add(u)
        return total,influence,adj,{}

    def candidates(self,state,budget,negative=False):
        nodes,edges,effects,observations=state
        total,influence=score(nodes,edges);actions=[]
        def add(kind,u,v,gain):
            if negative or gain>.25:
                actions.append({'action':kind,'target_node_1':u,'target_node_2':v,
                    'prompt_id':1 if kind=='comm' else None,'cost':self.COST[kind],
                    'gain':gain,'roi':gain/self.COST[kind]})
        if budget>=2:
            for i,d in nodes.items():
                if d['comm_left']>0:add('comm',i,None,influence[i]*self.effect(i,d,effects,observations)[0]*.5**(3-d['comm_left']))
        if budget>=5 and len(nodes)>1:
            for i in nodes:
                ns={j:d for j,d in nodes.items() if j!=i};es={e for e in edges if i not in e}
                add('shield',i,None,score(ns,es)[0]-total)
        if budget>=3:
            for u,v in sorted(edges):
                if u in nodes and v in nodes:add('cut',u,v,score(nodes,edges-{(u,v)})[0]-total)
        actions.sort(key=lambda a:(-a['roi'],-a['gain'],a['target_node_1']))
        return actions


def run(case,variant,folder='P2',force=False):
    validation=json.loads((OUT/'mechanism-validation.json').read_text(encoding='utf-8'))
    if validation['status']!='PASSED':raise RuntimeError('P1 must pass before P2/P3 experiments')
    spec,seed=load_case(case)
    if spec['pool'] not in ('dev','corr','select'):raise ValueError('Only development pools allowed')
    model_class={'B0':B0,'F0':F0}[variant]
    people=json.loads((OUT/'baseline/config.json').read_text(encoding='utf-8'))['person']
    identity={'seed_sha256':digest(seed),'base_graph_group':case,'variant':variant,'simulator_version':MODEL_VERSION,
        'belief_version':'frozen-B0','continuation_policy':variant,'selector':'deterministic-first','repeat':0,'purpose':'historical-development',
        'dependencies':{name:file_hash(ROOT/'workspace'/name) for name in ('experiment_round3.py','mechanism_round5.py','round4_common.py','round4_seeds.py','benchmark.py','team_submission/starnet_model.py')},
        'baseline_sha256':file_hash(OUT/'baseline/starnet_model.py'),'runner_sha256':file_hash(__file__),'config_sha256':digest(people)}
    path=OUT/folder/(case+'-'+variant+'.json')
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old['identity']!=identity:raise RuntimeError('Existing identity mismatch')
        return old
    raw=FiniteEnv(seed);logged=LoggedEnv(raw);model=fast_model(PublicEnv(logged),people,model_class)
    start=time.monotonic();states=[];steps=[]
    with contextlib.redirect_stdout(io.StringIO()):
        for step in range(seed['global_setting']['max_api_calls']):
            states.append({'nodes':copy.deepcopy(model.local_nodes),'edges':sorted(model.local_edges),
                'effects':dict(model.comm_effects),'observations':copy.deepcopy(model.persona_observations),'budget':raw.get_remaining_budget(),
                'operation_count':model.operation_count,'logical_calls':model.llm_counter.call_count,'prefix_length':len(logged.actions)})
            before=time.monotonic();stop=model.step();steps.append(time.monotonic()-before)
            if stop:break
        else:raise RuntimeError('Resource fuse reached')
    result={'identity':identity,'case':case,'variant':variant,'spec':spec,'final_score':raw.evaluate(),
        'elapsed_seconds':time.monotonic()-start,'step_seconds':steps,'actions':logged.actions,'states':states,
        'decisions':model.decision_history,'remaining_budget':raw.get_remaining_budget(),'stop_reason':model.stop_reason,
        'logical_calls':model.llm_counter.call_count,'glm_http_calls':0,'tokens':0}
    save(path,result)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--cases',nargs='+',required=True);p.add_argument('--variants',nargs='+',default=['B0','F0'])
    args=p.parse_args()
    for case in args.cases:
        for variant in args.variants:
            r=run(case,variant);print(json.dumps({k:r[k] for k in ('case','variant','final_score','elapsed_seconds')}),flush=True)
