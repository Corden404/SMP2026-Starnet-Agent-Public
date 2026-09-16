"""Round 3: disjoint synthetic manifests, labeled proxy screening and real sandbox pairs.

The model receives ONLY the five public environment methods. Synthetic hidden
attributes remain in the test environment. Proxy results are never leaderboard scores.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace

import networkx as nx
from api_client import RemoteStarNetEnv
from benchmark import LoggedEnv
from team_submission.starnet_model import ParticipantSquadModel, CountingLLM
from zhipu import ZhipuLLM, APICreditExhausted

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'agent_notes/experiments/round3'
SOURCE = ROOT / 'workspace/team_submission'
SERVER = 'http://8.222.218.162:5000'
VARIANTS = {
    'score': {'strategy': 'score'},
    'learning': {'strategy': 'planner', 'planner': {'decision_mode': 'roi', 'learning': True}},
    'plan': {'strategy': 'planner', 'planner': {'learning': False}},
    'combined': {'strategy': 'planner'},
    'adaptive': {'strategy': 'planner', 'planner': {'scan_policy': 'adaptive'}},
    'tight': {'strategy': 'planner', 'planner': {'plan_tolerance': 0.0}},
    'wide': {'strategy': 'planner', 'planner': {'plan_tolerance': .08}},
    'partial70': {'strategy': 'planner', 'experiment_override': 'fixed_70_percent_scan'},
    'deep': {'strategy': 'planner', 'planner': {'depth': 3}},
    'roi_tight': {'strategy': 'planner', 'planner': {'decision_mode': 'roi', 'roi_floor': 1.0}},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def code_hash():
    return digest({str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(SOURCE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts})


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def seed_for(case):
    group, index = case.split('-')
    index = int(index)
    number = {'dev': 370000, 'holdout': 910000, 'stress': 730000}[group] + index * 101
    n = 100 if group == 'stress' else 50
    rng = random.Random(number)
    kind = index % 6
    if kind == 0:
        graph = nx.barabasi_albert_graph(n, 2, seed=number)
    elif kind == 1:
        graph = nx.watts_strogatz_graph(n, 4, .25, seed=number)
    elif kind == 2:
        graph = nx.gnp_random_graph(n, .12, seed=number)
    elif kind == 3:
        graph = nx.stochastic_block_model([n//2, n-n//2], [[.17,.008],[.008,.17]], seed=number)
    elif kind == 4:
        graph = nx.disjoint_union(nx.barabasi_albert_graph(n//2, 1, seed=number),
                                  nx.watts_strogatz_graph(n-n//2, 4, .1, seed=number+1))
    else:
        graph = nx.random_tree(n, seed=number) if hasattr(nx, 'random_tree') else nx.random_labeled_tree(n, seed=number)
    # Relabel every graph randomly so ascending IDs do not reveal hub positions.
    ids = list(range(1, n+1))
    rng.shuffle(ids)
    graph = nx.relabel_nodes(graph, {i: ids[i] for i in range(n)})
    distribution = [[.35,.35,.30],[.2,.2,.6],[.65,.25,.1]][(index//6)%3]
    scale = [.55, 1.0, 1.5][index%3]
    nodes = []
    for i in range(1,n+1):
        persona = rng.choices(['和平','中立','暴力'], distribution)[0]
        lo,hi = {'和平':(5,30), '中立':(-12,12), '暴力':(-50,-12)}[persona]
        rlo,rhi = {'和平':(.8,1.8), '中立':(.5,1.4), '暴力':(.05,.5)}[persona]
        nodes.append({'id':i,'w':round(rng.uniform(lo,hi),5),
                      'r':round(rng.uniform(rlo,rhi)*scale,5),'persona':persona,'comm_left':3})
    return {'global_setting':{'max_budget':2*n,'max_api_calls':250 if n==100 else 120},
            'original_total':sum(d['w'] for d in nodes),'nodes':nodes,
            'edges':[list(e) for e in sorted(tuple(sorted(e)) for e in graph.edges())],
            'prompts':{'1':15.,'2':10.,'3':-5.}}


def manifest():
    cases = [f'dev-{i:03}' for i in range(30)] + [f'holdout-{i:03}' for i in range(20)] + [f'stress-{i:03}' for i in range(6)]
    seeds = {case:seed_for(case) for case in cases}
    hashes = {c:digest(s) for c,s in seeds.items()}
    assert len(set(hashes.values())) == len(hashes)
    path = OUT/'manifest.json'
    data = {'purpose':'Development and held-out SYNTHETIC distributions; not hidden competition seeds',
            'seeds':hashes, 'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8'))['seeds'] == hashes, 'Manifest drift'
    else:
        save(path,data)
    for c,s in seeds.items():
        p = OUT/'seeds'/(c+'.json')
        if not p.exists():save(p,s)
    return cases


class ProxyEnv:
    """Hypothesized dynamics for screening, never an official score."""
    def __init__(self, seed):
        self._nodes = {d['id']:copy.deepcopy(d) for d in seed['nodes']}
        self._edges = {tuple(sorted(e)) for e in seed['edges']}
        self._budget = float(seed['global_setting']['max_budget'])
        self._prompts = seed['prompts']
        self._scanned = set()
    def get_remaining_budget(self):return self._budget
    def scan_node(self,i):
        assert i not in self._scanned and self._budget >= .5
        self._budget -= .5
        self._scanned.add(i)
        if i not in self._nodes:return None
        d=self._nodes[i]
        return {k:d[k] for k in ('w','persona','comm_left')} | {'neighbors':[v if u==i else u for u,v in self._edges if i in (u,v)]}
    def communicate(self,i,p):
        assert i in self._scanned and i in self._nodes and self._budget>=2
        d=self._nodes[i]
        assert d['comm_left']>0 and str(p) in self._prompts
        d['w']+=self._prompts[str(p)]*d['r']*.5**(3-d['comm_left'])
        d['comm_left']-=1
        self._budget-=2
        return {'status':'success','new_w':d['w']}
    def cut_link(self,u,v):
        edge=tuple(sorted((u,v)))
        assert u in self._scanned and v in self._scanned and edge in self._edges and self._budget>=3
        self._edges.remove(edge)
        self._budget-=3
        return True
    def shield_node(self,i):
        assert i in self._scanned and i in self._nodes and self._budget>=5
        del self._nodes[i]
        self._edges={e for e in self._edges if i not in e}
        self._budget-=5
        return True
    def evaluate(self):return ParticipantSquadModel.estimate_score(self._nodes,self._edges)[0]


class TopChoiceLLM:
    """Deterministic proxy policy: chooses first candidate. No real LLM claim."""
    http_request_count=0
    def send_message(self,prompt,json_flag=False):return '{"index":0,"targets":[]}'


class PartialScanAblation(ParticipantSquadModel):
    """Test-only fixed exploration budget; not silently enabled in submissions."""
    def _should_scan(self, budget):
        return self._next_scan_target() is not None and len(self.local_nodes) < math.ceil(.7*self.node_count)


def fast_model(env, people, model_class=ParticipantSquadModel):
    # Only for proxy screening; real runs and integration tests use __init__ and CaseVO.
    m=model_class.__new__(model_class)
    opts=m._load_settings(people,env.get_remaining_budget())
    m.strategy=opts['strategy'];m.planner_options=opts['planner']
    m.scan_queue=[];m.comm_effects={};m.persona_observations={};m.scan_closed=False
    m.action_history=[];m.decision_history=[]
    m.node_count=opts['node_count'];m.operation_limit=opts['operation_limit'];m.llm_limit=opts['llm_limit'];m.prompt_ids=opts['prompt_ids']
    m.operation_count=0;m.no_progress_count=0;m.stopped=False;m.stop_reason=''
    m.env=env;m.llm_counter=CountingLLM(TopChoiceLLM(),m.llm_limit)
    m.local_nodes={};m.local_edges=set();m.dead_nodes=set();m.schedule=SimpleNamespace(time=0,steps=0)
    class Chain:
        def _run_chain(self,*args):
            return json.loads(m.llm_counter.send_message('offline proxy'))
    m.scout=Chain();m.commander=Chain()
    return m


def run_one(case,variant,remote=False,repeat=0,output=None):
    seed=seed_for(case)
    source_hash=code_hash()
    path=(output or OUT/('remote' if remote else 'proxy'))/(case+'-'+variant+f'-r{repeat}.json')
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old.get('source_hash')==source_hash and old.get('variant_config')==VARIANTS[variant] and 'final_score' in old:
            print(json.dumps({'cached':str(path),'score':old['final_score']}),flush=True)
            return old
        raise RuntimeError('Existing result differs; use a new output directory: '+str(path))
    people=json.loads((SOURCE/'config.json').read_text(encoding='utf-8-sig'))['person']
    limit=seed['global_setting']['max_api_calls']
    people[0]['runtime']={'node_count':len(seed['nodes']),'operation_limit':limit,'llm_limit':limit,
                           'prompt_ids':[1,2,3],**copy.deepcopy(VARIANTS[variant])}
    if remote:
        key=os.environ.get('ZHIPUAI_API_KEY','').strip()
        if not key:raise RuntimeError('Missing API key')
        base_path=OUT/'natural'/(case+'.json')
        if base_path.exists():
            base=json.loads(base_path.read_text(encoding='utf-8'))
            assert base['seed_sha256']==digest(seed)
            natural=base['score']
        else:
            natural=RemoteStarNetEnv(SERVER,seed,timeout=30.0).trigger_eval()
            save(base_path,{'seed_sha256':digest(seed),'score':natural,'time':time.time()})
        env=LoggedEnv(RemoteStarNetEnv(SERVER,seed,timeout=30.0))
        llm=ZhipuLLM(key,5)
    else:
        raw=ProxyEnv(seed);natural=raw.evaluate();env=LoggedEnv(raw);llm=TopChoiceLLM()
    start=time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        model_class = PartialScanAblation if variant=='partial70' else ParticipantSquadModel
        model=model_class(env,people,llm) if remote else fast_model(env,people,model_class)
    print(json.dumps({'start':case,'variant':variant,'remote':remote,'natural':natural}),flush=True)
    step=0
    while step<limit:
        step+=1
        if remote:
            status=model.step()
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                status=model.step()
        if remote and step % 20 == 0:
            print(json.dumps({'progress':case,'variant':variant,'step':step,'llm_calls':model.llm_counter.call_count}),flush=True)
        if status:break
    if step>=limit:raise RuntimeError('Model reached fuse without returning early')
    budget=env.get_remaining_budget()
    result={'case':case,'variant':variant,'variant_config':VARIANTS[variant],
        'evaluation':'official sandbox + real glm-4-plus' if remote else 'PROXY estimator + deterministic first-choice policy; NOT official score',
        'repeat':repeat,'natural_score':natural,'steps':step,'llm_calls':model.llm_counter.call_count,
        'http_llm_calls':llm.http_request_count,'remaining_budget':budget,'stop_reason':model.stop_reason,
        'elapsed_seconds':round(time.monotonic()-start,3),'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_hash':source_hash,'seed_sha256':digest(seed),
        'actions':env.actions,'decisions':getattr(model,'decision_history',[])}
    save(path,result)
    result['final_score']=env.delegate.trigger_eval() if remote else env.delegate.evaluate()
    result['gain']=result['final_score']-natural
    save(path,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('actions','decisions')},ensure_ascii=False),flush=True)
    return result


def summarize(folder):
    rows=[json.loads(p.read_text(encoding='utf-8')) for p in folder.glob('*.json')]
    rows=[r for r in rows if 'variant' in r and 'final_score' in r]
    groups={}
    base={(r['case'],r['repeat']):r for r in rows if r['variant']=='score'}
    for variant in sorted({r['variant'] for r in rows}):
        rs=[r for r in rows if r['variant']==variant]
        deltas=[r['final_score']-base[(r['case'],r['repeat'])]['final_score'] for r in rs if (r['case'],r['repeat']) in base]
        means=[r['final_score'] for r in rs]
        rng=random.Random(71234)
        boot=sorted(statistics.mean(rng.choices(deltas,k=len(deltas))) for _ in range(3000)) if deltas else []
        groups[variant]={'n':len(rs),'mean':statistics.mean(means),'mean_gain_over_natural':statistics.mean(r['gain'] for r in rs),
            'paired_n':len(deltas),'paired_delta':statistics.mean(deltas) if deltas else None,
            'paired_bootstrap_95':[boot[int(.025*len(boot))],boot[int(.975*len(boot))]] if boot else None,
            'paired_wins':sum(x>0 for x in deltas),'paired_losses':sum(x<0 for x in deltas),
            'min':min(means),'max_steps':max(r['steps'] for r in rs),'max_calls':max(r['llm_calls'] for r in rs),
            'mean_scans':statistics.mean(sum(a['action']=='scan_node' for a in r['actions']) for r in rs)}
    save(folder/'summary.json',groups)
    print(json.dumps(groups,ensure_ascii=False,indent=2))


def calibrate():
    results=[]
    for case in ['dev-000','dev-001','dev-002','dev-003','dev-004','dev-005']:
        seed=seed_for(case)
        real=RemoteStarNetEnv(SERVER,seed,timeout=30.0).trigger_eval()
        proxy=ProxyEnv(seed).evaluate()
        results.append({'case':case,'official':real,'proxy':proxy,'error':real-proxy})
        print(json.dumps(results[-1]),flush=True)
        save(OUT/'calibration.json',results)
    for pid in (1,2,3):
        seed=seed_for('dev-000')
        env=RemoteStarNetEnv(SERVER,seed,timeout=30.0)
        node=seed['nodes'][0]
        observed=env.scan_node(node['id'])
        reply=env.communicate(node['id'],pid)
        delta=reply['new_w']-observed['w']
        results.append({'prompt_id':pid,'actual_delta':delta,'expected_delta':node['r']*seed['prompts'][str(pid)]})
        env.trigger_eval()
        save(OUT/'calibration.json',results)
        print(json.dumps(results[-1]),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['manifest','proxy','remote','summary','calibrate'],required=True)
    parser.add_argument('--cases',nargs='+',default=['dev-000'])
    parser.add_argument('--variants',nargs='+',choices=list(VARIANTS),default=['score','combined'])
    parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--no-summary',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    manifest()
    if args.mode=='manifest':return
    if args.mode=='summary':return summarize(args.output or OUT/'proxy')
    if args.mode=='calibrate':return calibrate()
    cases=args.cases
    if cases==['dev']:cases=[f'dev-{i:03}' for i in range(30)]
    elif cases==['holdout']:cases=[f'holdout-{i:03}' for i in range(20)]
    elif cases==['stress']:cases=[f'stress-{i:03}' for i in range(6)]
    for case in cases:
        for variant in args.variants:
            run_one(case,variant,args.mode=='remote',args.repeat,args.output)
    if not args.no_summary:
        summarize(args.output or OUT/('remote' if args.mode=='remote' else 'proxy'))

if __name__=='__main__':
    try:main()
    except APICreditExhausted:
        print('CREDIT_EXHAUSTED: task stopped immediately; recharge required.',flush=True)
        sys.exit(13)
