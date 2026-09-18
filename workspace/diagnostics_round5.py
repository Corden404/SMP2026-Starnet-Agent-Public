"""P3 diagnostic-only variants; oracle inputs are explicitly labelled and never shipped."""
import contextlib
import copy
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from experiment_round5 import OUT, ROOT, F0, FiniteRollout, FiniteEnv, PublicEnv, run as run_reference
from experiment_round3 import fast_model
from benchmark import LoggedEnv
from round4_common import save,digest,file_hash
from round4_seeds import load_case

SEARCH={'rollout_seconds':3.,'rollout_width':2,'rollout_samples':1,'rollout_learning':False,'rollout_combinations':False,'rollout_margin':.5}


def update_response(effects,observations,node,before,after):
    first=(after-before['w'])/(.5**max(0,3-before['comm_left']))
    effects[node]=first;observations.setdefault(before['persona'],{})[node]=first


class LearningRollout(FiniteRollout):
    def apply(self,state,action,latent,learn):
        result=super().apply(state,action,latent,False)
        if learn and action['action']=='comm':
            i=action['target_node_1'];update_response(result[2],result[3],i,state[0][i],result[0][i]['w'])
        return result


class L0(F0):
    def _execute_planned_action(self,action,budget,before):
        result=super()._execute_planned_action(action,budget,before)
        if action['action']=='comm' and self.action_history[-1]['success']:
            i=action['target_node_1'];update_response(self.comm_effects,self.persona_observations,i,before,self.local_nodes[i]['w'])
        return result


class Diagnostic(L0):
    def _effect_estimate(self,i,data):
        if self.oracle is not None:return self.oracle[i],0.
        return super()._effect_estimate(i,data)

    def _learned_candidates(self,budget):
        reference=super()._learned_candidates(budget)
        if not self.search or not reference:return reference
        slots=min(self.operation_limit-1-self.operation_count,self.llm_counter.safe_limit-self.llm_counter.call_count)
        effects=self.comm_effects if self.oracle is None else self.oracle
        rollout=FiniteRollout(self.local_nodes,self.local_edges,effects,self.persona_observations,SEARCH,slots)
        start=time.monotonic();plans=rollout.choose(budget,baseline_action=reference[0])
        self.search_records.append({'operation':self.operation_count,'seconds':time.monotonic()-start,'plans':plans})
        return [plans[0]['first']] if plans else reference


def run(case,variant):
    if json.loads((OUT/'mechanism-validation.json').read_text(encoding='utf-8'))['status']!='PASSED':raise RuntimeError('P1 gate')
    if not (OUT/'P2-analysis.json').exists():raise RuntimeError('Complete P2 first')
    spec,seed=load_case(case)
    if spec['pool'] not in ('dev','corr','select'):raise ValueError('Development only')
    people=json.loads((OUT/'baseline/config.json').read_text(encoding='utf-8'))['person']
    identity={'case':case,'seed_sha256':digest(seed),'variant':variant,'runner_sha256':file_hash(__file__),
        'simulation_sha256':file_hash(ROOT/'workspace/experiment_round5.py'),'search':SEARCH,
        'selector':'deterministic-first','purpose':'diagnostic oracle' if variant in ('D-response','D-both') else 'development',
        'repeat':0,'group':case,'source_baseline':file_hash(OUT/'baseline/starnet_model.py')}
    path=OUT/'P3'/(case+'-'+variant+'.json')
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'));assert old['identity']==identity;return old
    raw=FiniteEnv(seed);logged=LoggedEnv(raw);model=fast_model(PublicEnv(logged),people,Diagnostic)
    model.oracle={d['id']:seed['prompts']['1']*d['r'] for d in seed['nodes']} if variant in ('D-response','D-both') else None
    model.search=variant in ('D-search','D-both');model.search_records=[]
    start=time.monotonic();states=[]
    with contextlib.redirect_stdout(io.StringIO()):
        for step in range(seed['global_setting']['max_api_calls']):
            states.append({'nodes':copy.deepcopy(model.local_nodes),'edges':sorted(model.local_edges),
                'effects':dict(model.comm_effects),'observations':copy.deepcopy(model.persona_observations),'budget':raw.get_remaining_budget(),
                'operation_count':model.operation_count,'logical_calls':model.llm_counter.call_count,'prefix_length':len(logged.actions)})
            if model.step():break
        else:raise RuntimeError('Resource fuse')
    result={'identity':identity,'case':case,'variant':variant,'spec':spec,'final_score':raw.evaluate(),'elapsed_seconds':time.monotonic()-start,
        'actions':logged.actions,'states':states,'search_records':model.search_records,'remaining_budget':raw.get_remaining_budget(),
        'stop_reason':model.stop_reason,'logical_calls':model.llm_counter.call_count,'glm_http_calls':0,'tokens':0}
    save(path,result);return result


def worker(job):
    result=run(*job)
    return {k:result[k] for k in ('case','variant','final_score','elapsed_seconds')}


def main(workers=4):
    jobs=[(f'{pool}-{i:03}',variant) for pool,n in [('dev',108),('corr',72),('select',54)] for i in range(n) for variant in ['D-response','D-search','D-both']]
    timing=[worker(j) for j in jobs[:6]]
    save(OUT/'P3-timing.json',{'first_two_seeds':timing,'remaining_estimate_seconds':sum(r['elapsed_seconds'] for r in timing)/6*(len(jobs)-6)/workers,'workers':workers})
    print(json.dumps({'timing':timing}),flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,result in enumerate(pool.map(worker,jobs[6:])):
            if i%20==19:print(json.dumps({'completed':i+7,'last':result}),flush=True)
    from analyze_round5 import paired
    rows=[json.loads(p.read_text(encoding='utf-8')) for folder in ('P2','P3') for p in (OUT/folder).glob('*.json')]
    results={}
    for pool in ('development','select'):
        data=[r for r in rows if (r['spec']['pool']=='select')==(pool=='select')]
        results[pool]={variant:paired(data,candidate=variant) for variant in ['F0','D-response','D-search','D-both']}
    save(OUT/'loss-decomposition.json',{'results':results,'search_scope':SEARCH,'oracle_is_not_submit_eligible':True,'glm_calls':0,'tokens':0})


if __name__=='__main__':main()
