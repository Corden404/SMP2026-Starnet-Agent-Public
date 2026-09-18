"""Same-information combination search diagnostic on existing development seeds only."""
import contextlib
import io
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import diagnostics_round5 as diagnostic
from experiment_round5 import OUT, ROOT, FiniteEnv, PublicEnv
from experiment_round3 import fast_model
from benchmark import LoggedEnv
from round4_common import digest,file_hash,save
from round4_seeds import load_case

# Isolate combination prefixes: all other mean-only bounded-search settings unchanged.
OPTIONS={**diagnostic.SEARCH,'rollout_combinations':True}
diagnostic.SEARCH=OPTIONS


def run(case):
    if not (OUT/'loss-decomposition.json').exists():raise RuntimeError('Finish existing diagnostic batch first')
    spec,seed=load_case(case)
    if spec['pool'] not in ('dev','corr','select'):raise ValueError('Existing development seeds only')
    people=json.loads((OUT/'baseline/config.json').read_text(encoding='utf-8'))['person']
    identity={'case':case,'seed_sha256':digest(seed),'variant':'D-combination','options':OPTIONS,
        'source_sha256':file_hash(__file__),'diagnostic_sha256':file_hash(ROOT/'workspace/diagnostics_round5.py'),
        'scorer_sha256':file_hash(ROOT/'workspace/experiment_round5.py'),'baseline_sha256':file_hash(OUT/'baseline/starnet_model.py'),
        'config_sha256':digest(people),'selector':'deterministic-first','purpose':'same-information development diagnosis','repeat':0}
    path=OUT/'P3-combination'/(case+'.json')
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'));assert old['identity']==identity;return old
    raw=FiniteEnv(seed);logged=LoggedEnv(raw);model=fast_model(PublicEnv(logged),people,diagnostic.Diagnostic)
    model.oracle=None;model.search=True;model.search_records=[]
    start=time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        for step in range(seed['global_setting']['max_api_calls']):
            if model.step():break
        else:raise RuntimeError('Resource fuse')
    result={'identity':identity,'case':case,'variant':'D-combination','spec':spec,'final_score':raw.evaluate(),
        'elapsed_seconds':time.monotonic()-start,'actions':logged.actions,'search_records':model.search_records,
        'remaining_budget':raw.get_remaining_budget(),'stop_reason':model.stop_reason,
        'logical_calls':model.llm_counter.call_count,'glm_http_calls':0,'tokens':0}
    save(path,result);return result


def worker(case):
    r=run(case);return {k:r[k] for k in ('case','final_score','elapsed_seconds')}


def main():
    cases=[f'select-{i:03}' for i in range(54)]
    timing=[worker(c) for c in cases[:2]]
    save(OUT/'combination-timing.json',{'first_two':timing,'remaining_estimate_seconds':sum(r['elapsed_seconds'] for r in timing)/2*52/4,'workers':4})
    print(json.dumps({'timing':timing}),flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(worker,cases[2:])):
            if i%10==9:print(json.dumps({'completed':i+3,'last':result}),flush=True)


if __name__=='__main__':main()
