"""Public custom-seed branch diagnostics; deterministic continuation, NOT leaderboard."""
import argparse
import contextlib
import copy
import io
import json
import time

from api_client import RemoteStarNetEnv
from experiment_round3 import fast_model
from experiment_round4 import AuditedEnv, SERVER, VARIANTS
from round4_common import OUT, SOURCE, digest, save, source_files
from round4_seeds import load_case


def canonical(value):
    if isinstance(value,dict):
        return {k:sorted(v) if k=='neighbors' else canonical(v) for k,v in value.items()}
    if isinstance(value,list):return [canonical(v) for v in value]
    if isinstance(value,float):return round(value,7)
    return value


def hydrate(env,row,state):
    people=json.loads((SOURCE/'config.json').read_text(encoding='utf-8-sig'))['person']
    people[0]['runtime']={'node_count':row['spec']['n'],'operation_limit':120,'llm_limit':120,
                          'prompt_ids':[1,2,3],**copy.deepcopy(VARIANTS['B0'])}
    m=fast_model(env,people)
    m.local_nodes={int(k):v for k,v in copy.deepcopy(state['nodes']).items()}
    m.local_edges={tuple(e) for e in state['edges']}
    m.comm_effects={int(k):v for k,v in state['comm_effects'].items()}
    m.persona_observations={p:{int(k):v for k,v in values.items()} for p,values in state['persona_observations'].items()}
    m.operation_count=state['first_action'];m.llm_counter.call_count=state['llm_calls']
    m.dead_nodes={a['args'][0] for a in row['actions'][:state['first_action']] if a['action']=='shield_node' and a.get('result') is True}
    return m


def branch(row,state,action,label,output):
    if output.exists():raise RuntimeError('Refusing to overwrite a branch run; use a new output batch')
    spec,seed=load_case(row['case'])
    record={'case':row['case'],'label':label,'prefix_sha256':digest(row['actions'][:state['first_action']]),
            'state_sha256':digest(state),'action':action,'status':'running','started':time.time(),
            'evaluation':'official custom-seed sandbox; deterministic B0 continuation; no real LLM',
            'submission_files':source_files(),'seed_sha256':digest(seed)}
    save(output,record);env=None
    try:
        env=AuditedEnv(RemoteStarNetEnv(SERVER,seed,timeout=30))
        for expected in row['actions'][:state['first_action']]:
            actual=getattr(env,expected['action'])(*expected['args'])
            if canonical(actual)!=canonical(expected['result']):raise RuntimeError('Prefix receipt mismatch')
        if abs(env.get_remaining_budget()-state['budget'])>1e-7:raise RuntimeError('Prefix budget mismatch')
        m=hydrate(env,row,state)
        before=copy.deepcopy(m.local_nodes[action['target_node_1']])
        # One logical decision slot reserved for the forced first action; no HTTP LLM claim.
        m.llm_counter.call_count+=1
        with contextlib.redirect_stdout(io.StringIO()):
            status=m._execute_planned_action(action,state['budget'],before)
            for _ in range(120):
                if status:break
                status=m.step()
            else:raise RuntimeError('Continuation fuse')
        if '不确定' in m.stop_reason or 'unavailable' in m.stop_reason.lower():raise RuntimeError('Uncertain continuation')
        record.update(actions=env.actions,budget_left=env.get_remaining_budget(),
                      proxy_final=m._fast_score(m.local_nodes,m.local_edges)[0],stop_reason=m.stop_reason,
                      logical_decision_slots=m.llm_counter.call_count,actual_llm_http_calls=0,
                      status='awaiting_evaluation')
        save(output,record)
        record.update(final_score=env.delegate.trigger_eval(),status='completed',seconds=time.time()-record['started'])
        save(output,record)
        return record
    except BaseException as exc:
        record.update(status='failed',error_type=type(exc).__name__)
        if env is not None:record['actions']=env.actions
        save(output,record)
        raise


def run_diagnostics(input_batch,output_batch,cases):
    summary=[]
    for case in cases:
        row=json.loads((OUT/input_batch/f'{case}-B0-r0-a0.json').read_text(encoding='utf-8'))
        states=[s for s in row['states'] if len(s['nodes'])+sum(a['action']=='shield_node' for a in row['actions'][:s['first_action']])==row['spec']['n'] and s['budget']>=5]
        # Preregistered first post-scan and first <=10-budget states; not cherry-picked by gains.
        chosen=[states[0]] if states else []
        tail=next((s for s in states if s['budget']<=10 and s!=states[0]),None)
        if tail:chosen.append(tail)
        for idx,state in enumerate(chosen):
            class Budget:
                def get_remaining_budget(self):return state['budget']
            m=hydrate(Budget(),row,state)
            candidates=m._learned_candidates(state['budget'])
            if len(candidates)<2:continue
            a=candidates[0]
            b=next((x for x in candidates[1:] if x['action']!=a['action']),candidates[1])
            directory=OUT/output_batch/f'{case}-{idx}'
            aa=branch(row,state,a,'A',directory/'A.json')
            repeat=branch(row,state,a,'A-repeat',directory/'A-repeat.json')
            bb=branch(row,state,b,'B',directory/'B.json')
            delta=bb['final_score']-aa['final_score'];proxy_delta=bb['proxy_final']-aa['proxy_final']
            summary.append({'case':case,'decision':idx,'budget':state['budget'],
                            'repeat_difference':repeat['final_score']-aa['final_score'],
                            'official_delta_B_minus_A':delta,'proxy_delta_B_minus_A':proxy_delta,
                            'delta_error':proxy_delta-delta,'ordering_flip':delta*proxy_delta<0})
            save(OUT/output_batch/'summary.json',summary)
    print(json.dumps(summary))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input-batch',required=True)
    p.add_argument('--output-batch',required=True);p.add_argument('--cases',nargs='+',required=True)
    a=p.parse_args();run_diagnostics(a.input_batch,a.output_batch,a.cases)
