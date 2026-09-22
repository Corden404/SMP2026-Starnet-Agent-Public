"""Read-only historical analysis for a REVIEW plan, not a new policy experiment.

Reads every Round18 dev pair; replays forecasts for ALL four Cut1 replacements.
Writes only a new immutable round19-review evidence file. Never invokes a new
whole-game strategy, an API, model training, an upload, or an old study writer.
The full-S44 forecast remains an approximation: it is NOT recursive full PC2.
"""
from __future__ import annotations
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
sys.dont_write_bytecode = True
import round18_common as h

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'agent_notes/experiments/round18'
DEST = ROOT / 'agent_notes/experiments/round19-review/historical-audit.json'
SOURCE_HASHES = {}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    SOURCE_HASHES[str(Path(path).relative_to(ROOT))] = digest(path)
    return json.loads(Path(path).read_text(encoding='utf8'))


def snapshot(row, event):
    op = event['operation']
    return dict(operation=op, budget=event['budget'],
                calls=sum(c['operation'] < op for c in row['chains']),
                pc2_seen=sorted({tag for r in row['candidate_records']
                                if r['operation'] <= op for tag in r['tags']}))


def fixed_path_value(module, initial, actions, real_receipts):
    # Future actions and receipts are used ONLY after the experiment, for
    # arithmetic decomposition; they are NOT inputs to an admissible live policy.
    nodes, edges, effects, observations = module.BudgetRollout.clone(initial)
    means = {i: module.BudgetRollout.effect(i,d,effects,observations)[0]
             for i,d in nodes.items()}
    for a in actions:
        u = a['args'][0]
        if a['action'] == 'communicate':
            if real_receipts:
                nodes[u]['w'] = a['result']['new_w']
            else:
                nodes[u]['w'] += means[u] * .5 ** (3 - nodes[u]['comm_left'])
            nodes[u]['comm_left'] -= 1
        elif a['action'] == 'shield_node':
            del nodes[u]
            edges = {e for e in edges if u not in e}
        elif a['action'] == 'cut_link':
            edges.remove(tuple(sorted(a['args'])))
        else:
            raise ValueError('Unexpected suffix action')
    return module.pc2_score(nodes, edges)[0]


def action(key):
    kind,u,v = key
    return dict(action=kind,target_node_1=u,target_node_2=v,
                prompt_id=1 if kind=='comm' else None,
                cost={'comm':2.,'cut':3.,'shield':5.}[kind],gain=0.,roi=0.)


def main():
    if DEST.exists():
        raise RuntimeError('Existing review evidence is immutable; inspect rather than overwrite')
    for folder in ('workspace','tests','docs'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.md','.json','.txt','.ps1') and '__pycache__' not in path.parts:
                SOURCE_HASHES[str(path.relative_to(ROOT))] = digest(path)
    for variant in ('PC2','Cut1','Fast'):
        assert (OLD/'packages'/variant/'team_submission/starnet_model.py').is_file()
    summary = read(OLD/'study/summaries/dev.json')
    read(OLD/'closeout/audit.json')
    module = h.load('Cut1','review_audit')
    events = []
    reasons = collections.Counter()
    fast_new_completed = collections.Counter()
    fast_sequences_same = fast_bags_same = 0
    source_pairs = []
    for path in sorted((OLD/'study/runs/dev').glob('*-Cut1.json')):
        candidate = read(path)
        base = read(path.with_name(path.name.replace('-Cut1','-PC2')))
        fast = read(path.with_name(path.name.replace('-Cut1','-Fast')))
        assert all(r['status']=='PASS' for r in (candidate,base,fast))
        assert h.scan_prefix(candidate)==h.scan_prefix(base)==h.scan_prefix(fast)
        source_pairs.append(candidate['spec']['case'])
        fast_sequences_same += int([(a['action'],a['args']) for a in base['actions']]==[(a['action'],a['args']) for a in fast['actions']])
        fast_bags_same += int(h.bag(base)==h.bag(fast))
        for b,f in zip(base['candidate_records'],fast['candidate_records']):
            assert b['tags']==f['tags']
            if b.get('reason')=='incomplete_keep_original' and f.get('completed_worlds')==3:
                fast_new_completed[f['reason']] += 1
        for ev in candidate['extra_records']:
            reasons[ev['reason']] += 1
            if not ev.get('replaced'):
                continue
            op=ev['operation']
            with h.c.offline():
                _,raw,logged,model=h.restore(candidate,snapshot(candidate,ev),'review_restore')
            state=model.completed_state()
            slots=min(model.operation_limit-1-model.operation_count,
                      model.llm_counter.safe_limit-model.llm_counter.call_count)
            result=dict(case=candidate['spec']['case'],family=candidate['spec']['family'],
                        budget=ev['budget'],operation=op,original_event=ev,
                        actual_delta=candidate['final_score']-base['final_score'],fixed_actual_paths={})
            for name,row in (('PC2',base),('Cut1',candidate)):
                acts=row['actions'][op:]
                result['fixed_actual_paths'][name]=dict(actions=acts,
                    known_response_at_window=[dict(node=a['args'][0],known=a['args'][0] in state[2])
                        for a in acts if a['action']=='communicate'],
                    mean_value=fixed_path_value(module,state,acts,False),
                    receipt_value=fixed_path_value(module,state,acts,True))
            fixed=result['fixed_actual_paths']
            result['fixed_path_mean_delta']=fixed['Cut1']['mean_value']-fixed['PC2']['mean_value']
            result['fixed_path_receipt_delta']=fixed['Cut1']['receipt_value']-fixed['PC2']['receipt_value']
            result['actual_same_action_bag']=h.bag(base)==h.bag(candidate)
            worlds=module.pc2_worlds(state)
            for name,depth in (('original_depth2',2),('full_s44_numeric_suffix',100)):
                rows=[]
                for index,world in enumerate(worlds):
                    branches={}
                    for which,key in (('incumbent',ev['incumbent']),('challenger',ev['challenger'])):
                        module.r18_clear_private()
                        branches[which]=module.r18_cut_continue(state,ev['budget'],slots,
                            set(model.local_nodes),action(key),world,time.monotonic()+10.,depth)
                    rows.append(dict(delta=branches['challenger']['score']-branches['incumbent']['score'],branches=branches))
                result[name]=dict(mean=sum(r['delta'] for r in rows)/3,worlds=rows)
            if abs(result['original_depth2']['mean']-ev['predicted_delta'])>1e-8:
                raise RuntimeError('Original forecast did not replay')
            events.append(result)
    assert len(source_pairs)==54 and len(events)==4
    for name,old_hash in SOURCE_HASHES.items():
        if digest(ROOT/name)!=old_hash:
            raise RuntimeError('An existing inspected file changed: '+name)
    result=dict(status='PASS',purpose='Historical evidence for user review; NOT candidate validation',
        baseline_archive='dist/team_submission-round17-pc2-exploratory-20260921.zip',
        baseline_sha256=digest(ROOT/'dist/team_submission-round17-pc2-exploratory-20260921.zip'),
        graphs_read=54,old_whole_games_read=162,cut1_comparison_reasons=dict(reasons),
        complete_replacement_cohort=len(events),forecast_suffixes=48,
        fast_newly_completed=dict(fast_new_completed),fast_sequences_same=fast_sequences_same,
        fast_action_bags_same=fast_bags_same,
        full_game_runs=0,new_independent_graphs=0,api_requests=0,training_runs=0,
        events=events,source_hashes=SOURCE_HASHES,
        interpretation='Recorded forecasts replayed from public belief only. Full-S44 suffix is not exact PC2 and not optimal. Fixed actual-path analyses use future actions/receipts post hoc, cannot be deployed or credited as prospective score gain. All four replacements included; untriggered opportunities require later complete diagnosis.')
    DEST.parent.mkdir(parents=True,exist_ok=True)
    with DEST.open('x',encoding='utf8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('events','source_hashes')},ensure_ascii=False))
    for event in events:
        print(json.dumps(dict(case=event['case'],old_prediction=event['original_depth2']['mean'],
            full_s44_prediction=event['full_s44_numeric_suffix']['mean'],actual_delta=event['actual_delta'],
            fixed_mean=event['fixed_path_mean_delta'],fixed_receipts=event['fixed_path_receipt_delta']),ensure_ascii=False))


if __name__=='__main__':
    main()
