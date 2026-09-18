"""The user's narrowed three-check report: existing data only, no external calls."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import numpy as np
import networkx as nx
from analyze_round5 import paired
from experiment_round5 import OUT
from round4_common import save
from round4_seeds import load_case


def action_key(event):
    args=tuple(event['args'])
    if event['action']=='cut_link':args=tuple(sorted(args))
    return event['action'],args


def topology(case):
    _,seed=load_case(case);g=nx.Graph();g.add_nodes_from(d['id'] for d in seed['nodes']);g.add_edges_from(seed['edges'])
    values=[]
    for comp in nx.connected_components(g):
        ids=sorted(comp)
        if len(ids)<2:continue
        a=nx.to_numpy_array(g,nodelist=ids)+np.eye(len(ids));degree=a.sum(axis=1)
        symmetric=a/np.sqrt(degree[:,None]*degree[None,:])
        eigen=np.linalg.eigvalsh(symmetric)
        values.append(float(max(abs(eigen[:-1]))))
    slem=max(values,default=0.)
    return {'slem':slem,'slow_mixing':slem>=.95,'components':nx.number_connected_components(g),'bridges':len(list(nx.bridges(g)))}


def action_changes(rows):
    index={(r['case'],r['variant']):r for r in rows}
    details=[]
    for case in sorted({r['case'] for r in rows}):
        a,b=index[case,'B0'],index[case,'F0']
        aa=[action_key(e) for e in a['actions'] if e['action']!='scan_node']
        bb=[action_key(e) for e in b['actions'] if e['action']!='scan_node']
        n=max(len(aa),len(bb));changed=sum((aa[i] if i<len(aa) else None)!=(bb[i] if i<len(bb) else None) for i in range(n))
        common=sum((Counter(aa)&Counter(bb)).values())
        scans_a=[action_key(e) for e in a['actions'] if e['action']=='scan_node'];scans_b=[action_key(e) for e in b['actions'] if e['action']=='scan_node']
        details.append({'case':case,'pool':a['spec']['pool'],'family':a['spec']['family'],'intervention_positions':n,'changed_positions':changed,
            'unordered_action_changes':n-common,'order_only':changed>0 and Counter(aa)==Counter(bb),
            'scan_sequence_equal':scans_a==scans_b,'delta':b['final_score']-a['final_score'],
            'score_changed':abs(b['final_score']-a['final_score'])>1e-8,**topology(case)})
    def summarize(rr):
        total=sum(r['intervention_positions'] for r in rr)
        return {'n':len(rr),'changed_trajectory_seeds':sum(r['changed_positions']>0 for r in rr),
            'changed_trajectory_rate':statistics.mean(r['changed_positions']>0 for r in rr),
            'changed_intervention_positions':sum(r['changed_positions'] for r in rr),'intervention_positions':total,
            'intervention_position_change_rate':sum(r['changed_positions'] for r in rr)/total,
            'unordered_change_rate':sum(r['unordered_action_changes'] for r in rr)/total,
            'order_only_seeds':sum(r['order_only'] for r in rr),'different_action_multiset_seeds':sum(r['unordered_action_changes']>0 for r in rr),
            'score_changed_seeds':sum(r['score_changed'] for r in rr),'scan_sequences_all_equal':all(r['scan_sequence_equal'] for r in rr),
            'mean_delta':statistics.mean(r['delta'] for r in rr),'sum_delta':sum(r['delta'] for r in rr),
            'positive_gain_sum':sum(max(0.,r['delta']) for r in rr),'negative_gain_sum':sum(min(0.,r['delta']) for r in rr)}
    groups={'all':summarize(details),'dev_corr':summarize([r for r in details if r['pool']!='select']),
        'select':summarize([r for r in details if r['pool']=='select'])}
    for name,condition in [('slow_mixing',lambda r:r['slow_mixing']),('faster_mixing',lambda r:not r['slow_mixing'])]:
        groups[name]=summarize([r for r in details if condition(r)])
    groups['by_family']={f:summarize([r for r in details if r['family']==f]) for f in sorted({r['family'] for r in details})}
    return {'definition':'Actual non-scan actions aligned by intervention index; unordered change counts ignore order and compare action multisets. These are descriptive metrics, not independent observations.',
        'weakness_rule':'Initial self-loop averaging operator: max nontrivial absolute eigenvalue across components >=0.95; rule fixed before computing group gains. Disconnected components analyzed separately within each graph.',
        'summary':groups,'details':details}


def costs(rows):
    times=[s['seconds'] for r in rows for s in r.get('search_records',[])]
    times.sort()
    plans=[s['plans'] for r in rows for s in r.get('search_records',[])]
    return {'games':len(rows),'mean_game_seconds':statistics.mean(r['elapsed_seconds'] for r in rows),
        'search_calls':len(times),'search_p95_seconds':times[int(.95*(len(times)-1))] if times else None,
        'search_max_seconds':max(times) if times else None,'fallback_calls':sum(not p or not p[0].get('search_completed') for p in plans),
        'truncated_calls':sum(bool(p) and bool(p[0].get('search_truncated')) for p in plans),
        'selected_nonbaseline_calls':sum(bool(p) and p[0].get('search_completed') and not p[0].get('baseline') for p in plans),
        'timing_scope':'first two games sequential, remaining CPU diagnostics use up to four parallel processes; not deployment latency certification'}


def read_compact(path):
    row=json.loads(path.read_text(encoding='utf-8'))
    return {k:row[k] for k in ('case','variant','spec','final_score','elapsed_seconds','actions','search_records') if k in row}


def main(actions_only=False):
    reference=[read_compact(p) for p in (OUT/'P2').glob('*.json')]
    actions=action_changes(reference);save(OUT/'action-change-analysis.json',actions)
    if actions_only:
        print(json.dumps(actions['summary'],ensure_ascii=False,indent=2));return
    diagnostic=[read_compact(p) for p in (OUT/'P3').glob('*.json')]
    combination=[read_compact(p) for p in (OUT/'P3-combination').glob('*.json')]
    assert len(reference)==468 and len(diagnostic)==702 and len(combination)==54
    rows=reference+diagnostic+combination
    result={'action_changes':actions,'response':{},'search':{},'scope':'Existing development seeds and public-receipt-calibrated finite50 simulator. No new sandbox or GLM in narrowed three checks. No final pool evaluated.'}
    for name,condition in [('dev_corr',lambda r:r['spec']['pool']!='select'),('select',lambda r:r['spec']['pool']=='select')]:
        selected=[r for r in rows if condition(r)]
        result['response'][name]=paired(selected,baseline='F0',candidate='D-response')
        result['search'][name]=paired(selected,baseline='F0',candidate='D-search')
    result['search']['combination_select']=paired(rows,baseline='F0',candidate='D-combination')
    result['search']['combination_vs_single_select']=paired(rows,baseline='D-search',candidate='D-combination')
    result['search']['combination_costs']=costs(combination)
    result['search']['greedy_select_costs']=costs([r for r in reference if r['variant']=='F0' and r['spec']['pool']=='select'])
    save(OUT/'three-checks-report.json',result)
    compact={k:({name:{kk:vv for kk,vv in stat.items() if kk not in ('pairs',)} for name,stat in value.items()} if k in ('response','search') else value) for k,value in result.items() if k!='action_changes'}
    print(json.dumps(compact,ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--actions-only',action='store_true');args=p.parse_args();main(args.actions_only)
