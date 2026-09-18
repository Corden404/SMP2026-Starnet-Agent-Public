"""Separate known-state score, unmeasured response, and predecision Q errors."""
import json
import statistics
from experiment_round5 import OUT
from round4_common import save


def main():
    rows={r['point']:r for r in map(json.loads,(OUT/'predecision-branches.jsonl').read_text(encoding='utf-8').splitlines())}
    progress=json.loads((OUT/'branches-progress.json').read_text(encoding='utf-8'))
    if progress['completed']!=36:raise RuntimeError('Incomplete registered branch batch')
    responses=[]
    for result in progress['results']:
        if result['label']=='A-repeat':continue
        row=rows[result['point']];action=row[result['label']]
        if action['action']!='comm':continue
        i=str(action['target_node_1']);event=result['actions'][0]
        actual=event['measured_first_effect'];mean=row['response_distribution']['means'][i];sd=row['response_distribution']['sd'][i]
        unknown=i not in row['state']['effects']
        responses.append({'point':result['point'],'branch':result['label'],'node':i,'unknown_before_action':unknown,
            'predicted_mean':mean,'actual_first_effect':actual,'error':mean-actual,'heuristic_95_covered':abs(mean-actual)<=1.96*sd+1e-8})
    unknown=[r for r in responses if r['unknown_before_action']]
    q=json.loads((OUT/'Q-calibration.json').read_text(encoding='utf-8'))
    result={'known_terminal_score_max_abs_error':max(abs(r['terminal_score_error']) for r in progress['results']),
        'unmeasured_response_n':len(unknown),'unmeasured_response_MAE':statistics.mean(abs(r['error']) for r in unknown) if unknown else None,
        'heuristic_interval_coverage':statistics.mean(r['heuristic_95_covered'] for r in unknown) if unknown else None,
        'response_records':responses,'Q_mean_abs_error':statistics.mean(abs(p['error']) for p in q['pairs']),
        'Q_confirmation_status':q['status'],'scope':'12 selected diagnostic seeds, not random response-distribution samples; intervals are heuristic, coverage here is descriptive only.'}
    save(OUT/'error-decomposition.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='response_records'},ensure_ascii=False))


if __name__=='__main__':main()
