"""Descriptive action, search-cost and interaction diagnostics, separate from selection."""
from collections import defaultdict
import json
import math
import statistics
from experiment_round5 import OUT
from round4_common import save


def main():
    rows=[json.loads(p.read_text(encoding='utf-8')) for folder in ('P2','P3') for p in (OUT/folder).glob('*.json')]
    groups=defaultdict(list)
    for row in rows:groups[(row['spec']['pool']=='select',row['variant'])].append(row)
    summary={}
    for (select,variant),records in groups.items():
        first=[];repeat=[];structure=[];timings=[];fallback=0;searches=0;changed=0
        for r in records:
            counts=defaultdict(int);struct=0
            for a in r['actions']:
                if a['action']=='communicate':counts[a['args'][0]]+=1
                if a['action'] in ('cut_link','shield_node'):struct+=1
            first.append(len(counts));repeat.append(sum(max(0,n-1) for n in counts.values()));structure.append(struct)
            for search in r.get('search_records',[]):
                searches+=1;timings.append(search['seconds']);plans=search['plans']
                if not plans or not plans[0].get('search_completed',False):fallback+=1
                elif not plans[0].get('baseline'):changed+=1
        timings.sort()
        summary[('select' if select else 'dev_corr')+'-'+variant]={'n':len(records),'mean_first_communications':statistics.mean(first),
            'mean_repeat_communications':statistics.mean(repeat),'mean_structure_actions':statistics.mean(structure),
            'search_calls':searches,'search_fallbacks':fallback,'nonbaseline_choices':changed,
            'search_seconds_p95':timings[min(len(timings)-1,int(.95*len(timings)))] if timings else None,
            'search_seconds_max':max(timings) if timings else None}
    index={(r['case'],r['variant']):r['final_score'] for r in rows}
    interactions={}
    for pool in ('dev_corr','select'):
        cases=[c for c,v in index if v=='F0' and (c.startswith('select-'))==(pool=='select')]
        interactions[pool]=statistics.mean(index[c,'D-both']-index[c,'D-response']-index[c,'D-search']+index[c,'F0'] for c in cases)
    save(OUT/'loss-details.json',{'actions_and_cost':summary,'response_search_interaction':interactions,
        'scope':'Oracle variants diagnose response gaps, not optimal upper bounds or submit-eligible policies. Search diagnostic tests only one additional bounded first-action prefix at a time.'})
    print(json.dumps({'interaction':interactions},ensure_ascii=False))


if __name__=='__main__':main()
