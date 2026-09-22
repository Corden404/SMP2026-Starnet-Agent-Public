"""Round24 post-hoc hypothesis lock. Old evidence only; no new validation."""
from __future__ import annotations
import json,statistics
import round24_state as s
ROOT=s.ROOT;OUT=s.OUT/'hypothesis'
def main():
    s.preserved()
    summary=s.read(ROOT/'agent_notes/experiments/round22/causal/summary.json')
    proto=s.read(ROOT/'agent_notes/experiments/round22/causal/protocol.json')
    detail={x['case']:x for x in summary['details']};rows=[]
    for e in proto['events']:
        src=s.read(ROOT/e['source_path'])
        ev=min([x for x in src.get('extra_records',[]) if x.get('kind')=='bundle' and x.get('reason')=='selected'],key=lambda x:x['operation'])
        rows.append(dict(case=e['case'],pool=e['pool'],family=e['family'],actual_delta=detail[e['case']]['delta'],
                         forecast_mean=ev['selected']['mean_delta'],forecast_min=ev['selected']['min_delta'],
                         label=ev['selected']['label'],operation=e['operation'],budget=e['budget']))
    table={}
    for margin in (1.0,2.0,3.0):
        z=[x for x in rows if x['forecast_mean']>=margin];ds=[x['actual_delta'] for x in z]
        table[str(margin)]=dict(n=len(z),mean=statistics.mean(ds),median=statistics.median(ds),
            wins=sum(x>1e-8 for x in ds),ties=sum(abs(x)<=1e-8 for x in ds),losses=sum(x<-1e-8 for x in ds),
            cases=[x['case'] for x in z])
    rec=dict(status='PASS',scope='POST_HOC_HYPOTHESIS_GENERATION_ONLY',events=28,
             candidates={'B2B':1.0,'B2M2':2.0,'B2M3':3.0},threshold_table=table,
             prohibition='No new threshold/family/label rule may be added after fresh Round24 results.')
    s.write(OUT/'locked.json',rec,True)
    s.update('R1','COMPLETE','旧28事件仅用于锁定候选：M1 %.6f(%d/%d/%d)，M2 %.6f(%d/%d/%d)，M3 %.6f(%d/%d/%d)。'%(
        table['1.0']['mean'],table['1.0']['wins'],table['1.0']['ties'],table['1.0']['losses'],
        table['2.0']['mean'],table['2.0']['wins'],table['2.0']['ties'],table['2.0']['losses'],
        table['3.0']['mean'],table['3.0']['wins'],table['3.0']['ties'],table['3.0']['losses']),
        next_action='R2实现PC2/B2B/B2M2/B2M3并运行边界测试。',evidence='hypothesis/locked.json')
    print(json.dumps(rec,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
