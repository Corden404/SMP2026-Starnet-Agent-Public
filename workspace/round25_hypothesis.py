"""Round25 historical hypothesis lock. Post-hoc only."""
from __future__ import annotations
import json,pathlib
import round25_state as s
OUT=s.OUT/'hypothesis'
def risk(budget,incumbent,selected):
    if not selected or selected.get('label')!='S2' or abs(float(budget)-20.0)>1e-8:return False
    if not incumbent or incumbent[0]!='comm':return False
    n=len(selected.get('prefix') or [])
    worlds=selected.get('worlds') or []
    return len(worlds)==3 and all(any(a[0]=='comm' for a in w.get('actions',[])[n:]) for w in worlds)
def main():
    s.preserved();hits=[]
    summ=s.read(s.ROOT/'agent_notes/experiments/round22/causal/summary.json');proto=s.read(s.ROOT/'agent_notes/experiments/round22/causal/protocol.json');detail={x['case']:x for x in summ['details']}
    for e in proto['events']:
        row=s.read(s.ROOT/e['source_path']);ev=min([x for x in row.get('extra_records',[]) if x.get('kind')=='bundle' and x.get('reason')=='selected'],key=lambda x:x['operation'])
        if risk(ev['budget'],ev['incumbent'],ev['selected']):hits.append(dict(source='round22',case=e['case'],actual_delta=detail[e['case']]['delta'],operation=ev['operation']))
    base=s.ROOT/'agent_notes/experiments/round24/study/runs/dev'
    for path in sorted(base.glob('*-B2M3.json')):
        row=s.read(path);case=row['spec']['case'];pc=s.read(base/(case+'-PC2.json'));delta=row['final_score']-pc['final_score']
        for ev in [x for x in row.get('extra_records',[]) if x.get('kind')=='bundle' and x.get('reason')=='selected']:
            if risk(ev['budget'],ev['incumbent'],ev['selected']):hits.append(dict(source='round24',case=case,actual_delta=delta,operation=ev['operation']))
    expected={'r21-confirm-066','r24-dev-004','r24-dev-010'}
    if {x['case'] for x in hits}!=expected:raise RuntimeError('Historical risk set drift '+repr(hits))
    rec=dict(status='PASS',scope='POST_HOC_HYPOTHESIS_GENERATION_ONLY',
        rule=dict(label='S2',budget=20.0,incumbent='comm',all_world_suffixes_contain_comm=True),
        hits=hits,wins=sum(x['actual_delta']>1e-8 for x in hits),ties=sum(abs(x['actual_delta'])<=1e-8 for x in hits),losses=sum(x['actual_delta']<-1e-8 for x in hits),
        prohibition='No second veto rule, family exception, degree rule, or margin change after fresh Round25 data.')
    s.write(OUT/'locked.json',rec,True)
    s.update('R1','COMPLETE','历史risk集合锁定：3例，0胜/1平/2负；仅用于生成B2R，不作为验证。',
             next_action='R2实现B2R并做边界测试。',evidence='hypothesis/locked.json')
    print(json.dumps(rec,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
