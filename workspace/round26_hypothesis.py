"""Round26 post-hoc hypothesis lock. Old evidence only."""
from __future__ import annotations
import json,pathlib,statistics
import round26_state as s
OUT=s.OUT/'hypothesis'
def rows():
    out=[]
    summ=s.read(s.ROOT/'agent_notes/experiments/round22/causal/summary.json');proto=s.read(s.ROOT/'agent_notes/experiments/round22/causal/protocol.json');det={x['case']:x for x in summ['details']}
    for e in proto['events']:
        row=s.read(s.ROOT/e['source_path']);ev=min([x for x in row.get('extra_records',[]) if x.get('kind')=='bundle' and x.get('reason')=='selected'],key=lambda x:x['operation']);sel=ev['selected']
        out.append(dict(src='r22',case=e['case'],actual=det[e['case']]['delta'],label=sel['label'],inc=ev['incumbent'][0]))
    for rnd,var in [('round24','B2M3'),('round25','B2B')]:
        base=s.ROOT/'agent_notes/experiments'/rnd/'study/runs/dev';src='r24' if rnd=='round24' else 'r25'
        for f in sorted(base.glob('*-'+var+'.json')):
            row=s.read(f);case=row['spec']['case'];pc=s.read(base/(case+'-PC2.json'));d=row['final_score']-pc['final_score']
            for ev in [x for x in row.get('extra_records',[]) if x.get('kind')=='bundle' and x.get('reason')=='selected']:
                sel=ev['selected'];out.append(dict(src=src,case=case,actual=d,label=sel['label'],inc=ev['incumbent'][0]))
    return out
def stat(z):
    ds=[x['actual'] for x in z];ss=sorted(ds)
    return dict(n=len(ds),mean=statistics.mean(ds),median=statistics.median(ds),wins=sum(x>1e-8 for x in ds),ties=sum(abs(x)<=1e-8 for x in ds),losses=sum(x<-1e-8 for x in ds),
        without_best_two=statistics.mean(ss[:-2]) if len(ss)>2 else None,min=min(ds),max=max(ds),cases=[x['case'] for x in z])
def main():
    s.preserved();rs=rows()
    table=dict(
      C1=stat([x for x in rs if x['label']=='C1']),
      C1_inc_shield=stat([x for x in rs if x['label']=='C1' and x['inc']=='shield']),
      C1_inc_comm=stat([x for x in rs if x['label']=='C1' and x['inc']=='comm']),
      C1_C=stat([x for x in rs if x['label']=='C1-C']))
    if table['C1']['n']!=14 or table['C1_inc_shield']['wins']!=6 or table['C1_inc_shield']['losses']!=0:
        raise RuntimeError('Historical hypothesis table drift')
    rec=dict(status='PASS',scope='POST_HOC_HYPOTHESIS_GENERATION_ONLY',candidates={'B2C1':'label==C1','B2CS':'label==C1 and incumbent==shield'},table=table,
      prohibition='No third filter, family/budget exception, or margin change after fresh Round26 data.')
    s.write(OUT/'locked.json',rec,True)
    s.update('R1','COMPLETE','历史假设锁定：C1 14例11/0/3 mean%+.3f wob2%+.3f；C1+shield 6例6/0/0 mean%+.3f wob2%+.3f。'%(
      table['C1']['mean'],table['C1']['without_best_two'],table['C1_inc_shield']['mean'],table['C1_inc_shield']['without_best_two']),
      next_action='R2实现B2C1/B2CS并做边界测试。',evidence='hypothesis/locked.json')
    print(json.dumps(rec,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
