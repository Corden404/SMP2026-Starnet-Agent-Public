"""Round21 read-only historical shrink rules over frozen Round20 states."""
from __future__ import annotations
import collections,json,statistics
import round21_state as s
ROOT=s.ROOT;OUT=s.OUT/'historical-rule-audit.json'
def main():
 s.preserved();rows=[]
 for p in sorted((ROOT/'agent_notes/experiments/round20/bundle-diagnostic/states').glob('*.json')):
  r=s.read(p)
  if r.get('status')!='PASS':continue
  rows.append(r)
 if len(rows)!=132:raise RuntimeError('Expected132 Round20 states')
 rules={'B2A':lambda sel:sel and sel['label']!='C2',
        'B2B':lambda sel:sel and sel['label']!='C2' and sel['mean_delta']>=1.0}
 result={}
 for name,fn in rules.items():
  by=collections.defaultdict(list)
  for r in rows:
   sel=r.get('forecast',{}).get('selected');use=bool(fn(sel));real=r.get('realized_delta',0.) if use else 0.
   opp=max(0.,r['old_best_cut_delta'],r.get('realized_delta',0.))
   by[r['case']].append((real,opp-real,use,r['family']))
  gm=[statistics.mean(x[0] for x in v) for v in by.values()]
  rg=[statistics.mean(x[1] for x in v) for v in by.values()]
  pos=[(case,statistics.mean(x[0] for x in v),v[0][3]) for case,v in by.items() if statistics.mean(x[0] for x in v)>1e-8]
  result[name]=dict(graph_realized_mean=statistics.mean(gm),graph_regret=statistics.mean(rg),
    selected=sum(x[2] for v in by.values() for x in v),positive_graphs=len(pos),positive_families=len({x[2] for x in pos}))
 base=s.read(ROOT/'agent_notes/experiments/round20/bundle-diagnostic/summary.json')
 ok=all(x['graph_realized_mean']>base['graph_realized_mean'] and x['graph_regret']<base['graph_regret'] for x in result.values())
 rec=dict(status='PASS',rules=result,round20=dict(mean=base['graph_realized_mean'],regret=base['graph_regret']),eligible_new_dev=ok,
          scope='Post-hoc development shrink rules only; must validate on new Round21 graphs.',new_runs=0,api_requests=0)
 s.write(OUT,rec,True)
 s.update('R1','COMPLETE','历史收缩复核：B2A mean %+.6f/regret %.6f；B2B mean %+.6f/regret %.6f；新开发资格%s。'%(
  result['B2A']['graph_realized_mean'],result['B2A']['graph_regret'],result['B2B']['graph_realized_mean'],result['B2B']['graph_regret'],ok),
  next_action='实现B2A/B2B并运行专项测试、新54图开发。' if ok else '未改善Round20，停止后续实验并收尾。',evidence='historical-rule-audit.json')
 if not ok:
  for k in ('R2','R3','R4','R5'):s.update(k,'SKIPPED_GATE','历史收缩规则未同时改善Round20均值和regret。',evidence='historical-rule-audit.json')
 print(json.dumps(rec,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
