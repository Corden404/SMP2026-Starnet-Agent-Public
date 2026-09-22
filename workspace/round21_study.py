"""Round21 frozen fresh development/confirmation study."""
from __future__ import annotations
import argparse,collections,concurrent.futures,json,statistics
from pathlib import Path
import round21_state as s
import round21_common as h
import round19_common as r19old
from round9_lock import ExclusiveBatch
OUT=s.OUT/'study';DEPS=('workspace/round21_study.py','tests/test_round21_study.py','tests/test_round21_policy.py')
ALL=('PC2','B2A','B2B')
POOLS=('dev','integration','confirm','stress')
def code():return h.fingerprint(DEPS)
def prepare():
 s.preserved()
 if not s.read(s.OUT/'historical-rule-audit.json')['eligible_new_dev']:raise RuntimeError('Historical candidate gate failed')
 seeds={p:h.entries(p) for p in POOLS}
 fresh=[e['sha256'] for rows in seeds.values() for e in rows]
 old=set()
 for pool in ('dev','integration','confirm','stress'):
  old.update(e['sha256'] for e in h.r18.entries(pool))
  old.update(e['sha256'] for e in r19old.entries(pool))
 if len(set(fresh))!=len(fresh) or set(fresh)&old:raise RuntimeError('Fresh cohort collision')
 p=dict(code=code(),variants=list(ALL),seeds=seeds,workers=2,
  development_gate='54 complete graphs; mean>0; >=1 win; scan drift0; nontriggered score/bag drift0; unexplained changed games0; ops/calls<120.',
  selection='Highest dev mean among eligible; tie<=1e-8 prefers B2A. Frozen thereafter.',
  integration='6 graphs, all3 variants; safety only, not score selection.',
  local_gate='selected candidate: confirm mean>0, stress mean>=0, safety pass. CI lower bounds only confirmed-local label.',
  no_upload=True,no_api_before_local=True)
 for v in ALL:h.load(v,'prepare')
 s.write(OUT/'protocol.json',p,True)
 s.update('R2','COMPLETE','B2A/B2B源码与包已冻结；专项测试通过；新54图/6图集成身份无历史碰撞。',
  next_action='运行R3新54图开发：PC2/B2A/B2B共162局。',evidence='study/protocol.json')
 s.update('R3','RUNNING','新54图开发协议已冻结；不按中途分数停跑。',evidence='study/protocol.json')
 print(json.dumps(dict(prepared=True,counts={k:len(v) for k,v in seeds.items()},variants=ALL),ensure_ascii=False),flush=True)
def check():
 p=s.read(OUT/'protocol.json')
 if p['code']!=code():raise RuntimeError('Frozen Round21 study source drift')
 return p
def variants(pool):
 p=check()
 if pool in ('dev','integration'):return list(p['variants'])
 d=s.read(OUT/'development-decision.json')
 if not d.get('selected'):raise RuntimeError('No selected candidate')
 return ['PC2',d['selected']]
def run_one(pool,index,variant):
 p=check();entry=p['seeds'][pool][index];spec=entry['spec']
 if s.digest(h.c.generate(spec))!=entry['sha256']:raise RuntimeError('Seed drift')
 mode='integration' if pool=='integration' else 'fast';ident=dict(entry=entry,variant=variant,code=p['code'],mode=mode)
 path=OUT/'runs'/pool/(spec['case']+'-'+variant+'.json')
 return h.checked(path,ident,lambda:h.run_game(spec,variant,mode))
def group(pool,index):
 vs=variants(pool);shift=index%len(vs);order=vs[shift:]+vs[:shift]
 if (index//len(vs))%2:order=order[::-1]
 rs=[run_one(pool,index,v) for v in order]
 return dict(case=rs[0]['spec']['case'],games=len(rs),seconds=sum(x['elapsed_seconds'] for x in rs))
def stats(pool,variant):
 p=check();pairs=[];prefix=nontriggered=unexplained=0;maxops=maxcalls=0;maxerr=0.;triggers=0;changedbags=0
 for e in p['seeds'][pool]:
  case=e['spec']['case'];a=s.read(OUT/'runs'/pool/(case+'-PC2.json'));b=s.read(OUT/'runs'/pool/(case+'-'+variant+'.json'))
  for r,v in ((a,'PC2'),(b,variant)):
   if r.get('status')!='PASS' or r['identity']['entry']!=e or r['identity']['code']!=p['code']:raise RuntimeError('Mixed cohort')
   maxerr=max(maxerr,h.c.audit_game(r));maxops=max(maxops,r['operations']);maxcalls=max(maxcalls,r['logical_calls'])
  delta=b['final_score']-a['final_score'];samebag=h.bag(a)==h.bag(b);sameprefix=h.scan_prefix(a)==h.scan_prefix(b)
  trig=any(x.get('kind')=='bundle' and x.get('reason')=='selected' for x in b.get('extra_records',[]))
  prefix+=int(not sameprefix);nontriggered+=int(not trig and (abs(delta)>1e-8 or not samebag))
  unexplained+=int((abs(delta)>1e-8 or not samebag) and not trig)
  triggers+=int(trig);changedbags+=int(not samebag)
  pairs.append(dict(case=case,family=e['spec']['family'],delta=delta,triggered=trig,same_action_bag=samebag))
 st=h.c.paired_stats(pairs);st.update(pairs=pairs,prefix_drift=prefix,nontriggered_different=nontriggered,
  unexplained_changed=unexplained,max_operations=maxops,max_calls=maxcalls,max_forward_error=maxerr,triggers=triggers,changed_bags=changedbags)
 return st
def safe(x):return x['prefix_drift']==0 and x['nontriggered_different']==0 and x['unexplained_changed']==0 and x['max_operations']<120 and x['max_calls']<120 and x['max_forward_error']<=1e-8
def summarize(pool):
 ss={v:stats(pool,v) for v in variants(pool) if v!='PC2'};rec=dict(status='PASS',pool=pool,graphs=len(check()['seeds'][pool]),comparisons=ss);s.write(OUT/'summaries'/(pool+'.json'),rec,True)
 if pool=='dev':
  elig=[v for v,x in ss.items() if x['mean']>0 and x['wins']>=1 and safe(x)]
  elig.sort(key=lambda v:(-ss[v]['mean'],0 if v=='B2A' else 1))
  selected=elig[0] if elig else None
  d=dict(selected=selected,eligible=elig,code=check()['code'],summary_sha=s.sha(OUT/'summaries/dev.json'),
   candidates={v:dict(mean=x['mean'],wins=x['wins'],ties=x['ties'],losses=x['losses'],safe=safe(x),triggers=x['triggers'],changed_bags=x['changed_bags']) for v,x in ss.items()})
  s.write(OUT/'development-decision.json',d,True)
  s.update('R3','RUNNING','54图开发完成：'+'；'.join('%s %+.6f (%d/%d/%d)'%(v,x['mean'],x['wins'],x['ties'],x['losses']) for v,x in ss.items())+'；暂选 '+str(selected)+'；继续固定6图集成。',
   next_action='运行6图CaseVO/FakeLLM集成；安全通过后决定是否进入确认。',evidence='study/development-decision.json')
 elif pool=='integration':
  d=s.read(OUT/'development-decision.json');selected=d['selected']
  ok=selected is not None and safe(ss[selected])
  d=dict(d,integration_safe=ok,integration_sha=s.sha(OUT/'summaries/integration.json'));s.write(OUT/'development-decision.json',d)
  s.update('R3','COMPLETE','6图集成完成；入选 '+str(selected)+'；integration安全='+str(ok)+'.',
   next_action='进入216确认/54压力。' if ok else '开发/集成未通过，跳过确认与真实层并收尾。',evidence='study/summaries/integration.json')
  if not ok:
   for k in ('R4','R5'):s.update(k,'SKIPPED_GATE','Round21开发/集成未通过冻结门槛。',evidence='study/development-decision.json')
 elif pool in ('confirm','stress'):
  cpath=OUT/'summaries/confirm.json';spath=OUT/'summaries/stress.json'
  if cpath.exists() and spath.exists():
   d=s.read(OUT/'development-decision.json');v=d['selected'];co=s.read(cpath)['comparisons'][v];st=s.read(spath)['comparisons'][v]
   eligible=co['mean']>0 and st['mean']>=0 and safe(co) and safe(st)
   loc=dict(candidate=v,eligible_real=eligible,confirmed_local=eligible and co['ci95_t'][0]>0 and co['ci95_family_bootstrap'][0]>0,
            confirm_mean=co['mean'],stress_mean=st['mean'],code=check()['code'])
   s.write(OUT/'local-decision.json',loc,True)
   s.update('R4','COMPLETE','216确认/54压力完成；均差%+.6f/%+.6f；真实层资格%s。'%(co['mean'],st['mean'],eligible),
    next_action='准备真实LLM层。' if eligible else '跳过真实LLM并收尾。',evidence='study/local-decision.json')
   if not eligible:s.update('R5','SKIPPED_GATE','确认/压力未满足冻结门槛。',evidence='study/local-decision.json')
 print(json.dumps(dict(pool=pool,comparisons={v:{k:x[k] for k in ('mean','wins','ties','losses','ci95_t','triggers','changed_bags','prefix_drift','nontriggered_different','unexplained_changed')} for v,x in ss.items()}),ensure_ascii=False),flush=True);return rec
def run(pool,start,stop):
 p=check();vs=variants(pool);n=len(p['seeds'][pool])
 if not 0<=start<stop<=n:raise ValueError('Invalid block')
 if pool in ('confirm','stress'):
  d=s.read(OUT/'development-decision.json')
  if not d.get('integration_safe'):raise RuntimeError('Integration gate not passed')
 step='R3' if pool in ('dev','integration') else 'R4'
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
  pending={};cur=start
  while cur<min(stop,start+2):pending[ex.submit(group,pool,cur)]=cur;cur+=1
  while pending:
   done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in done:
    pending.pop(f);info=f.result()
    complete=sum(all((OUT/'runs'/pool/(e['spec']['case']+'-'+v+'.json')).exists() and s.read(OUT/'runs'/pool/(e['spec']['case']+'-'+v+'.json')).get('status')=='PASS' for v in vs) for e in p['seeds'][pool])
    if complete%5==0 or complete==n:s.update(step,'RUNNING','%s已持久化%d/%d完整图；最近%s。'%(pool,complete,n,info['case']),next_action='继续冻结队列；PASS复用，失败先检查。',evidence='study/runs/'+pool)
    print(json.dumps(dict(pool=pool,completed=complete,**info)),flush=True)
    if cur<stop:pending[ex.submit(group,pool,cur)]=cur;cur+=1
 if all((OUT/'runs'/pool/(e['spec']['case']+'-'+v+'.json')).exists() for e in p['seeds'][pool] for v in vs):return summarize(pool)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary']);ap.add_argument('--pool',choices=POOLS,default='dev');ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int);a=ap.parse_args()
 with ExclusiveBatch(s.OUT/'execution.lock'):
  if a.stage=='prepare':prepare()
  elif a.stage=='summary':summarize(a.pool)
  else:run(a.pool,a.start,a.stop if a.stop is not None else len(check()['seeds'][a.pool]))
if __name__=='__main__':main()
