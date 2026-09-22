"""Round24 score-first fresh dev/integration/confirm/stress study."""
from __future__ import annotations
import argparse,collections,concurrent.futures,copy,json,statistics
from pathlib import Path
import round24_state as s
import round24_common as h
from round9_lock import ExclusiveBatch
OUT=s.OUT/'study'
DEV_VARIANTS=('PC2','B2B','B2M2','B2M3')
POOLS=('dev','integration','confirm','stress')
DEPS=('workspace/round24_study.py','tests/test_round24_study.py','workspace/round24_common.py','tests/test_round24_core.py')

def code():return h.fingerprint(DEPS)

def specs(pool):
    if pool=='dev':
        rows=[dict(family=f,scale=sc,persona_mix=pm,rep=240,mode='independent',n=50,ood=False,generator_version=1)
              for f in h.FAMILIES for sc in h.SCALES for pm in range(3)]
    elif pool=='integration':
        rows=[dict(family=f,scale=h.SCALES[i%3],persona_mix=(i+1)%3,rep=241,mode='independent',n=50,ood=False,generator_version=1)
              for i,f in enumerate(h.FAMILIES)]
    elif pool=='confirm':
        rows=[dict(family=f,scale=sc,persona_mix=pm,rep=rep,mode='independent',n=50,ood=False,generator_version=1)
              for rep in range(250,254) for f in h.FAMILIES for sc in h.SCALES for pm in range(3)]
    elif pool=='stress':
        rows=copy.deepcopy(h.c.stress_design.specs())
        for x in rows:x['rep']=260
    else:raise ValueError(pool)
    for i,x in enumerate(rows):x.update(case='r24-%s-%03d'%(pool,i),pool='r24-'+pool,round24=1)
    return rows

def entries(pool):return [dict(spec=x,sha256=s.digest(h.c.generate(x))) for x in specs(pool)]

def known_graph_hashes():
    hashes=set();seen=set()
    base=s.ROOT/'agent_notes/experiments'
    def walk(x):
        if isinstance(x,dict):
            sp=x.get('spec')
            if isinstance(sp,dict) and all(k in sp for k in ('family','scale','persona_mix','n')):
                key=s.digest(sp)
                if key not in seen:
                    seen.add(key)
                    try:hashes.add(s.digest(h.c.generate(sp)))
                    except Exception:pass
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    for path in base.rglob('*.json'):
        if 'round24' in path.parts:continue
        try:walk(s.read(path))
        except (ValueError,OSError):pass
    return hashes

def prepare():
    s.preserved()
    hyp=s.read(s.OUT/'hypothesis/locked.json')
    if hyp['candidates']!={'B2B':1.0,'B2M2':2.0,'B2M3':3.0}:raise RuntimeError('Hypothesis lock drift')
    seeds={p:entries(p) for p in POOLS};fresh=[e['sha256'] for v in seeds.values() for e in v]
    old=known_graph_hashes()
    if len(set(fresh))!=len(fresh):raise RuntimeError('Round24 cohort internal collision')
    overlap=set(fresh)&old
    if overlap:raise RuntimeError('Round24 cohort overlaps prior graphs: '+repr(list(overlap)[:3]))
    for v in DEV_VARIANTS:h.load(v,'prepare')
    protocol=dict(code=code(),seeds=seeds,dev_variants=list(DEV_VARIANTS),workers=2,
      dev_gate='Attributable mean>0; without-best-two>=0; >=3 nonzero; wins>losses; scan/audit/resource pass. Nontriggered drift delta is zeroed for attribution and separately reported.',
      selection='Highest attributable fresh-dev mean among eligible; tie<=1e-8: B2M3 > B2M2 > B2B.',
      integration='PC2 + selected on fresh6; safety only.',
      confirm_gate='raw confirm mean>0, without-best-two>=0, wins>losses; stress mean>=0 and without-best-two>=0; integrity pass.',
      no_speed_gate=True,no_api_before_real=True,real_model_order=list(s.REAL_MODEL_ORDER))
    s.write(OUT/'protocol.json',protocol,True)
    s.update('R2','COMPLETE','PC2/B2B/B2M2/B2M3包已冻结；核心边界测试通过；四个新 cohort 与旧图无hash重叠。',
             next_action='R3运行fresh dev54四策略共216局。',evidence='study/protocol.json')
    s.update('R3','RUNNING','fresh dev54协议已冻结；目标只看最终分数/完整性，不看速度。',evidence='study/protocol.json')
    print(json.dumps(dict(prepared=True,counts={k:len(v) for k,v in seeds.items()},old_hashes=len(old)),ensure_ascii=False),flush=True)

def check():
    p=s.read(OUT/'protocol.json')
    if p['code']!=code():raise RuntimeError('Round24 study source drift')
    return p

def variants(pool):
    p=check()
    if pool=='dev':return list(p['dev_variants'])
    d=s.read(OUT/'development-decision.json')
    if not d.get('selected'):raise RuntimeError('No selected candidate')
    return ['PC2',d['selected']]

def run_one(pool,index,variant):
    p=check();entry=p['seeds'][pool][index];spec=entry['spec']
    if s.digest(h.c.generate(spec))!=entry['sha256']:raise RuntimeError('Seed drift')
    mode='integration' if pool=='integration' else 'fast'
    ident=dict(entry=entry,variant=variant,code=p['code'],mode=mode)
    path=OUT/'runs'/pool/(spec['case']+'-'+variant+'.json')
    return h.checked(path,ident,lambda:h.run_game(spec,variant,mode))

def group(pool,index):
    vs=variants(pool);shift=index%len(vs);order=vs[shift:]+vs[:shift]
    if (index//len(vs))%2:order=order[::-1]
    rs=[run_one(pool,index,v) for v in order]
    return dict(case=rs[0]['spec']['case'],games=len(rs))

def triggered(row):
    return any(x.get('kind')=='bundle' and x.get('reason')=='selected' for x in row.get('extra_records',[]))

def compare(pool,variant):
    p=check();rawpairs=[];attrpairs=[];prefix=0;maxerr=0.;maxops=maxcalls=0;drifts=[];triggers=changed=0
    for e in p['seeds'][pool]:
        case=e['spec']['case'];a=s.read(OUT/'runs'/pool/(case+'-PC2.json'));b=s.read(OUT/'runs'/pool/(case+'-'+variant+'.json'))
        for r in (a,b):
            if r.get('status')!='PASS' or r['identity']['entry']!=e or r['identity']['code']!=p['code']:raise RuntimeError('Mixed cohort')
            maxerr=max(maxerr,h.c.audit_game(r));maxops=max(maxops,r['operations']);maxcalls=max(maxcalls,r['logical_calls'])
        delta=b['final_score']-a['final_score'];trig=triggered(b);samebag=h.bag(a)==h.bag(b);sameprefix=h.scan_prefix(a)==h.scan_prefix(b)
        prefix+=int(not sameprefix);triggers+=int(trig);changed+=int(not samebag or abs(delta)>1e-8)
        unexplained=(not trig) and (not samebag or abs(delta)>1e-8)
        if unexplained:drifts.append(dict(case=case,delta=delta,same_action_bag=samebag))
        rawpairs.append(dict(case=case,family=e['spec']['family'],delta=delta))
        attrpairs.append(dict(case=case,family=e['spec']['family'],delta=delta if not unexplained else 0.0))
    raw=h.c.paired_stats(rawpairs);attr=h.c.paired_stats(attrpairs)
    return dict(raw=raw,attributable=attr,prefix_drift=prefix,unexplained_nontriggered=drifts,
                triggers=triggers,changed_games=changed,max_operations=maxops,max_calls=maxcalls,max_forward_error=maxerr)

def integrity(x):
    return x['prefix_drift']==0 and x['max_operations']<120 and x['max_calls']<120 and x['max_forward_error']<=1e-8

def eligible_dev(x):
    a=x['attributable']
    return a['mean']>0 and a['without_best_two']>=0 and (a['wins']+a['losses'])>=3 and a['wins']>a['losses'] and integrity(x)

def summarize(pool):
    vs=[v for v in variants(pool) if v!='PC2'];comp={v:compare(pool,v) for v in vs}
    rec=dict(status='PASS',pool=pool,graphs=len(check()['seeds'][pool]),comparisons=comp);s.write(OUT/'summaries'/(pool+'.json'),rec,True)
    if pool=='dev':
        elig=[v for v,x in comp.items() if eligible_dev(x)]
        priority={'B2M3':0,'B2M2':1,'B2B':2}
        elig.sort(key=lambda v:(-comp[v]['attributable']['mean'],priority[v]))
        selected=elig[0] if elig else None
        d=dict(selected=selected,eligible=elig,code=check()['code'],summary_sha=s.sha(OUT/'summaries/dev.json'),
               candidates={v:dict(raw_mean=x['raw']['mean'],attributable_mean=x['attributable']['mean'],
                                  wins=x['attributable']['wins'],ties=x['attributable']['ties'],losses=x['attributable']['losses'],
                                  without_best_two=x['attributable']['without_best_two'],triggers=x['triggers'],
                                  drift_cases=len(x['unexplained_nontriggered']),eligible=eligible_dev(x)) for v,x in comp.items()})
        s.write(OUT/'development-decision.json',d,True)
        s.update('R3','RUNNING','fresh dev54完成：'+'；'.join('%s raw%+.4f attr%+.4f %d/%d/%d'%(v,x['raw']['mean'],x['attributable']['mean'],x['attributable']['wins'],x['attributable']['ties'],x['attributable']['losses']) for v,x in comp.items())+'；选择 '+str(selected),
                 next_action='运行fresh integration6。' if selected else '无候选过分数门槛，跳过R4/R5并收尾。',evidence='study/development-decision.json')
        if not selected:
            s.update('R3','COMPLETE','fresh dev54无候选通过冻结分数门槛。',evidence='study/development-decision.json')
            for k in ('R4','R5'):s.update(k,'SKIPPED_GATE','R3无候选。',evidence='study/development-decision.json')
    elif pool=='integration':
        d=s.read(OUT/'development-decision.json');v=d['selected'];ok=integrity(comp[v])
        d=dict(d,integration_safe=ok,integration_sha=s.sha(OUT/'summaries/integration.json'));s.write(OUT/'development-decision.json',d)
        s.update('R3','COMPLETE','fresh integration6：候选%s，安全=%s，raw mean=%+.6f。'%(v,ok,comp[v]['raw']['mean']),
                 next_action='进入R4 confirm216+stress54。' if ok else '集成失败，跳过R4/R5。',evidence='study/summaries/integration.json')
        if not ok:
            for k in ('R4','R5'):s.update(k,'SKIPPED_GATE','R3集成安全失败。',evidence='study/summaries/integration.json')
    elif pool in ('confirm','stress'):
        cp=OUT/'summaries/confirm.json';sp=OUT/'summaries/stress.json'
        if cp.exists() and sp.exists():
            d=s.read(OUT/'development-decision.json');v=d['selected']
            co=s.read(cp)['comparisons'][v];st=s.read(sp)['comparisons'][v]
            eligible=(co['raw']['mean']>0 and co['raw']['without_best_two']>=0 and co['raw']['wins']>co['raw']['losses'] and
                      st['raw']['mean']>=0 and st['raw']['without_best_two']>=0 and integrity(co) and integrity(st) and
                      not co['unexplained_nontriggered'] and not st['unexplained_nontriggered'])
            decision=dict(candidate=v,eligible_real=eligible,
                confirmed_local=eligible and co['raw']['ci95_t'][0]>0 and co['raw']['ci95_family_bootstrap'][0]>0,
                confirm_mean=co['raw']['mean'],stress_mean=st['raw']['mean'],code=check()['code'])
            s.write(OUT/'local-decision.json',decision,True)
            s.update('R4','COMPLETE','confirm/stress完成：%s mean %+.6f/%+.6f；真实层资格%s。'%(v,co['raw']['mean'],st['raw']['mean'],eligible),
                     next_action='进入R5真实LLM。' if eligible else '未过独立分数门槛，跳过R5并收尾。',evidence='study/local-decision.json')
            if not eligible:s.update('R5','SKIPPED_GATE','R4独立分数/完整性门槛未通过。',evidence='study/local-decision.json')
    print(json.dumps(dict(pool=pool,comparisons={v:dict(raw_mean=x['raw']['mean'],attr_mean=x['attributable']['mean'],
        wtl=[x['attributable']['wins'],x['attributable']['ties'],x['attributable']['losses']],triggers=x['triggers'],
        drift=len(x['unexplained_nontriggered'])) for v,x in comp.items()}),ensure_ascii=False),flush=True);return rec

def run(pool,start,stop):
    p=check();vs=variants(pool);n=len(p['seeds'][pool])
    if not 0<=start<stop<=n:raise ValueError('Invalid block')
    if pool in ('confirm','stress'):
        d=s.read(OUT/'development-decision.json')
        if not d.get('integration_safe'):raise RuntimeError('Integration gate not passed')
        s.update('R4','RUNNING','%s固定队列开始。'%pool,next_action='跑完整固定队列；不按中途分数停止。',evidence='study/runs/'+pool)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        pending={};cur=start
        while cur<min(stop,start+2):pending[ex.submit(group,pool,cur)]=cur;cur+=1
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                pending.pop(f);info=f.result()
                complete=sum(all((OUT/'runs'/pool/(e['spec']['case']+'-'+v+'.json')).exists() and s.read(OUT/'runs'/pool/(e['spec']['case']+'-'+v+'.json')).get('status')=='PASS' for v in vs) for e in p['seeds'][pool])
                step='R3' if pool in ('dev','integration') else 'R4'
                if complete%5==0 or complete==n:s.update(step,'RUNNING','%s已持久化%d/%d完整图；最近%s。'%(pool,complete,n,info['case']),next_action='继续固定队列。',evidence='study/runs/'+pool)
                print(json.dumps(dict(pool=pool,completed=complete,**info),ensure_ascii=False),flush=True)
                if cur<stop:pending[ex.submit(group,pool,cur)]=cur;cur+=1
    if all((OUT/'runs'/pool/(e['spec']['case']+'-'+v+'.json')).exists() for e in p['seeds'][pool] for v in vs):return summarize(pool)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary']);ap.add_argument('--pool',choices=POOLS,default='dev');ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int);a=ap.parse_args()
    with ExclusiveBatch(s.OUT/'execution.lock'):
        if a.stage=='prepare':prepare()
        elif a.stage=='summary':summarize(a.pool)
        else:run(a.pool,a.start,a.stop if a.stop is not None else len(check()['seeds'][a.pool]))
