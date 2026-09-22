"""Round26 fresh score study: B2C1/B2CS."""
from __future__ import annotations
import argparse,concurrent.futures,copy,json
from pathlib import Path
import round26_state as s
import round26_common as h
from round9_lock import ExclusiveBatch
OUT=s.OUT/'study';POOLS=('dev','integration','confirm','stress');DEV_VARIANTS=('PC2','B2B','B2C1','B2CS')
DEPS=('workspace/round26_study.py','tests/test_round26_study.py','workspace/round26_common.py','workspace/round26_policy.py','tests/test_round26_core.py')
def code():return h.fingerprint(DEPS)
def specs(pool):
    if pool=='dev':
        rows=[dict(family=f,scale=sc,persona_mix=pm,rep=rep,mode='independent',n=50,ood=False,generator_version=1)
              for rep in (300,301) for f in h.FAMILIES for sc in h.SCALES for pm in range(3)]
    elif pool=='integration':
        rows=[dict(family=f,scale=h.SCALES[i%3],persona_mix=i%3,rep=302,mode='independent',n=50,ood=False,generator_version=1) for i,f in enumerate(h.FAMILIES)]
    elif pool=='confirm':
        rows=[dict(family=f,scale=sc,persona_mix=pm,rep=rep,mode='independent',n=50,ood=False,generator_version=1)
              for rep in range(310,314) for f in h.FAMILIES for sc in h.SCALES for pm in range(3)]
    elif pool=='stress':
        rows=copy.deepcopy(h.c.stress_design.specs())
        for x in rows:x['rep']=320
    else:raise ValueError(pool)
    for i,x in enumerate(rows):x.update(case='r26-%s-%03d'%(pool,i),pool='r26-'+pool,round26=1)
    return rows
def entries(pool):return [dict(spec=x,sha256=s.digest(h.c.generate(x))) for x in specs(pool)]
def known_graph_hashes():
    hashes=set();seen=set();base=s.ROOT/'agent_notes/experiments'
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
        if 'round26' in path.parts:continue
        try:walk(s.read(path))
        except (ValueError,OSError):pass
    return hashes
def prepare():
    s.preserved();hyp=s.read(s.OUT/'hypothesis/locked.json')
    if hyp['candidates']!={'B2C1':'label==C1','B2CS':'label==C1 and incumbent==shield'}:raise RuntimeError('Hypothesis drift')
    seeds={p:entries(p) for p in POOLS};fresh=[e['sha256'] for xs in seeds.values() for e in xs];old=known_graph_hashes()
    if len(set(fresh))!=len(fresh):raise RuntimeError('Round26 internal collision')
    overlap=set(fresh)&old
    if overlap:raise RuntimeError('Round26 graph overlap '+repr(list(overlap)[:3]))
    for v in DEV_VARIANTS:h.load(v,'prepare')
    protocol=dict(code=code(),seeds=seeds,dev_variants=list(DEV_VARIANTS),workers=2,
      dev_gate='Candidate-PC2 attributable mean>0; wob2>=0; wins>losses; >=3 nonzero; >=3 selected C1; losses<=B2B; wob2>=B2B; integrity.',
      selection='Eligible B2C1/B2CS with highest attributable mean; tie<=1e-8 prefers B2C1.',
      confirm_gate='Candidate confirm positive robust, losses<=B2B, wob2>=B2B; stress nonnegative robust; >=3 C1 confirm+stress; integrity.',
      no_speed_gate=True,no_api_before_real=True,real_model_order=list(s.REAL_MODEL_ORDER))
    s.write(OUT/'protocol.json',protocol,True)
    s.update('R2','COMPLETE','B2C1/B2CS包与边界测试通过；fresh dev108等cohort与全部旧图hash无重叠。',
      next_action='R3运行fresh dev108四策略共432局。',evidence='study/protocol.json')
    s.update('R3','RUNNING','fresh dev108协议冻结；只看最终分数与稳健性。',evidence='study/protocol.json')
    print(json.dumps(dict(prepared=True,counts={k:len(v) for k,v in seeds.items()},old_hashes=len(old)),ensure_ascii=False),flush=True)
def check():
    p=s.read(OUT/'protocol.json')
    if p['code']!=code():raise RuntimeError('Study source drift')
    return p
def variants(pool):
    if pool=='dev':return list(DEV_VARIANTS)
    d=s.read(OUT/'development-decision.json');v=d.get('selected')
    if not v:raise RuntimeError('No selected candidate')
    return ['PC2',v] if pool=='integration' else ['PC2','B2B',v]
def run_one(pool,index,variant):
    p=check();entry=p['seeds'][pool][index];spec=entry['spec']
    if s.digest(h.c.generate(spec))!=entry['sha256']:raise RuntimeError('Seed drift')
    mode='integration' if pool=='integration' else 'fast';ident=dict(entry=entry,variant=variant,code=p['code'],mode=mode)
    return h.checked(OUT/'runs'/pool/(spec['case']+'-'+variant+'.json'),ident,lambda:h.run_game(spec,variant,mode))
def group(pool,index):
    vs=variants(pool);shift=index%len(vs);order=vs[shift:]+vs[:shift]
    if (index//len(vs))%2:order=order[::-1]
    rs=[run_one(pool,index,v) for v in order];return dict(case=rs[0]['spec']['case'],games=len(rs))
def integrity(x):return x['prefix_drift']==0 and x['max_operations']<120 and x['max_calls']<120 and x['max_forward_error']<=1e-8
def compare(pool,candidate,baseline='PC2'):
    p=check();rawpairs=[];attrpairs=[];prefix=0;maxerr=0.;maxops=maxcalls=0;drifts=[];sels=veto=changed=0
    for e in p['seeds'][pool]:
        case=e['spec']['case'];a=s.read(OUT/'runs'/pool/(case+'-'+baseline+'.json'));b=s.read(OUT/'runs'/pool/(case+'-'+candidate+'.json'))
        for r in (a,b):
            if r.get('status')!='PASS' or r['identity']['entry']!=e or r['identity']['code']!=p['code']:raise RuntimeError('Mixed cohort')
            maxerr=max(maxerr,h.c.audit_game(r));maxops=max(maxops,r['operations']);maxcalls=max(maxcalls,r['logical_calls'])
        delta=b['final_score']-a['final_score'];samebag=h.bag(a)==h.bag(b);sameprefix=h.scan_prefix(a)==h.scan_prefix(b);prefix+=int(not sameprefix)
        sels+=len(h.selections(b));veto+=len(h.vetoes(b));changed+=int(not samebag or abs(delta)>1e-8)
        mechanism=bool(h.vetoes(b)) if baseline=='B2B' else bool(h.selections(b))
        unexplained=(not mechanism) and (not samebag or abs(delta)>1e-8)
        if unexplained:drifts.append(dict(case=case,delta=delta,same_action_bag=samebag))
        rawpairs.append(dict(case=case,family=e['spec']['family'],delta=delta));attrpairs.append(dict(case=case,family=e['spec']['family'],delta=delta if not unexplained else 0.0))
    return dict(raw=h.c.paired_stats(rawpairs),attributable=h.c.paired_stats(attrpairs),prefix_drift=prefix,unexplained=drifts,selections=sels,vetoes=veto,
      changed_games=changed,max_operations=maxops,max_calls=maxcalls,max_forward_error=maxerr)
def eligible_dev(comp,v):
    base=comp['B2B-PC2'];x=comp[v+'-PC2'];s1=x['attributable'];sb=base['attributable']
    return (s1['mean']>0 and s1['without_best_two']>=0 and s1['wins']>s1['losses'] and (s1['wins']+s1['losses'])>=3 and x['selections']>=3 and
      s1['losses']<=sb['losses'] and s1['without_best_two']+1e-8>=sb['without_best_two'] and integrity(base) and integrity(x) and integrity(comp[v+'-B2B']))
def summarize(pool):
    vs=variants(pool)
    if pool=='dev':
        comp={'B2B-PC2':compare(pool,'B2B','PC2')}
        for v in ('B2C1','B2CS'):
            comp[v+'-PC2']=compare(pool,v,'PC2');comp[v+'-B2B']=compare(pool,v,'B2B')
    elif pool=='integration':
        v=vs[1];comp={v+'-PC2':compare(pool,v,'PC2')}
    else:
        v=vs[2];comp={'B2B-PC2':compare(pool,'B2B','PC2'),v+'-PC2':compare(pool,v,'PC2'),v+'-B2B':compare(pool,v,'B2B')}
    rec=dict(status='PASS',pool=pool,graphs=len(check()['seeds'][pool]),comparisons=comp);s.write(OUT/'summaries'/(pool+'.json'),rec,True)
    if pool=='dev':
        elig=[v for v in ('B2C1','B2CS') if eligible_dev(comp,v)];prio={'B2C1':0,'B2CS':1};elig.sort(key=lambda v:(-comp[v+'-PC2']['attributable']['mean'],prio[v]));selected=elig[0] if elig else None
        d=dict(selected=selected,eligible=elig,code=check()['code'],summary_sha=s.sha(OUT/'summaries/dev.json'),
          b2b=dict(mean=comp['B2B-PC2']['attributable']['mean'],wob2=comp['B2B-PC2']['attributable']['without_best_two'],losses=comp['B2B-PC2']['attributable']['losses']),
          candidates={v:dict(mean=comp[v+'-PC2']['attributable']['mean'],wob2=comp[v+'-PC2']['attributable']['without_best_two'],
            wins=comp[v+'-PC2']['attributable']['wins'],ties=comp[v+'-PC2']['attributable']['ties'],losses=comp[v+'-PC2']['attributable']['losses'],
            selections=comp[v+'-PC2']['selections'],vetoes=comp[v+'-PC2']['vetoes'],eligible=eligible_dev(comp,v),
            vs_b2b_mean=comp[v+'-B2B']['attributable']['mean']) for v in ('B2C1','B2CS')})
        s.write(OUT/'development-decision.json',d,True)
        s.update('R3','RUNNING' if selected else 'COMPLETE','fresh dev108：B2B %+.6f wob2%+.6f；B2C1 %+.6f wob2%+.6f sel%d；B2CS %+.6f wob2%+.6f sel%d；选择%s。'%(
          d['b2b']['mean'],d['b2b']['wob2'],d['candidates']['B2C1']['mean'],d['candidates']['B2C1']['wob2'],d['candidates']['B2C1']['selections'],
          d['candidates']['B2CS']['mean'],d['candidates']['B2CS']['wob2'],d['candidates']['B2CS']['selections'],selected),
          next_action='运行fresh integration6。' if selected else '无候选过分数门槛，跳过R4/R5并收尾。',evidence='study/development-decision.json')
        if not selected:
            for k in ('R4','R5'):s.update(k,'SKIPPED_GATE','R3无候选通过fresh dev108分数门槛。',evidence='study/development-decision.json')
    elif pool=='integration':
        d=s.read(OUT/'development-decision.json');v=d['selected'];x=comp[v+'-PC2'];ok=integrity(x) and not x['unexplained']
        d=dict(d,integration_safe=ok,integration_sha=s.sha(OUT/'summaries/integration.json'));s.write(OUT/'development-decision.json',d)
        s.update('R3','COMPLETE','fresh integration6：%s安全=%s，mean=%+.6f。'%(v,ok,x['raw']['mean']),next_action='进入R4 confirm216+stress54。' if ok else '集成失败，跳过R4/R5。',evidence='study/summaries/integration.json')
        if not ok:
            for k in ('R4','R5'):s.update(k,'SKIPPED_GATE','R3集成安全失败。',evidence='study/summaries/integration.json')
    elif pool in ('confirm','stress'):
        cp=OUT/'summaries/confirm.json';sp=OUT/'summaries/stress.json'
        if cp.exists() and sp.exists():
            d=s.read(OUT/'development-decision.json');v=d['selected'];co=s.read(cp)['comparisons'];st=s.read(sp)['comparisons']
            x=co[v+'-PC2'];b=co['B2B-PC2'];z=st[v+'-PC2'];sb=st['B2B-PC2'];sx=x['raw'];bb=b['raw'];sz=z['raw'];event_count=x['selections']+z['selections']
            ok=(sx['mean']>0 and sx['without_best_two']>=0 and sx['wins']>sx['losses'] and sx['losses']<=bb['losses'] and sx['without_best_two']+1e-8>=bb['without_best_two'] and
                sz['mean']>=0 and sz['without_best_two']>=0 and event_count>=3 and integrity(x) and integrity(b) and integrity(z) and integrity(sb) and not x['unexplained'] and not z['unexplained'])
            dec=dict(candidate=v,eligible_real=ok,confirmed_local=ok and sx['ci95_t'][0]>0 and sx['ci95_family_bootstrap'][0]>0,
              confirm_mean=sx['mean'],b2b_confirm_mean=bb['mean'],stress_mean=sz['mean'],selected_events=event_count,code=check()['code'])
            s.write(OUT/'local-decision.json',dec,True)
            s.update('R4','COMPLETE','confirm/stress完成：%s %+.6f/%+.6f；B2B confirm%+.6f；C1事件%d；真实资格%s。'%(v,sx['mean'],sz['mean'],bb['mean'],event_count,ok),
              next_action='进入R5真实LLM。' if ok else '独立分数门槛未过，跳过R5并收尾。',evidence='study/local-decision.json')
            if not ok:s.update('R5','SKIPPED_GATE','R4独立分数门槛未通过。',evidence='study/local-decision.json')
    print(json.dumps(dict(pool=pool,comparisons={k:dict(raw=v['raw']['mean'],attr=v['attributable']['mean'],wtl=[v['attributable']['wins'],v['attributable']['ties'],v['attributable']['losses']],sel=v['selections'],veto=v['vetoes'],drift=len(v['unexplained'])) for k,v in comp.items()}),ensure_ascii=False),flush=True)
    return rec
def run(pool,start,stop):
    p=check();vs=variants(pool);n=len(p['seeds'][pool])
    if not 0<=start<stop<=n:raise ValueError('Invalid block')
    if pool in ('confirm','stress'):
        d=s.read(OUT/'development-decision.json')
        if not d.get('integration_safe'):raise RuntimeError('Integration not passed')
        s.update('R4','RUNNING',pool+'固定队列开始。',next_action='完整跑固定队列；不按中途分数停止。',evidence='study/runs/'+pool)
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as ex:
        pending={};cur=start
        while cur<min(stop,start+2):pending[ex.submit(group,pool,cur)]=cur;cur+=1
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                pending.pop(f);info=f.result()
                complete=sum(all((OUT/'runs'/pool/(e['spec']['case']+'-'+v+'.json')).exists() and s.read(OUT/'runs'/pool/(e['spec']['case']+'-'+v+'.json')).get('status')=='PASS' for v in vs) for e in p['seeds'][pool])
                step='R3' if pool in ('dev','integration') else 'R4'
                if complete%10==0 or complete==n:s.update(step,'RUNNING','%s已持久化%d/%d完整图；最近%s。'%(pool,complete,n,info['case']),next_action='继续固定队列。',evidence='study/runs/'+pool)
                print(json.dumps(dict(pool=pool,completed=complete,**info),ensure_ascii=False),flush=True)
                if cur<stop:pending[ex.submit(group,pool,cur)]=cur;cur+=1
    if all((OUT/'runs'/pool/(e['spec']['case']+'-'+v+'.json')).exists() for e in p['seeds'][pool] for v in vs):return summarize(pool)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','run','summary']);ap.add_argument('--pool',choices=POOLS,default='dev');ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int);a=ap.parse_args()
    with ExclusiveBatch(s.OUT/'execution.lock'):
        if a.stage=='prepare':prepare()
        elif a.stage=='summary':summarize(a.pool)
        else:run(a.pool,a.start,a.stop if a.stop is not None else len(check()['seeds'][a.pool]))
