"""Round25 fresh score study for B2R."""
from __future__ import annotations
import argparse,concurrent.futures,copy,json
from pathlib import Path
import round25_state as s
import round25_common as h
import round24_study as oldstudy
from round9_lock import ExclusiveBatch
OUT=s.OUT/'study';POOLS=('dev','integration','confirm','stress')
DEPS=('workspace/round25_study.py','tests/test_round25_study.py','workspace/round25_common.py','workspace/round25_policy.py','tests/test_round25_core.py')
def code():return h.fingerprint(DEPS)
def specs(pool):
    if pool=='dev':
        rows=[dict(family=f,scale=sc,persona_mix=pm,rep=270,mode='independent',n=50,ood=False,generator_version=1) for f in h.FAMILIES for sc in h.SCALES for pm in range(3)]
    elif pool=='integration':
        rows=[dict(family=f,scale=h.SCALES[i%3],persona_mix=(i+2)%3,rep=271,mode='independent',n=50,ood=False,generator_version=1) for i,f in enumerate(h.FAMILIES)]
    elif pool=='confirm':
        rows=[dict(family=f,scale=sc,persona_mix=pm,rep=rep,mode='independent',n=50,ood=False,generator_version=1) for rep in range(280,284) for f in h.FAMILIES for sc in h.SCALES for pm in range(3)]
    elif pool=='stress':
        rows=copy.deepcopy(h.c.stress_design.specs())
        for x in rows:x['rep']=290
    else:raise ValueError(pool)
    for i,x in enumerate(rows):x.update(case='r25-%s-%03d'%(pool,i),pool='r25-'+pool,round25=1)
    return rows
def entries(pool):return [dict(spec=x,sha256=s.digest(h.c.generate(x))) for x in specs(pool)]
def prepare():
    s.preserved();hyp=s.read(s.OUT/'hypothesis/locked.json')
    if hyp['rule']!=dict(label='S2',budget=20.0,incumbent='comm',all_world_suffixes_contain_comm=True):raise RuntimeError('Hypothesis drift')
    seeds={p:entries(p) for p in POOLS};fresh=[e['sha256'] for xs in seeds.values() for e in xs]
    old=oldstudy.known_graph_hashes()
    if len(set(fresh))!=len(fresh):raise RuntimeError('Round25 internal graph collision')
    overlap=set(fresh)&old
    if overlap:raise RuntimeError('Round25 graph overlap '+repr(list(overlap)[:3]))
    for v in h.VARIANTS:h.load(v,'prepare')
    protocol=dict(code=code(),seeds=seeds,workers=2,dev_variants=['PC2','B2B','B2R'],
      dev_gate='B2R-PC2 attributable mean>0, w/o-best2>=0, wins>losses, >=3 nonzero, >=1 veto, B2R mean>=B2B mean, B2R losses<=B2B losses, integrity pass.',
      confirm_gate='B2R confirm score gates + >=B2B mean/fewer-or-equal losses; stress nonnegative robust; >=1 veto in confirm or stress; integrity pass.',
      no_speed_gate=True,no_api_before_real=True,real_model_order=list(s.REAL_MODEL_ORDER))
    s.write(OUT/'protocol.json',protocol,True)
    s.update('R2','COMPLETE','B2R包/边界测试通过；fresh dev/integration/confirm/stress 与全部旧图hash无重叠。',
             next_action='R3运行fresh dev54：PC2/B2B/B2R共162局。',evidence='study/protocol.json')
    s.update('R3','RUNNING','fresh dev54协议冻结；只看最终分数与veto有效性。',evidence='study/protocol.json')
    print(json.dumps(dict(prepared=True,counts={k:len(v) for k,v in seeds.items()},old_hashes=len(old)),ensure_ascii=False),flush=True)
def check():
    p=s.read(OUT/'protocol.json')
    if p['code']!=code():raise RuntimeError('Study source drift')
    return p
def variants(pool):
    if pool=='dev':return ['PC2','B2B','B2R']
    d=s.read(OUT/'development-decision.json')
    if not d.get('eligible_b2r'):raise RuntimeError('B2R not selected')
    if pool=='integration':return ['PC2','B2R']
    return ['PC2','B2B','B2R']
def run_one(pool,index,variant):
    p=check();entry=p['seeds'][pool][index];spec=entry['spec']
    if s.digest(h.c.generate(spec))!=entry['sha256']:raise RuntimeError('Seed drift')
    mode='integration' if pool=='integration' else 'fast';ident=dict(entry=entry,variant=variant,code=p['code'],mode=mode)
    return h.checked(OUT/'runs'/pool/(spec['case']+'-'+variant+'.json'),ident,lambda:h.run_game(spec,variant,mode))
def group(pool,index):
    vs=variants(pool);shift=index%len(vs);order=vs[shift:]+vs[:shift]
    if (index//len(vs))%2:order=order[::-1]
    rows=[run_one(pool,index,v) for v in order];return dict(case=rows[0]['spec']['case'],games=len(rows))
def integrity(x):return x['prefix_drift']==0 and x['max_operations']<120 and x['max_calls']<120 and x['max_forward_error']<=1e-8
def mechanism(row,comparison):
    if comparison=='pc2':return bool(h.selections(row))
    if comparison=='b2b':return bool(h.vetoes(row))
    raise ValueError(comparison)
def compare(pool,candidate,baseline='PC2'):
    p=check();rawpairs=[];attrpairs=[];prefix=0;maxerr=0.;maxops=maxcalls=0;drifts=[];selected=veto=changed=0
    comp='pc2' if baseline=='PC2' else 'b2b'
    for e in p['seeds'][pool]:
        case=e['spec']['case'];a=s.read(OUT/'runs'/pool/(case+'-'+baseline+'.json'));b=s.read(OUT/'runs'/pool/(case+'-'+candidate+'.json'))
        for r in (a,b):
            if r.get('status')!='PASS' or r['identity']['entry']!=e or r['identity']['code']!=p['code']:raise RuntimeError('Mixed cohort')
            maxerr=max(maxerr,h.c.audit_game(r));maxops=max(maxops,r['operations']);maxcalls=max(maxcalls,r['logical_calls'])
        delta=b['final_score']-a['final_score'];samebag=h.bag(a)==h.bag(b);sameprefix=h.scan_prefix(a)==h.scan_prefix(b)
        prefix+=int(not sameprefix);selected+=len(h.selections(b));veto+=len(h.vetoes(b));changed+=int(not samebag or abs(delta)>1e-8)
        unexplained=(not mechanism(b,comp)) and (not samebag or abs(delta)>1e-8)
        if unexplained:drifts.append(dict(case=case,delta=delta,same_action_bag=samebag))
        rawpairs.append(dict(case=case,family=e['spec']['family'],delta=delta))
        attrpairs.append(dict(case=case,family=e['spec']['family'],delta=delta if not unexplained else 0.0))
    return dict(raw=h.c.paired_stats(rawpairs),attributable=h.c.paired_stats(attrpairs),prefix_drift=prefix,
        unexplained=drifts,selections=selected,vetoes=veto,changed_games=changed,max_operations=maxops,max_calls=maxcalls,max_forward_error=maxerr)
def eligible_dev(comp):
    b=comp['B2B-PC2'];r=comp['B2R-PC2'];d=comp['B2R-B2B'];x=r['attributable'];y=b['attributable']
    return (x['mean']>0 and x['without_best_two']>=0 and x['wins']>x['losses'] and (x['wins']+x['losses'])>=3 and r['vetoes']>=1 and
            x['mean']+1e-8>=y['mean'] and x['losses']<=y['losses'] and d['attributable']['mean']>=-1e-8 and
            integrity(b) and integrity(r) and integrity(d))
def summarize(pool):
    if pool=='dev':
        comp={'B2B-PC2':compare(pool,'B2B','PC2'),'B2R-PC2':compare(pool,'B2R','PC2'),'B2R-B2B':compare(pool,'B2R','B2B')}
    elif pool=='integration':
        comp={'B2R-PC2':compare(pool,'B2R','PC2')}
    else:
        comp={'B2B-PC2':compare(pool,'B2B','PC2'),'B2R-PC2':compare(pool,'B2R','PC2'),'B2R-B2B':compare(pool,'B2R','B2B')}
    rec=dict(status='PASS',pool=pool,graphs=len(check()['seeds'][pool]),comparisons=comp);s.write(OUT/'summaries'/(pool+'.json'),rec,True)
    if pool=='dev':
        ok=eligible_dev(comp);d=dict(eligible_b2r=ok,code=check()['code'],summary_sha=s.sha(OUT/'summaries/dev.json'),
          b2b_mean=comp['B2B-PC2']['attributable']['mean'],b2r_mean=comp['B2R-PC2']['attributable']['mean'],
          b2r_without_best_two=comp['B2R-PC2']['attributable']['without_best_two'],b2r_wtl=[comp['B2R-PC2']['attributable'][k] for k in ('wins','ties','losses')],
          vetoes=comp['B2R-PC2']['vetoes'],b2r_minus_b2b=comp['B2R-B2B']['attributable']['mean'])
        s.write(OUT/'development-decision.json',d,True)
        s.update('R3','RUNNING' if ok else 'COMPLETE','fresh dev54：B2B %+.6f；B2R %+.6f；B2R-B2B %+.6f；veto=%d；资格%s。'%(
            d['b2b_mean'],d['b2r_mean'],d['b2r_minus_b2b'],d['vetoes'],ok),
            next_action='运行fresh integration6。' if ok else 'B2R未过fresh分数/veto门槛，跳过R4/R5并收尾。',evidence='study/development-decision.json')
        if not ok:
            for k in ('R4','R5'):s.update(k,'SKIPPED_GATE','R3 B2R未过fresh开发门槛。',evidence='study/development-decision.json')
    elif pool=='integration':
        d=s.read(OUT/'development-decision.json');ok=integrity(comp['B2R-PC2']) and not comp['B2R-PC2']['unexplained']
        d=dict(d,integration_safe=ok,integration_sha=s.sha(OUT/'summaries/integration.json'));s.write(OUT/'development-decision.json',d)
        s.update('R3','COMPLETE','fresh integration6安全=%s；mean=%+.6f。'%(ok,comp['B2R-PC2']['raw']['mean']),
            next_action='进入R4 confirm216+stress54。' if ok else '集成失败，跳过R4/R5。',evidence='study/summaries/integration.json')
        if not ok:
            for k in ('R4','R5'):s.update(k,'SKIPPED_GATE','R3集成失败。',evidence='study/summaries/integration.json')
    elif pool in ('confirm','stress'):
        cp=OUT/'summaries/confirm.json';sp=OUT/'summaries/stress.json'
        if cp.exists() and sp.exists():
            co=s.read(cp)['comparisons'];st=s.read(sp)['comparisons'];rco=co['B2R-PC2'];bco=co['B2B-PC2'];rst=st['B2R-PC2'];bst=st['B2B-PC2']
            x=rco['raw'];y=bco['raw'];z=rst['raw'];vetoes=rco['vetoes']+rst['vetoes']
            ok=(x['mean']>0 and x['without_best_two']>=0 and x['wins']>x['losses'] and x['mean']+1e-8>=y['mean'] and x['losses']<=y['losses'] and
                z['mean']>=0 and z['without_best_two']>=0 and vetoes>=1 and integrity(rco) and integrity(bco) and integrity(rst) and integrity(bst) and
                not rco['unexplained'] and not rst['unexplained'])
            dec=dict(candidate='B2R',eligible_real=ok,confirmed_local=ok and x['ci95_t'][0]>0 and x['ci95_family_bootstrap'][0]>0,
                confirm_mean=x['mean'],b2b_confirm_mean=y['mean'],stress_mean=z['mean'],vetoes=vetoes,code=check()['code'])
            s.write(OUT/'local-decision.json',dec,True)
            s.update('R4','COMPLETE','confirm/stress完成：B2R %+.6f/%+.6f；B2B confirm %+.6f；veto=%d；真实资格%s。'%(
                x['mean'],z['mean'],y['mean'],vetoes,ok),next_action='进入R5真实LLM。' if ok else '独立分数门槛未过，跳过R5并收尾。',evidence='study/local-decision.json')
            if not ok:s.update('R5','SKIPPED_GATE','R4独立分数门槛未通过。',evidence='study/local-decision.json')
    print(json.dumps(dict(pool=pool,comparisons={k:dict(raw=v['raw']['mean'],attr=v['attributable']['mean'],wtl=[v['attributable']['wins'],v['attributable']['ties'],v['attributable']['losses']],veto=v['vetoes'],drift=len(v['unexplained'])) for k,v in comp.items()}),ensure_ascii=False),flush=True)
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
