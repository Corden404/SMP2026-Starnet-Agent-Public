"""Round5 P1: pre-registered, bounded public-API probes, no model calls."""
import argparse
import copy
import json
import math
import time
from pathlib import Path
import networkx as nx
from round4_common import ROOT, save, digest, file_hash
from recovery_store import Store, Journal, RecoveryStop, InfrastructureFailure
from recovery_env import Gate, Transport, DurableEnv

OUT=ROOT/'agent_notes/experiments/round5'
SERVER='http://8.222.218.162:5000'


def score(seed, stationary=False):
    nodes={d['id']:d for d in seed['nodes']}
    adj={i:{i} for i in nodes}
    for u,v in seed['edges']:
        if u in nodes and v in nodes:adj[u].add(v);adj[v].add(u)
    if stationary:
        unseen=set(nodes);value=0.
        while unseen:
            stack=[unseen.pop()];comp=[]
            while stack:
                i=stack.pop();comp.append(i)
                for j in adj[i]&unseen:unseen.remove(j);stack.append(j)
            value+=len(comp)*sum(len(adj[i])*nodes[i]['w'] for i in comp)/sum(len(adj[i]) for i in comp)
        return value
    x={i:d['w'] for i,d in nodes.items()}
    for _ in range(50):x={i:sum(x[j] for j in js)/len(js) for i,js in adj.items()}
    return sum(x.values())


def seed(graph, weights=None, persona='中立', r=.2):
    ids=sorted(graph)
    weights=weights or {i:0. for i in ids}
    return {'global_setting':{'max_budget':100,'max_api_calls':120},
        'original_total':sum(weights.values()),
        'nodes':[{'id':i,'w':weights[i],'r':r,'persona':persona,'comm_left':3} for i in ids],
        'edges':[list(e) for e in sorted(tuple(sorted(e)) for e in graph.edges())],
        'prompts':{'1':15.,'2':10.,'3':-5.}}


def prediction(s,actions):
    state=copy.deepcopy(s);nodes={d['id']:d for d in state['nodes']};budget=s['global_setting']['max_budget'];receipts=[]
    for name,args in actions:
        if name=='scan_node':
            i=args[0];d=nodes.get(i);budget-=.5
            res=None if d is None else {**{k:d[k] for k in ('id','w','persona','comm_left')},'neighbors':sorted(v if u==i else u for u,v in state['edges'] if i in (u,v))}
        elif name=='communicate':
            i,p=args;d=nodes[i];d['w']+=s['prompts'][str(p)]*d['r']*.5**(3-d['comm_left']);d['comm_left']-=1;budget-=2
            res={'status':'success','new_w':d['w']}
        elif name=='cut_link':
            state['edges']=[e for e in state['edges'] if set(e)!=set(args)];budget-=3;res=True
        elif name=='shield_node':
            del nodes[args[0]];state['edges']=[e for e in state['edges'] if args[0] not in e];budget-=5;res=True
        receipts.append({'action':name,'args':args,'result':res,'budget':budget})
    state['nodes']=list(nodes.values())
    return {'finite50':score(state),'stationary':score(state,True),'receipts':receipts,'final_state':state,'budget':budget}


def register():
    path=OUT/'mechanism-probes.jsonl'
    if path.exists():return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
    graphs=[('isolated',nx.empty_graph(4)),('path',nx.path_graph(50)),('star',nx.star_graph(9)),('ring',nx.cycle_graph(10)),('bridge',nx.barbell_graph(5,0)),('weak50',nx.barbell_graph(20,10))]
    rows=[]
    def add(name,s,actions=(),group=None,repeat=0,pair=None):
        row={'id':name,'seed':s,'seed_sha256':digest(s),'base_graph_group':group or name,'variant':'mechanism',
            'simulator_version':'direct-synchronous-finite50-v1','belief_version':'none','continuation_policy':'fixed-probe-v1','selector':'scripted-no-LLM',
            'repeat':repeat,'purpose':'mechanism-identification','actions':list(actions),'pair':pair,'prediction':prediction(s,actions),
            'source_sha256':file_hash(__file__),'registered_at':time.time()}
        row['prediction_sha256']=digest(row);rows.append(row)
    for kind,g in graphs:
        g=nx.relabel_nodes(g,{i:i+1 for i in g})
        n=len(g);mixed={i:(20. if i<=n//2 else -20.) for i in g}
        for variant in range(4):
            w={i:0. for i in g};w[1]=20.*(variant+1)
            gg=g
            if variant>=2:w=mixed
            if variant==3:
                mapping={i:n+1-i for i in g};gg=nx.relabel_nodes(g,mapping);w={mapping[i]:v for i,v in mixed.items()}
            add(f'score-{kind}-{variant}',seed(gg,w),group=kind)
    for persona in ['和平','中立','暴力']:
        for r in [.2,1.5]:
            for w in [-20.,20.]:
                s=seed(nx.empty_graph(range(1,5)),{i:w for i in range(1,5)},persona,r)
                actions=[('scan_node',[i]) for i in range(1,5)]
                actions += [('communicate',[i,p]) for i,prompts in [(1,[1]*3),(2,[2]*3),(3,[3]*3),(4,[1,2,1])] for p in prompts]
                add(f'comm-{persona}-{r}-{w}',s,actions)
    for index in [0,6,10,18,25,34]:
        original=rows[index]
        add(original['id']+'-repeat',original['seed'],original['actions'],group=original['base_graph_group'],repeat=1,pair=original['id'])
    g=nx.relabel_nodes(nx.barbell_graph(5,0),{i:i+1 for i in range(10)})
    s=seed(g,{i:20. if i<=5 else -20. for i in g})
    add('structure-cut',s,[('scan_node',[5]),('scan_node',[6]),('cut_link',[5,6])],group='structure-cut')
    g.remove_edge(5,6);add('structure-cut-direct',seed(g,{i:20. if i<=5 else -20. for i in g}),group='structure-cut',pair='structure-cut')
    g=nx.star_graph(range(1,11));s=seed(g,{i:-20. if i==1 else 20. for i in g})
    add('structure-shield',s,[('scan_node',[1]),('shield_node',[1])],group='structure-shield')
    g.remove_node(1);add('structure-shield-direct',seed(g,{i:20. for i in g}),group='structure-shield',pair='structure-shield')
    g=nx.path_graph(range(1,5));s=seed(g,{i:20. if i%2 else -20. for i in g})
    scans=[('scan_node',[1]),('scan_node',[2])]
    add('order-comm-cut',s,scans+[('communicate',[1,1]),('cut_link',[1,2])],group='order')
    add('order-cut-comm',s,scans+[('cut_link',[1,2]),('communicate',[1,1])],group='order',pair='order-comm-cut')
    assert len(rows)==48
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        f.flush()
        import os;os.fsync(f.fileno())
    save(OUT/'mechanism-registration.json',{'manifest_sha256':file_hash(path),'count':48,'registered_at':time.time(),'source_sha256':file_hash(__file__)})
    return rows


def equal(a,b):
    if type(a) in (float,int) and type(b) in (float,int):return math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8)
    if isinstance(b,dict):return isinstance(a,dict) and all(k in a and equal(a[k],v) for k,v in b.items())
    if isinstance(b,list):return isinstance(a,list) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def execute(row,store,gate,reproduce=False):
    ident={k:row[k] for k in ['id','seed_sha256','base_graph_group','variant','simulator_version','belief_version','continuation_policy','selector','repeat','purpose','prediction_sha256']}
    ident.update(runner_sha256=file_hash(__file__),reproduce=reproduce)
    tid=store.trial(ident)
    prior=store.execute('SELECT status,result FROM trials WHERE id=?',(tid,)).fetchone()
    if prior['status']=='COMPLETED':return json.loads(prior['result'])
    attempts=store.execute('SELECT COUNT(*) FROM attempts WHERE trial=?',(tid,)).fetchone()[0]
    if attempts>=3:raise RecoveryStop('LOGICAL_SESSION_ATTEMPT_CAP',row['id'])
    aid=store.attempt(tid);start=time.monotonic()
    try:
        env=DurableEnv(SERVER,row['seed'],Journal(store,tid,aid),Transport(store,gate,aid))
        events=[];errors=[]
        initial=env.get_remaining_budget()
        if not equal(initial,row['seed']['global_setting']['max_budget']):errors.append({'kind':'initial_budget','actual':initial})
        for predicted in row['prediction']['receipts']:
            value=getattr(env,predicted['action'])(*predicted['args'])
            if predicted['action']=='scan_node' and value is not None:value={**value,'neighbors':sorted(value['neighbors'])}
            budget=env.get_remaining_budget()
            event={'action':predicted['action'],'args':predicted['args'],'result':value,'budget':budget};events.append(event)
            if not equal(value,predicted['result']) or not equal(budget,predicted['budget']):errors.append({'kind':'receipt','actual':event,'predicted':predicted})
        actual=env.trigger_eval();pred=row['prediction']['finite50']
        if abs(actual-pred)>.0051 or round(pred,2)!=actual:errors.append({'kind':'score','actual':actual,'predicted':pred})
        result={'id':row['id'],'actual':actual,'prediction_sha256':row['prediction_sha256'],'errors':errors,'receipts':events,'elapsed_seconds':time.monotonic()-start,'reproduce':reproduce}
        store.complete(tid,aid,result);gate.completed()
        return result
    except RecoveryStop as e:
        store.end_attempt(aid,'STOPPED',str(e))
        if isinstance(e,InfrastructureFailure):gate.outage(e.detail,trial_failure=True)
        raise


def run(limit):
    rows=register();store=Store(OUT/'mechanism-journal.sqlite');gate=Gate(store)
    # Known public endpoints only. First two registered probes serve as anchor receipts.
    if store.get('circuit') is None:store.put('circuit',{'phase':'READY','scope':'first two registered anchor probes; concurrency=1'})
    results=[]
    try:
        for row in rows[:limit]:
            result=execute(row,store,gate);results.append(result)
            save(OUT/'mechanism-progress.json',{'results':results,'logical_completed':len(results),'glm_calls':0,'tokens':0})
            if len(results)==2:save(OUT/'P1-timing.json',{'samples':[r['elapsed_seconds'] for r in results],'remaining_estimate_seconds':sum(r['elapsed_seconds'] for r in results)/2*46,'limitation':'score-only first anchors; communication probes take more requests'})
            if result['errors']:
                reproduced=execute(row,store,gate,True)
                save(OUT/'mechanism-validation.json',{'status':'FAILED','failed_probe':row['id'],'first':result,'replication':reproduced,'completed':len(results),'glm_calls':0,'tokens':0,'decision':'STOP_P2_THROUGH_P7'})
                return
        lookup={r['id']:r for r in results};pair_errors=[]
        for row in rows[:limit]:
            if row['pair'] and row['pair'] in lookup:
                a,b=lookup[row['id']],lookup[row['pair']]
                if a['actual']!=b['actual']:pair_errors.append([row['id'],row['pair']])
                if row['repeat'] and not equal(a['receipts'],b['receipts']):pair_errors.append([row['id'],'receipt repeat mismatch'])
        save(OUT/'mechanism-validation.json',{'status':'PASSED' if limit==48 and not pair_errors else 'PARTIAL' if limit<48 else 'FAILED',
            'completed':len(results),'pair_errors':pair_errors,'results':results,'glm_calls':0,'tokens':0})
    except RecoveryStop as e:
        save(OUT/'mechanism-validation.json',{'status':'BLOCKED_INFRASTRUCTURE','reason':str(e),'completed':len(results),'results':results,'glm_calls':0,'tokens':0,'decision':'STOP_DEPENDENT_STAGES'})
        raise
    finally:store.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');parser.add_argument('--limit',type=int,default=48)
    args=parser.parse_args()
    if args.register:print(len(register()))
    else:run(args.limit)
