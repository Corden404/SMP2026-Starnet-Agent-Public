"""Independent RNG streams and predeclared crossed synthetic distributions."""
import itertools
import json
import random
from collections import Counter

import networkx as nx
from round4_common import OUT, digest, file_hash, save

FAMILIES = ('ba', 'ws', 'er', 'sbm', 'disconnected', 'tree')
SCALES = (.55, 1., 1.5)
PERSONAS = ('和平', '中立', '暴力')
DISTRIBUTIONS = ((.35,.35,.30),(.2,.2,.6),(.65,.25,.1))
MODES = ('degree_w', 'degree_r', 'community', 'w_r')


def specs():
    result=[]
    def add(pool,f,s,p,rep=0,mode='independent',n=50,ood=False):
        i=sum(x['pool']==pool for x in result)
        result.append({'case':f'{pool}-{i:03}', 'pool':pool, 'family':FAMILIES[f],
                       'scale':SCALES[s], 'persona_mix':p, 'rep':rep, 'mode':mode,
                       'n':n, 'ood':ood, 'generator_version':1})
    cells=list(itertools.product(range(6),range(3),range(3)))
    for pool,repeats in [('dev',2),('select',1),('final',1)]:
        for rep in range(repeats):
            for f,s,p in cells:add(pool,f,s,p,rep)
    for f,s in itertools.product(range(6),range(3)):
        add('final',f,s,(f+s)%3,1)
    for f,c,s in itertools.product(range(6),range(4),range(3)):
        add('corr',f,s,(f+c+s)%3,mode=MODES[c])
    for f,s,rep in itertools.product(range(6),range(3),range(2)):
        add('final',f,s,(f+s+rep)%3,rep,MODES[(f+s+rep)%4],ood=True)
        add('stress',f,s,(f+s+rep)%3,rep,n=100)
    return result


def generate(spec):
    def rng(tag):return random.Random(int(digest([spec,tag]),16))
    gr=rng('graph'); pr=rng('persona'); wr=rng('w'); rr=rng('r'); ir=rng('ids')
    n=spec['n'];f=spec['family'];ood=spec['ood'];gs=gr.randrange(2**32)
    if f=='ba':g=nx.barabasi_albert_graph(n, gr.choice([1,2,3] if not ood else [4,5]), seed=gs)
    elif f=='ws':g=nx.watts_strogatz_graph(n, gr.choice([2,4,6] if not ood else [8,10]), gr.uniform(.1,.5), seed=gs)
    elif f=='er':g=nx.gnp_random_graph(n, gr.uniform(.05,.16) if not ood else gr.uniform(.18,.25), seed=gs)
    elif f=='sbm':
        sizes=[n//2,n-n//2]
        g=nx.stochastic_block_model(sizes,[[.17,.008],[.008,.17]] if not ood else [[.28,.002],[.002,.28]],seed=gs)
    elif f=='disconnected':g=nx.disjoint_union(nx.barabasi_albert_graph(n//2,gr.choice([1,2]),seed=gs),nx.watts_strogatz_graph(n-n//2,4,.2,seed=gs+1))
    else:g=nx.random_tree(n,seed=gs) if hasattr(nx,'random_tree') else nx.random_labeled_tree(n,seed=gs)
    personas=pr.choices(PERSONAS,DISTRIBUTIONS[spec['persona_mix']],k=n)
    if spec['mode']=='community':
        communities=list(nx.algorithms.community.greedy_modularity_communities(g)) if g.number_of_edges() else [set(g)]
        order=[i for c in sorted(communities,key=lambda c:min(c)) for i in sorted(c)]
        values=sorted(personas)
        for i,p in zip(order,values):personas[i]=p
    nodes={}
    for i in range(n):
        p=personas[i];lo,hi={'和平':(5,30),'中立':(-12,12),'暴力':(-50,-12)}[p]
        rlo,rhi={'和平':(.8,1.8),'中立':(.5,1.4),'暴力':(.05,.5)}[p]
        nodes[i]={'id':i,'persona':p,'w':wr.uniform(lo,hi),'r':rr.uniform(rlo,rhi)*spec['scale'],'comm_left':3}
    for p in PERSONAS:
        ids=[i for i in nodes if nodes[i]['persona']==p]
        if spec['mode'] in ('degree_w','degree_r'):
            key='w' if spec['mode']=='degree_w' else 'r'
            values=sorted(nodes[i][key] for i in ids)
            order=sorted(ids,key=lambda i:(g.degree(i),i),reverse=key=='w')
            for i,v in zip(order,values):nodes[i][key]=v
        elif spec['mode']=='w_r':
            values=sorted(nodes[i]['r'] for i in ids)
            order=sorted(ids,key=lambda i:nodes[i]['w'],reverse=spec['ood'])
            for i,v in zip(order,values):nodes[i]['r']=v
    shuffled=list(range(1,n+1));ir.shuffle(shuffled)
    for i in nodes:
        nodes[i]['id']=shuffled[i]
        for k in ('w','r'):nodes[i][k]=round(nodes[i][k],5)
    rows=sorted(nodes.values(),key=lambda d:d['id'])
    return {'global_setting':{'max_budget':2*n,'max_api_calls':250 if n==100 else 120},
            'original_total':sum(d['w'] for d in rows),'nodes':rows,
            'edges':sorted([sorted([shuffled[u],shuffled[v]]) for u,v in g.edges()]),
            'prompts':{'1':15.,'2':10.,'3':-5.}}


def manifest():
    entries=[]
    for spec in specs():
        seed=generate(spec)
        p=OUT/'seeds'/(spec['case']+'.json')
        if p.exists():
            if digest(json.loads(p.read_text(encoding='utf-8'))) != digest(seed):
                raise RuntimeError('Seed drift: '+spec['case'])
        else:save(p,seed)
        entries.append({'spec':spec,'seed_sha256':digest(seed)})
    if len({x['seed_sha256'] for x in entries}) != len(entries):raise AssertionError('Seed collision')
    result={'schema':1,'generator_sha256':file_hash(__file__),'entries':entries,
            'counts':dict(Counter(x['spec']['pool'] for x in entries)),
            'final_usage':'sealed until candidate freeze; generation/hashes only'}
    target=OUT/'manifest.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=result:
        raise RuntimeError('Manifest drift')
    save(target,result)
    return result


def load_case(case, allow_final=False):
    if case.startswith('final-') and not allow_final:raise ValueError('Final set is sealed')
    spec=next(x for x in specs() if x['case']==case)
    path=OUT/'seeds'/(case+'.json')
    seed=json.loads(path.read_text(encoding='utf-8'))
    return spec,seed


if __name__=='__main__':
    print(json.dumps({'counts':manifest()['counts'],'final_attributes_viewed':False}))
