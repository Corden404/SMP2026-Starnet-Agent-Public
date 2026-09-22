"""Round23 E1: exact root TopologyPlan acceleration appended after PC2."""
_R23OriginalRestrictedRollout = RestrictedRollout

@lru_cache(maxsize=512)
def r23_e1_plan(ids, edges, eligible):
    allowed=set(eligible)
    base=coefficients(ids,edges,MODEL_VERSION)
    shields=[]
    for i in ids:
        if i not in allowed:continue
        ns=tuple(j for j in ids if j!=i)
        es=tuple(e for e in edges if i not in e)
        shields.append((i,ns,coefficients(ns,es,MODEL_VERSION)))
    cuts=[]
    for u,v in edges:
        if u not in allowed or v not in allowed:continue
        es=tuple(e for e in edges if e!=(u,v))
        cuts.append((u,v,ids,coefficients(ids,es,MODEL_VERSION)))
    return base,tuple(shields),tuple(cuts)

class R23E1Restricted(_R23OriginalRestrictedRollout):
    def candidates(self,state,budget,negative=False):
        nodes,edges,effects,observations=state
        ids=tuple(sorted(nodes))
        effective=tuple(sorted({tuple(sorted((u,v))) for u,v in edges if u in nodes and v in nodes and u!=v}))
        if set(effective)!=set(edges):
            return super().candidates(state,budget,negative)
        allowed=tuple(sorted(self.eligible))
        base,shields,cuts=r23_e1_plan(ids,effective,allowed)
        total=sum(nodes[i]['w']*v for i,v in zip(ids,base))
        influence=dict(zip(ids,base));allowed_set=set(allowed);actions=[]
        def add(kind,u,v,gain):
            if (negative or gain>.25) and u in allowed_set and (kind!='cut' or v in allowed_set):
                actions.append({'action':kind,'target_node_1':u,'target_node_2':v,
                    'prompt_id':1 if kind=='comm' else None,'cost':self.COST[kind],
                    'gain':gain,'roi':gain/self.COST[kind]})
        if budget>=2:
            for i,d in nodes.items():
                if d['comm_left']>0:
                    mean,_=self.effect(i,d,effects,observations)
                    add('comm',i,None,influence[i]*mean*.5**(3-d['comm_left']))
        if budget>=5 and len(nodes)>1:
            for i,ns,values in shields:
                modified=sum(nodes[j]['w']*v for j,v in zip(ns,values))
                add('shield',i,None,modified-total)
        if budget>=3:
            for u,v,row_ids,values in cuts:
                modified=sum(nodes[j]['w']*c for j,c in zip(row_ids,values))
                add('cut',u,v,modified-total)
        actions.sort(key=lambda a:(-a['roi'],-a['gain'],a['target_node_1']))
        return actions

RestrictedRollout=R23E1Restricted

def r23_e1_clear():
    r23_e1_plan.cache_clear()
