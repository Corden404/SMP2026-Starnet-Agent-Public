"""Round23 E2: CandidateSkeleton layer on top of E1, preserving exact arithmetic order."""
@lru_cache(maxsize=512)
def r23_e2_plan(ids, edges, eligible):
    base,shields,cuts=r23_e1_plan(ids,edges,eligible)
    allowed=set(eligible)
    base_pairs=tuple(zip(ids,base))
    comm=tuple((i,base[pos]) for pos,i in enumerate(ids) if i in allowed)
    shield_rows=tuple((i,tuple(zip(ns,values))) for i,ns,values in shields)
    cut_rows=tuple((u,v,tuple(zip(row_ids,values))) for u,v,row_ids,values in cuts)
    return base_pairs,comm,shield_rows,cut_rows

class R23E2Restricted(_R23OriginalRestrictedRollout):
    def candidates(self,state,budget,negative=False):
        nodes,edges,effects,observations=state
        ids=tuple(sorted(nodes))
        effective=tuple(sorted({tuple(sorted((u,v))) for u,v in edges if u in nodes and v in nodes and u!=v}))
        if set(effective)!=set(edges):
            return super().candidates(state,budget,negative)
        allowed=tuple(sorted(self.eligible))
        base_pairs,comm,shields,cuts=r23_e2_plan(ids,effective,allowed)
        total=sum(nodes[i]['w']*v for i,v in base_pairs);actions=[]
        def add(kind,u,v,gain):
            if negative or gain>.25:
                actions.append({'action':kind,'target_node_1':u,'target_node_2':v,
                    'prompt_id':1 if kind=='comm' else None,'cost':self.COST[kind],
                    'gain':gain,'roi':gain/self.COST[kind]})
        if budget>=2:
            for i,influence in comm:
                d=nodes[i]
                if d['comm_left']>0:
                    mean,_=self.effect(i,d,effects,observations)
                    add('comm',i,None,influence*mean*.5**(3-d['comm_left']))
        if budget>=5 and len(nodes)>1:
            for i,row in shields:
                modified=sum(nodes[j]['w']*v for j,v in row)
                add('shield',i,None,modified-total)
        if budget>=3:
            for u,v,row in cuts:
                modified=sum(nodes[j]['w']*c for j,c in row)
                add('cut',u,v,modified-total)
        actions.sort(key=lambda a:(-a['roi'],-a['gain'],a['target_node_1']))
        return actions

RestrictedRollout=R23E2Restricted

def r23_e2_clear():
    r23_e2_plan.cache_clear();r23_e1_plan.cache_clear()
