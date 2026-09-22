"""Round22 root RestrictedRollout exact structural-row acceleration.

Appended after exact PC2. It changes no options, margins, worlds, scans or prompts.
Only candidate arithmetic is cached by topology; row dot products preserve score() order.
"""
_R22OriginalRestrictedRollout = RestrictedRollout

@lru_cache(maxsize=256)
def r22_root_rows(ids, edges, eligible, kind):
    rows={}
    allowed=set(eligible)
    if kind=='shield':
        for i in ids:
            if i not in allowed:continue
            ns=tuple(j for j in ids if j!=i)
            es=tuple(e for e in edges if i not in e)
            rows[i]=(ns,coefficients(ns,es,MODEL_VERSION))
    elif kind=='cut':
        for e in edges:
            if e[0] not in allowed or e[1] not in allowed:continue
            es=tuple(x for x in edges if x!=e)
            rows[e]=(ids,coefficients(ids,es,MODEL_VERSION))
    else:raise ValueError('Unknown root row kind')
    return rows

def r22_row_score(row,nodes):
    ids,values=row
    return sum(nodes[i]['w']*v for i,v in zip(ids,values))

class R22StableRestricted(_R22OriginalRestrictedRollout):
    def candidates(self,state,budget,negative=False):
        nodes,edges,effects,observations=state
        ids=tuple(sorted(nodes))
        effective=tuple(sorted({tuple(sorted((u,v))) for u,v in edges if u in nodes and v in nodes and u!=v}))
        # Preserve legacy behavior for non-canonical edge containers.
        if set(effective)!=set(edges):
            return super().candidates(state,budget,negative)
        total,influence=score(nodes,edges);actions=[];allowed=set(self.eligible)
        def add(kind,u,v,gain):
            if (negative or gain>.25) and u in allowed and (kind!='cut' or v in allowed):
                actions.append({'action':kind,'target_node_1':u,'target_node_2':v,
                    'prompt_id':1 if kind=='comm' else None,'cost':self.COST[kind],
                    'gain':gain,'roi':gain/self.COST[kind]})
        if budget>=2:
            for i,d in nodes.items():
                if i in allowed and d['comm_left']>0:
                    mean,_=self.effect(i,d,effects,observations)
                    add('comm',i,None,influence[i]*mean*.5**(3-d['comm_left']))
        if budget>=5 and len(nodes)>1:
            table=r22_root_rows(ids,effective,tuple(sorted(allowed)),'shield')
            for i in nodes:
                if i in allowed:add('shield',i,None,r22_row_score(table[i],nodes)-total)
        if budget>=3:
            table=r22_root_rows(ids,effective,tuple(sorted(allowed)),'cut')
            for u,v in sorted(edges):
                if u in nodes and v in nodes and u in allowed and v in allowed:
                    add('cut',u,v,r22_row_score(table[(u,v)],nodes)-total)
        actions.sort(key=lambda a:(-a['roi'],-a['gain'],a['target_node_1']))
        return actions

RestrictedRollout = R22StableRestricted

def r22_clear_private():
    r22_root_rows.cache_clear()
