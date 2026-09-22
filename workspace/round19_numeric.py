"""Private forecast kernel, appended after exact PC2 and frozen round18_cut1.

No env or seed access. All prospective selectors see belief, never the outer
world. Numerical rules follow frozen PC2, not recursive CT1. Acceleration has
independent caches; deadline-dependent equivalence is not promised.
"""

@lru_cache(maxsize=128)
def r19_rows(ids, edges, eligible, kind):
    rows={}
    if kind=='shield':
        for i in eligible:
            ns=tuple(j for j in ids if j!=i);es=tuple(e for e in edges if i not in e)
            rows[i]=(ns,r18_cut_coefficients(ns,es))
    elif kind=='cut':
        allowed=set(eligible)
        for e in edges:
            if e[0] in allowed and e[1] in allowed:
                es=tuple(x for x in edges if x!=e)
                rows[e]=(ids,r18_cut_coefficients(ids,es))
    else:raise ValueError('Unknown structure')
    return rows


def r19_row_value(row,nodes):
    return sum(nodes[i]['w']*v for i,v in zip(*row))


class R19Kernel(R18CutRollout):
    def candidates(self,state,budget,negative=False):
        nodes,edges,effects,observations=state
        ids=tuple(sorted(nodes))
        es=tuple(sorted({tuple(sorted(e)) for e in edges if e[0] in nodes and e[1] in nodes and e[0]!=e[1]}))
        if set(es)!=set(edges):return super().candidates(state,budget,negative)
        eligible=tuple(i for i in ids if i in self.eligible)
        total,influence=r18_cut_score(nodes,edges);actions=[]
        def add(kind,u,v,gain):
            if (negative or gain>.25) and u in self.eligible and (kind!='cut' or v in self.eligible):
                actions.append(dict(action=kind,target_node_1=u,target_node_2=v,prompt_id=1 if kind=='comm' else None,
                                    cost=self.COST[kind],gain=gain,roi=gain/self.COST[kind]))
        if budget>=2:
            for i,d in nodes.items():
                if d['comm_left']>0:add('comm',i,None,influence[i]*self.effect(i,d,effects,observations)[0]*.5**(3-d['comm_left']))
        if budget>=5 and len(nodes)>1:
            rows=r19_rows(ids,es,eligible,'shield')
            for i in nodes:
                if i in self.eligible:add('shield',i,None,r19_row_value(rows[i],nodes)-total)
        if budget>=3:
            rows=r19_rows(ids,es,eligible,'cut')
            for u,v in sorted(edges):
                if u in self.eligible and v in self.eligible:add('cut',u,v,r19_row_value(rows[(u,v)],nodes)-total)
        actions.sort(key=lambda a:(-a['roi'],-a['gain'],a['target_node_1']))
        return actions


def r19_slots(context):
    return min(context['operation_limit']-1-context['operation'],context['safe_limit']-context['calls'])


def r19_check(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Round19 complete forecast deadline')


def r19_s44(state,budget,slots,eligible,deadline,strict=False):
    r19_check(deadline)
    options=dict(SEARCH,rollout_seconds=min(3.,max(0.,deadline-time.monotonic())))
    engine=R19Kernel(*state,options,slots,eligible=eligible)
    actions=engine.candidates(engine.initial,budget)
    r19_check(deadline)
    if not actions or slots<=0:return None
    plans=engine.choose(budget,baseline_action=actions[0])
    r19_check(deadline)
    if strict and (not plans or not plans[0].get('search_completed') or plans[0].get('search_truncated')):
        raise TimeoutError('Frozen PC2 inner S44 incomplete')
    return plans[0]['first'] if plans else actions[0]


def r19_depth_continue(initial,budget,slots,eligible,first,world,deadline,depth=2):
    state=BudgetRollout.clone(initial);chosen=first;actions=[]
    engine=R19Kernel(*state,SEARCH,slots,eligible=eligible)
    while chosen is not None:
        r19_check(deadline)
        r19_legal(state,chosen,budget,slots,eligible)
        state=engine.apply(state,chosen,world,True);budget-=chosen['cost'];slots-=1
        actions.append(BudgetRollout.key(chosen))
        if budget<2 or slots<=0:break
        if len(actions)<=depth:chosen=r19_s44(state,budget,slots,eligible,deadline,strict=True)
        else:
            options=engine.candidates(state,budget);r19_check(deadline)
            chosen=options[0] if options else None
    r19_check(deadline)
    return dict(score=r18_cut_score(state[0],state[1])[0],actions=actions,budget=budget,slots=slots)


def r19_legal(state,action,budget,slots,eligible):
    kind=action['action'];u=action['target_node_1'];v=action['target_node_2']
    if (kind not in BudgetRollout.COST or action['cost']!=BudgetRollout.COST[kind]
        or slots<=0 or action['cost']>budget or u not in state[0] or u not in eligible
        or (kind=='cut' and (v not in eligible or v not in state[0] or tuple(sorted((u,v))) not in state[1]))):
        raise ValueError('Illegal private forecast action')
    if kind=='comm' and (state[0][u]['comm_left']<=0 or action.get('prompt_id')!=1):
        raise ValueError('Illegal private communication')


def r19_inner_compare(state,budget,slots,eligible,incumbent,challenger,deadline):
    # This is frozen PC2's inner depth-two comparison. The outer world is absent.
    inner_deadline=min(deadline,time.monotonic()+3.);values=[]
    try:
        for index,world in enumerate(pc2_worlds(state)):
            scores={}
            for branch in ((0,1) if index%2==0 else (1,0)):
                first=incumbent if branch==0 else challenger
                scores[branch]=r19_depth_continue(state,budget,slots,eligible,first,world,inner_deadline)['score']
            values.append(scores[1]-scores[0])
    except TimeoutError:
        r19_check(deadline)
        return incumbent,dict(reason='incomplete_keep_original',completed_worlds=len(values))
    gain=sum(values)/3
    return (challenger if gain>=.5 else incumbent),dict(reason='replaced' if gain>=.5 else 'no_margin',completed_worlds=3,predicted_delta=gain)


def r19_next(state,budget,eligible,context,deadline,mode='full_pc2'):
    # Receives only the public belief and resource/window context.
    slots=r19_slots(context)
    first=r19_s44(state,budget,slots,eligible,deadline)
    if first is None or mode=='full_s44':return first,None
    if mode!='full_pc2':raise ValueError(mode)
    tags=(['first'] if context['operation']==44 else [])+['budget'+str(b) for b in (40,20,10) if budget<=b]
    new=[t for t in tags if t not in context['seen']]
    if not new:return first,None
    context['seen'].update(new)
    engine=R19Kernel(*state,SEARCH,slots,eligible=eligible)
    pair=pc2_pair(engine,budget);key=BudgetRollout.key(first)
    other=next((a for a in pair if BudgetRollout.key(a)!=key),None)
    event=dict(operation=context['operation'],budget=budget,tags=new,incumbent=key)
    if other is None or key not in {BudgetRollout.key(a) for a in pair}:
        return first,dict(event,reason='no_second_original_prefix')
    result,evidence=r19_inner_compare(state,budget,slots,eligible,first,other,deadline)
    return result,dict(event,**evidence,chosen=BudgetRollout.key(result))


def r19_continue(initial,budget,eligible,context,first,world,deadline,mode='full_pc2'):
    state=BudgetRollout.clone(initial)
    ctx=dict(context,seen=set(context['seen']))
    if mode=='depth2':return r19_depth_continue(state,budget,r19_slots(ctx),eligible,first,world,deadline)
    if mode not in ('full_s44','full_pc2'):raise ValueError(mode)
    chosen=first;actions=[];events=[]
    engine=R19Kernel(*state,SEARCH,r19_slots(ctx),eligible=eligible)
    while chosen is not None:
        r19_check(deadline);r19_legal(state,chosen,budget,r19_slots(ctx),eligible)
        # Latent response is read ONLY by the simulator after choosing an action.
        state=engine.apply(state,chosen,world,True)
        budget-=chosen['cost'];ctx['operation']+=1;ctx['calls']+=1
        actions.append(BudgetRollout.key(chosen))
        if budget<2 or r19_slots(ctx)<=0:break
        chosen,event=r19_next(state,budget,eligible,ctx,deadline,mode)
        if event is not None:events.append(event)
    r19_check(deadline)
    return dict(score=r18_cut_score(state[0],state[1])[0],actions=actions,budget=budget,
                slots=r19_slots(ctx),operation=ctx['operation'],calls=ctx['calls'],seen=sorted(ctx['seen']),pc2_events=events)


def r19_compare(state,budget,eligible,context,incumbent,challenger,seconds=8.,mode='full_pc2',risk=False):
    if seconds<=0:return incumbent,dict(reason='timeout_keep_original',completed_worlds=0,replaced=False)
    deadline=time.monotonic()+seconds;values=[]
    worlds=pc2_worlds(state)
    try:
        for index,world in enumerate(worlds):
            outcomes={}
            for branch in ((0,1) if index%2==0 else (1,0)):
                first=incumbent if branch==0 else challenger
                outcomes[branch]=r19_continue(state,budget,eligible,context,first,world,deadline,mode)
            values.append(dict(incumbent=outcomes[0]['score'],challenger=outcomes[1]['score'],
                               delta=outcomes[1]['score']-outcomes[0]['score'],branches=outcomes))
        r19_check(deadline)
    except TimeoutError:
        return incumbent,dict(reason='incomplete_keep_original',completed_worlds=len(values),replaced=False)
    mean=sum(v['delta'] for v in values)/3
    veto=risk and any(v['delta']<0. for v in values)
    changed=mean>=.5 and not veto and BudgetRollout.key(challenger)!=BudgetRollout.key(incumbent)
    return (challenger if changed else incumbent),dict(reason='replaced' if changed else 'risk_veto' if veto and mean>=.5 else 'no_margin',
        completed_worlds=3,replaced=changed,predicted_delta=mean,min_delta=min(v['delta'] for v in values),worlds=values,
        posterior_calibrated=False,mode=mode)


def r19_clear_private():
    r19_rows.cache_clear();r18_clear_private()
