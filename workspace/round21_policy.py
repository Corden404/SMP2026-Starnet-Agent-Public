"""Round21 conservative Tail Bundle selector appended after exact PC2 + private numeric helpers."""
_R21Base = _R18CutBase

def r21_top_structural(state,budget,eligible,kind,n=2):
    engine=R19Kernel(*state,SEARCH,999,eligible=eligible)
    xs=[a for a in engine.candidates(state,budget) if a['action']==kind and a['gain']>.25]
    xs.sort(key=lambda a:(-a['gain'],-a['roi'],a['target_node_1'],a['target_node_2'] or -1))
    return xs[:n]

def r21_structural_apply(state,action,eligible):
    engine=R19Kernel(*state,SEARCH,999,eligible=eligible)
    return engine.apply(state,action,{},False)

def r21_bundles(state,budget,eligible):
    cuts=r21_top_structural(state,budget,eligible,'cut',2)
    shields=r21_top_structural(state,budget,eligible,'shield',2)
    plans=[]
    if cuts:plans.append(('C1',[cuts[0]]))
    if shields:plans.append(('S1',[shields[0]]))
    if len(shields)>1:plans.append(('S2',[shields[1]]))
    if cuts:
        first=cuts[0];post=r21_structural_apply(state,first,eligible);remain=budget-first['cost']
        nxt=r21_top_structural(post,remain,eligible,'cut',1)
        if nxt:plans.append(('C1-C',[first,nxt[0]]))
        nxt=r21_top_structural(post,remain,eligible,'shield',1)
        if nxt:plans.append(('C1-S',[first,nxt[0]]))
    if shields:
        first=shields[0];post=r21_structural_apply(state,first,eligible);remain=budget-first['cost']
        nxt=r21_top_structural(post,remain,eligible,'cut',1)
        if nxt:plans.append(('S1-C',[first,nxt[0]]))
    out=[];seen=set()
    for label,prefix in plans:
        key=tuple(BudgetRollout.key(a) for a in prefix)
        if key not in seen and sum(a['cost'] for a in prefix)<=budget:
            seen.add(key);out.append((label,prefix))
    if len(out)>6:raise RuntimeError('Round21 bundle cap')
    return out

def r21_prefix_continue(state,budget,eligible,context,prefix,world,deadline):
    st=BudgetRollout.clone(state);ctx=dict(context,seen=set(context['seen']))
    engine=R19Kernel(*st,SEARCH,r19_slots(ctx),eligible=eligible);actions=[]
    for a in prefix:
        r19_check(deadline);r19_legal(st,a,budget,r19_slots(ctx),eligible)
        st=engine.apply(st,a,world,True);budget-=a['cost'];ctx['operation']+=1;ctx['calls']+=1
        actions.append(BudgetRollout.key(a))
    if budget<2 or r19_slots(ctx)<=0:
        return dict(score=r18_cut_score(st[0],st[1])[0],actions=actions,budget=budget)
    first=r19_s44(st,budget,r19_slots(ctx),eligible,deadline,strict=False)
    if first is None:return dict(score=r18_cut_score(st[0],st[1])[0],actions=actions,budget=budget)
    suffix=r19_continue(st,budget,eligible,ctx,first,world,deadline,mode='full_s44')
    suffix['actions']=actions+suffix['actions'];return suffix

def r21_select(state,budget,eligible,context,incumbent,seconds=12.,margin=.5):
    if seconds<=0:return None,dict(reason='timeout',all_complete=False,selected=None)
    deadline=time.monotonic()+seconds;worlds=pc2_worlds(state);plans=r21_bundles(state,budget,eligible)
    try:
        base=[r19_continue(state,budget,eligible,context,incumbent,w,deadline,mode='full_s44')['score'] for w in worlds]
    except TimeoutError:return None,dict(reason='baseline_timeout',all_complete=False,selected=None)
    rows=[]
    for label,prefix in plans:
        vals=[]
        try:
            for w,b in zip(worlds,base):
                got=r21_prefix_continue(state,budget,eligible,context,prefix,w,deadline)
                vals.append(dict(score=got['score'],delta=got['score']-b,actions=got['actions']))
        except TimeoutError:
            return None,dict(reason='candidate_timeout',all_complete=False,selected=None,plans=rows)
        mean=sum(x['delta'] for x in vals)/3
        rows.append(dict(label=label,prefix=[BudgetRollout.key(a) for a in prefix],mean_delta=mean,min_delta=min(x['delta'] for x in vals),worlds=vals,_actions=prefix))
    best=max(rows,key=lambda x:x['mean_delta'],default=None)
    if best is None or best['mean_delta']<margin:
        clean=[{k:v for k,v in x.items() if k!='_actions'} for x in rows]
        return None,dict(reason='no_margin',all_complete=True,selected=None,plans=clean)
    actions=best['_actions'];clean=[{k:v for k,v in x.items() if k!='_actions'} for x in rows]
    selected={k:v for k,v in best.items() if k!='_actions'}
    return actions,dict(reason='selected',all_complete=True,selected=selected,plans=clean)

def r21_clear_private():
    r19_clear_private()

class ParticipantSquadModel(_R21Base):
    r21_margin=.5
    r21_seconds=12.
    def __init__(self,host_env,person_list,llm):
        super().__init__(host_env,person_list,llm);self.r21_records=[];self._r21_seen=set();self._r21_pending=None
    def _r21_init(self):
        if not hasattr(self,'r21_records'):self.r21_records=[];self._r21_seen=set();self._r21_pending=None
    def _learned_candidates(self,budget):
        self._r21_init()
        original=super()._learned_candidates(budget)
        if self.node_count!=50 or not original:return original
        if self._r21_pending is not None:
            pending=self._r21_pending;self._r21_pending=None
            state=self.completed_state();eligible=set(self.local_nodes)
            slots=min(self.operation_limit-1-self.operation_count,self.llm_counter.safe_limit-self.llm_counter.call_count)
            try:r19_legal(state,pending,budget,slots,eligible)
            except ValueError:
                self.r21_records.append(dict(kind='pending',operation=self.operation_count,budget=budget,reason='pending_illegal_fallback',action=BudgetRollout.key(pending)))
                return original
            self.r21_records.append(dict(kind='pending',operation=self.operation_count,budget=budget,reason='pending_execute',action=BudgetRollout.key(pending)))
            return [pending]
        tags=['budget'+str(b) for b in (20,10) if budget<=b and 'budget'+str(b) not in self._r21_seen]
        if not tags:return original
        self._r21_seen.update(tags);state=self.completed_state();eligible=set(self.local_nodes)
        ctx=dict(operation=self.operation_count,calls=self.llm_counter.call_count,operation_limit=self.operation_limit,
                 safe_limit=self.llm_counter.safe_limit,seen=set(self._pc2_seen))
        started=time.monotonic()
        actions,evidence=r21_select(state,budget,eligible,ctx,original[0],self.r21_seconds,self.r21_margin)
        event=dict(kind='bundle',operation=self.operation_count,budget=budget,tags=tags,incumbent=BudgetRollout.key(original[0]),
                   margin=self.r21_margin,seconds=time.monotonic()-started,**evidence)
        if not actions:
            self.r21_records.append(event);return original
        first=actions[0]
        if len(actions)>1:self._r21_pending=actions[1]
        event.update(chosen=BudgetRollout.key(first),pending=BudgetRollout.key(actions[1]) if len(actions)>1 else None)
        self.r21_records.append(event);return [first]
