"""Whole remaining-budget local exchanges, using only public fixed-mean beliefs."""
from __future__ import annotations
import itertools
import time
from round6_common import load_d5_module

D5=load_d5_module()
LIMITS=dict(evaluations=96, rounds=2, pool_per_kind=8, seconds=2.0, margin=.5)

def action(kind,u,v=None,gain=0.):
    cost={'comm':2.,'cut':3.,'shield':5.}[kind]
    return dict(action=kind,target_node_1=u,target_node_2=v,
                prompt_id=1 if kind=='comm' else None,cost=cost,gain=gain,roi=gain/cost)

def structural_key(a):
    return (a['action'],a['target_node_1'],a.get('target_node_2') or 0)

def normalize(structures):
    structures=set(structures)
    removed={u for kind,u,v in structures if kind=='shield'}
    return tuple(sorted(s for s in structures if s[0]!='cut' or not ({s[1],s[2]} & removed)))

def allocate(state,structures,budget,slots):
    """Exact communication allocation conditional on topology and fixed means."""
    original,edges,effects,observations=state
    structures=normalize(structures)
    removed={u for kind,u,v in structures if kind=='shield'}
    cuts={tuple(sorted((u,v))) for kind,u,v in structures if kind=='cut'}
    if not removed <= set(original) or not cuts <= edges:return None
    cost=sum(5 if kind=='shield' else 3 for kind,u,v in structures)
    if cost>budget+1e-9 or len(structures)>slots:return None
    nodes={i:d for i,d in original.items() if i not in removed}
    if not nodes:return None
    final_edges={e for e in edges-cuts if not (set(e)&removed)}
    total,influence=D5.score(nodes,final_edges)
    marginals=[]
    for i,d in nodes.items():
        mean=D5.FiniteRollout.effect(i,d,effects,observations)[0]
        for offset in range(d['comm_left']):
            gain=influence[i]*mean*.5**(3-d['comm_left']+offset)
            if gain>0:marginals.append((gain,i,offset))
    marginals.sort(key=lambda item:(-item[0],item[1],item[2]))
    chosen=marginals[:min(int((budget-cost+1e-9)//2),slots-len(structures))]
    actions=[action(kind,u,v if kind=='cut' else None) for kind,u,v in structures]
    actions.extend(action('comm',i,gain=gain) for gain,i,offset in chosen)
    return dict(score=total+sum(g for g,i,k in chosen),actions=actions,
                structures=structures,cost=cost+2*len(chosen))

def optimize(state,budget,slots,baseline_first,limits=None):
    limits=limits or LIMITS
    if limits['seconds']<=0:
        return dict(selected=False,reason='reference_timeout',evaluations=0)
    start=time.monotonic();deadline=start+limits['seconds']
    engine=D5.FiniteRollout(*state,D5.SEARCH,slots)
    means={i:engine.effect(i,d,state[2],state[3])[0] for i,d in state[0].items()}
    try:
        reference=engine.continuation(engine.initial,budget,slots,means,False,
                                      prefix=(baseline_first,),deadline=deadline)
    except TimeoutError:
        return dict(selected=False,reason='reference_timeout',evaluations=0)
    if not reference:return dict(selected=False,reason='reference_infeasible',evaluations=0)
    initial=normalize(structural_key(a) for a in reference['actions'] if a['action']!='comm')
    best=allocate(state,initial,budget,slots)
    if best is None:raise AssertionError('Reference structure must be feasible')
    candidates=engine.candidates(engine.initial,budget,negative=True)
    pool=[]
    for kind in ('shield','cut'):
        pool.extend(structural_key(a) for a in [a for a in candidates if a['action']==kind][:limits['pool_per_kind']])
    evaluated=1;seen={initial};truncated=False
    for _ in range(limits['rounds']):
        incumbent=best['structures']
        proposals=[()]
        proposals.extend(tuple(s for s in incumbent if s!=drop) for drop in incumbent)
        proposals.extend(tuple(s for s in incumbent if s not in pair)
                         for pair in itertools.islice(itertools.combinations(incumbent,2),16))
        proposals.extend(incumbent+(new,) for new in pool if new not in incumbent)
        proposals.extend(tuple(s for s in incumbent if s!=drop)+(new,)
                         for drop in incumbent for new in pool if new not in incumbent)
        for proposal in proposals:
            key=normalize(proposal)
            if key in seen:continue
            if evaluated>=limits['evaluations'] or time.monotonic()>deadline:
                truncated=True;break
            seen.add(key);candidate=allocate(state,key,budget,slots);evaluated+=1
            if candidate and candidate['score']>best['score']+1e-9:best=candidate
        if truncated or best['structures']==incumbent:break
    delta=best['score']-reference['score']
    selected=bool(best['actions']) and delta>limits['margin']
    # Preserve the D5 first action when it is part of the better complete plan.
    # All communication gains commute with structural actions at frozen means.
    first=next((a for a in best['actions'] if D5.FiniteRollout.key(a)==D5.FiniteRollout.key(baseline_first)),None)
    if first is None and best['actions']:
        ranks={D5.FiniteRollout.key(a):i for i,a in enumerate(candidates)}
        first=min(best['actions'],key=lambda a:ranks.get(D5.FiniteRollout.key(a),10**9))
    return dict(selected=selected,reason='improved' if selected else 'no_margin',
        first=first,reference_score=reference['score'],candidate_score=best['score'],
        predicted_delta=delta,evaluations=evaluated,truncated=truncated,
        structures=best['structures'],initial_structures=initial,
        seconds=time.monotonic()-start,changed_first=selected and
        D5.FiniteRollout.key(first)!=D5.FiniteRollout.key(baseline_first))

class X1(D5.ParticipantSquadModel):
    def __init__(self,host_env,person_list,llm):
        super().__init__(host_env,person_list,llm)
        self.exchange_records=[]

    def _learned_candidates(self,budget):
        reference=super()._learned_candidates(budget)
        if not reference or budget<3:return reference
        slots=min(self.operation_limit-1-self.operation_count,
                  self.llm_counter.safe_limit-self.llm_counter.call_count)
        state=(self.local_nodes,self.local_edges,self.comm_effects,self.persona_observations)
        result=optimize(state,budget,slots,reference[0])
        self.exchange_records.append(dict(result,operation=self.operation_count,budget=budget))
        return [result['first']] if result['selected'] else reference
