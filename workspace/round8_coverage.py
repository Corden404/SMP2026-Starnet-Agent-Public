"""One extra omitted-cut continuation, compared with equal-budget ROI control."""
import time
from round8_common import load_s48

M=load_s48()
EXTRA_SECONDS=2.0
EXTRA_MARGIN=.5

class CoverageRollout(M.RestrictedRollout):
    extra_kind='cut'

    def choose(self,budget,mode='rollout',baseline_action=None):
        plans=super().choose(budget,mode,baseline_action)
        if not plans or plans[0].get('predicted_final') is None:return plans
        actions=self.candidates(self.initial,budget)
        fallback=baseline_action or actions[0]
        # Exactly the two prefixes used by the frozen width=2 implementation.
        used={self.key(fallback)}
        extra=[]
        for kind in ('comm','shield','cut'):
            extra.extend([a for a in actions if a['action']==kind][:1])
        extra.extend(actions[:2])
        for a in extra:
            if self.key(a) not in used:
                used.add(self.key(a));break
        cut=next((a for a in actions if a['action']=='cut'),None)
        if cut is None or self.key(cut) in used:return plans
        action=cut if self.extra_kind=='cut' else next((a for a in actions if self.key(a) not in used),None)
        if action is None:return plans
        nodes,edges,effects,observations=self.initial
        means={i:self.effect(i,d,effects,observations)[0] for i,d in nodes.items()}
        diagnostic=dict(kind=self.extra_kind,action=self.key(action),completed=False)
        try:
            result=self.continuation(self.initial,budget,self.slots,means,False,
                                     prefix=(action,),deadline=time.monotonic()+EXTRA_SECONDS)
        except TimeoutError:
            plans[0]['coverage']=dict(diagnostic,timeout=True);return plans
        if result is None:return plans
        diagnostic.update(completed=True,predicted_delta=result['score']-plans[0]['predicted_final'])
        if diagnostic['predicted_delta']>EXTRA_MARGIN:
            new=dict(first=action,predicted_final=result['score'],uncertainty=0.,baseline=False,
                search_completed=True,search_truncated=False,coverage=diagnostic,
                baseline_final=plans[0]['predicted_final'],evaluated_prefixes=plans[0].get('evaluated_prefixes',2)+1,
                proposed_prefixes=3)
            return [new]+plans
        plans[0]['coverage']=diagnostic
        return plans

class ROICoverageRollout(CoverageRollout):
    extra_kind='roi'

class C48(M.ParticipantSquadModel):
    engine_class=CoverageRollout
    def _learned_candidates(self,budget):
        if self.node_count!=50:return super()._learned_candidates(budget)
        state=self.completed_state()
        slots=min(self.operation_limit-1-self.operation_count,
                  self.llm_counter.safe_limit-self.llm_counter.call_count)
        engine=self.engine_class(*state,M.SEARCH,slots,eligible=self.local_nodes)
        reference=engine.candidates(state,budget)
        if not reference:return reference
        start=time.monotonic();plans=engine.choose(budget,baseline_action=reference[0])
        self.search_records.append(dict(operation=self.operation_count,seconds=time.monotonic()-start,plans=plans))
        return [plans[0]['first']] if plans else reference

class R48(C48):
    engine_class=ROICoverageRollout
