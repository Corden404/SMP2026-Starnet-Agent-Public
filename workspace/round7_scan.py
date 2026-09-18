"""Late scan stop with a persistent phase and explicit unobserved vertices.

Public incident edges are retained. Unseen attributes use the scanned mean at
the stop boundary; unseen-unseen edges are assumed absent (diagnostic limitation).
Unscanned vertices are never eligible intervention targets.
"""
from __future__ import annotations
import time
from round6_common import load_d5_module

D5=load_d5_module()

class RestrictedRollout(D5.FiniteRollout):
    def __init__(self,*args,eligible,**kwargs):
        super().__init__(*args,**kwargs);self.eligible=set(eligible)

    def candidates(self,state,budget,negative=False):
        return [a for a in super().candidates(state,budget,negative)
                if a['target_node_1'] in self.eligible and
                (a['action']!='cut' or a['target_node_2'] in self.eligible)]

class LateScanStop(D5.ParticipantSquadModel):
    scan_target=50
    def __init__(self,host_env,person_list,llm):
        super().__init__(host_env,person_list,llm)
        self.scan_stop_reached=False;self.scan_diagnostic_records=[]

    def _next_scan_target(self):
        # This sentinel marks the intervention phase, not graph completeness.
        # Partial states are explicitly completed inside _learned_candidates.
        if self.scan_stop_reached:return None
        return super()._next_scan_target()

    def _should_scan(self,budget):
        if self.scan_stop_reached:return False
        if len(self.local_nodes)+len(self.dead_nodes)>=self.scan_target:
            self.scan_stop_reached=True;self.scan_closed=True
            self.unseen_w=sum(d['w'] for d in self.local_nodes.values())/max(1,len(self.local_nodes))
            self.scan_diagnostic_records.append(dict(event='stop',scanned=self._known_count(),
                budget=budget,unseen_w=self.unseen_w))
            return False
        return super()._should_scan(budget)

    def completed_state(self):
        nodes={i:dict(d) for i,d in self.local_nodes.items()}
        unknown=set(range(1,self.node_count+1))-set(nodes)-self.dead_nodes
        for i in unknown:nodes[i]=dict(w=self.unseen_w,persona='unobserved',comm_left=0)
        edges={e for e in self.local_edges if all(i in nodes for i in e)}
        return nodes,edges,dict(self.comm_effects),{p:dict(v) for p,v in self.persona_observations.items()}

    def _learned_candidates(self,budget):
        unknown=set(range(1,self.node_count+1))-set(self.local_nodes)-self.dead_nodes
        if not unknown:return super()._learned_candidates(budget)
        state=self.completed_state()
        slots=min(self.operation_limit-1-self.operation_count,
                  self.llm_counter.safe_limit-self.llm_counter.call_count)
        engine=RestrictedRollout(*state,D5.SEARCH,slots,eligible=self.local_nodes)
        reference=engine.candidates(engine.initial,budget)
        if not reference:return reference
        start=time.monotonic();plans=engine.choose(budget,baseline_action=reference[0])
        self.search_records.append(dict(operation=self.operation_count,
            seconds=time.monotonic()-start,plans=plans))
        self.scan_diagnostic_records.append(dict(event='intervention',operation=self.operation_count,
            known_alive=len(self.local_nodes),dead=len(self.dead_nodes),unseen=len(unknown),
            graph_vertices=len(state[0]),structural_candidates=sum(a['action']!='comm' for a in reference)))
        return [plans[0]['first']] if plans else reference

class S48(LateScanStop):scan_target=48
class S49(LateScanStop):scan_target=49
class S50(LateScanStop):scan_target=50
CLASSES={'S48':S48,'S49':S49,'S50':S50}
