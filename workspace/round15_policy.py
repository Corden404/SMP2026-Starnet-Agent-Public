"""Paired-world scan value on exact S44 + V2 scheduling, no F1 combination.

The SAME public-state decision rule is re-run after each simulated receipt on
both branches. Latent responses are fixed within a world and never passed to
candidate selection. Four empirical joint-attribute worlds are an approximation,
not a calibrated posterior. Missing unknown-unknown edges still assumed absent.
"""
import copy as _cv_copy
import hashlib as _cv_hashlib
import random as _cv_random


class CoupledScan(ActiveScan):
    coupled_samples = 4

    def _coupled_worlds(self, state):
        nodes, edges, effects, observations = state
        population = sorted(self._active_prior or [], key=lambda d: (d['w'], d['persona']))
        if not population:
            return []
        unknown = sorted(set(nodes) - set(self.local_nodes))
        signature = repr(tuple((i, d['w'], d['persona'], d['comm_left']) for i, d in sorted(nodes.items())))
        seed = int(_cv_hashlib.sha256(signature.encode()).hexdigest()[:16], 16)
        rng = _cv_random.Random(seed)
        offsets = {i:rng.randrange(len(population)) for i in unknown}
        gaussian = [{i:rng.gauss(0.,1.) for i in sorted(nodes)} for _ in range(2)]
        worlds = []
        for k in range(self.coupled_samples):
            physical = {i:dict(d) for i,d in nodes.items()}
            for i in unknown:
                index = (offsets[i] + (k * len(population)) // self.coupled_samples) % len(population)
                donor = population[index]
                physical[i] = dict(w=donor['w'], persona=donor['persona'], comm_left=3)
            latent = {}
            for i,d in physical.items():
                mean,sd = BudgetRollout.effect(i,d,effects,observations)
                latent[i] = effects[i] if i in effects else max(0., mean + (-1 if k%2==0 else 1)*gaussian[k//2][i]*sd)
            worlds.append(dict(nodes=physical,edges=set(edges),responses=latent))
        return worlds

    def _coupled_next(self, belief, budget, slots, eligible, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError('Coupled value deadline')
        options = dict(SEARCH, rollout_seconds=min(SEARCH['rollout_seconds'], max(0., deadline-time.monotonic())))
        engine = RestrictedRollout(*belief, options, slots, eligible=eligible)
        candidates = engine.candidates(engine.initial, budget)
        if not candidates:
            return None
        plans = engine.choose(budget, baseline_action=candidates[0])
        if not plans or not plans[0].get('search_completed') or plans[0].get('search_truncated'):
            raise TimeoutError('Incomplete paired-policy decision')
        return dict(plans[0]['first'])

    def _coupled_simulate(self, state, world, pair, reveal, budget, slots, eligible, deadline):
        # World is test/simulation state, built ONLY from public observations.
        # _coupled_next receives belief/eligibility, never this world or responses.
        belief = BudgetRollout.clone(state)
        physical = {i:dict(d) for i,d in world['nodes'].items()}
        physical_edges = set(world['edges'])
        allowed = set(eligible)
        if reveal:
            for i in pair:
                belief[0][i] = dict(physical[i])
                allowed.add(i)
            belief[1].update(e for e in physical_edges if any(i in e for i in pair))
        actions = []
        while budget >= 2. and slots > 0:
            chosen = self._coupled_next(belief,budget,slots,allowed,deadline)
            if chosen is None:
                break
            kind = chosen['action'];u=chosen['target_node_1'];v=chosen['target_node_2']
            if u not in allowed or u not in physical or chosen['cost'] > budget:
                raise ValueError('Illegal coupled action')
            if kind == 'comm':
                d = physical[u]
                delta = world['responses'][u] * (.5 ** (3-d['comm_left']))
                d['w'] += delta;d['comm_left'] -= 1
                # Public simulated receipt delivered only AFTER the chosen action.
                belief[0][u]['w'] = d['w'];belief[0][u]['comm_left'] = d['comm_left']
                belief[2][u] = world['responses'][u]
                belief[3].setdefault(d['persona'],{})[u] = world['responses'][u]
            elif kind == 'shield':
                del physical[u];del belief[0][u]
                physical_edges = {e for e in physical_edges if u not in e}
                belief = (belief[0],{e for e in belief[1] if u not in e},belief[2],belief[3])
            elif kind == 'cut':
                if v not in allowed:
                    raise ValueError('Unknown cut endpoint')
                edge = tuple(sorted((u,v)))
                if edge not in physical_edges or edge not in belief[1]:
                    raise ValueError('Absent simulated edge')
                physical_edges.remove(edge);belief[1].remove(edge)
            else:
                raise ValueError('Not an intervention')
            budget -= chosen['cost'];slots -= 1
            actions.append(BudgetRollout.key(chosen))
        if time.monotonic() >= deadline:
            raise TimeoutError('Incomplete paired-world value')
        return dict(score=score(physical,physical_edges)[0], actions=actions, budget=budget, slots=slots)

    def _coupled_pair_value(self,state,world,pair,budget,before_slots,after_slots,eligible,deadline,cost=1.,reverse=False):
        results={}
        for reveal in ((True,False) if reverse else (False,True)):
            results[reveal]=self._coupled_simulate(state,world,pair,reveal,budget-cost if reveal else budget,
                after_slots if reveal else before_slots,eligible,deadline)
        return dict(delta=results[True]['score']-results[False]['score'],
                    no_scan_score=results[False]['score'],scan_score=results[True]['score'])

    def _active_evaluate(self,budget,action):
        state=self.completed_state();pair=self._active_pair(state,action)
        worlds=self._coupled_worlds(state)
        if not pair or not worlds:
            return None,dict(reason='no_sensitive_pair')
        before=min(self.operation_limit-1-self.operation_count,self.llm_counter.safe_limit-self.llm_counter.call_count)
        after=min(self.operation_limit-1-self.operation_count-2,self.llm_counter.safe_limit-self.llm_counter.call_count-1)
        if budget<3. or after<1:
            return None,dict(reason='insufficient_resources')
        deadline=time.monotonic()+self.active_seconds
        evidence=dict(pair=pair,budget_before=budget,budget_after_scans=budget-1.,
            scenario_count=len(worlds),slots_before=before,slots_after=after,
            estimator='paired-public-worlds-feedback-exact-S44-numerical-policy',
            posterior_calibrated=False,unknown_unknown_edges='assumed_absent')
        values=[]
        try:
            for k,world in enumerate(worlds):
                values.append(self._coupled_pair_value(state,world,pair,budget,before,after,
                    set(self.local_nodes),deadline,reverse=bool(k%2)))
        except TimeoutError:
            evidence.update(reason='incomplete_no_scan',completed_worlds=len(values))
            return None,evidence
        gain=sum(v['delta'] for v in values)/len(values)
        evidence.update(reason='purchase' if gain>self.active_margin else 'cost_not_recovered',
            expected_gain=gain,paired_worlds=values,completed_worlds=len(values))
        return (pair if gain>self.active_margin else None),evidence


class ParticipantSquadModel(CoupledScan):
    pass
