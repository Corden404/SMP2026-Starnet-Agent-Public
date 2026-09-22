"""Budget-aware, two-node information purchase on exact S44, not combined with F1.

Only initial PUBLIC scan attributes form a two-bin empirical scenario model.
It is a deliberately small heuristic, not a calibrated posterior. Missing edges
retain the S44 assumption. Every scenario includes the full 1.0 scan cost and
lost operation/LLM slots. No partially evaluated expectation can trigger a scan.
"""
import copy as _as_copy
from collections import Counter as _ASCounter

_ActiveBase = ParticipantSquadModel


class ActiveScan(_ActiveBase):
    active_seconds = 8.0
    active_margin = 0.5
    active_windows = (40.0, 20.0)

    def __init__(self, host_env, person_list, llm):
        super().__init__(host_env, person_list, llm)
        self._active_init()

    def _active_init(self):
        self.active_records = []
        self._active_prior = None
        self._active_seen_windows = set()
        self._active_queue = []
        self._active_added = 0
        self._active_reference = None

    def _should_scan(self, budget):
        if not hasattr(self, '_active_prior'):
            self._active_init()
        result = super()._should_scan(budget)
        if not result and self.node_count == 50 and self.scan_stop_reached and self._active_prior is None:
            self._active_prior = [dict(d) for _, d in sorted(self.local_nodes.items())]
        return result

    def _active_support(self):
        population = sorted(self._active_prior or [], key=lambda d: (d['w'], d['persona']))
        if len(population) < 2:
            return []
        middle = len(population) // 2
        result = []
        for group in (population[:middle], population[middle:]):
            result.append(dict(probability=len(group) / len(population),
                w=sum(d['w'] for d in group) / len(group),
                persona=_ASCounter(d['persona'] for d in group).most_common(1)[0][0], comm_left=3))
        return result

    def _active_pair(self, state, action):
        nodes, edges = state[:2]
        unknown = sorted(set(nodes) - set(self.local_nodes))
        if len(unknown) < 2 or action['action'] != 'shield':
            return []
        target = action['target_node_1']
        before = score(nodes, edges)[1]
        after = score({i: d for i, d in nodes.items() if i != target},
                      {e for e in edges if target not in e})[1]
        ordered = sorted(unknown, key=lambda i: (-abs(after.get(i, 0.) - before[i]), i))
        if max(abs(after.get(i, 0.) - before[i]) for i in ordered) < 1e-9:
            return []
        return ordered[:2]

    def _active_value(self, state, budget, slots, eligible, deadline, first=None):
        if time.monotonic() >= deadline:
            raise TimeoutError('Active-scan deadline')
        options = dict(SEARCH, rollout_seconds=min(SEARCH['rollout_seconds'], max(0., deadline - time.monotonic())))
        engine = RestrictedRollout(*state, options, slots, eligible=eligible)
        if first is not None:
            means = {i: engine.effect(i, d, state[2], state[3])[0] for i, d in state[0].items()}
            plan = engine.continuation(engine.initial, budget, slots, means, False, (first,), deadline)
            if plan is None:
                raise TimeoutError('Incomplete incumbent')
            return plan['score']
        candidates = engine.candidates(engine.initial, budget)
        if not candidates:
            return score(state[0], state[1])[0]
        plans = engine.choose(budget, baseline_action=candidates[0])
        if (not plans or not plans[0].get('search_completed') or plans[0].get('search_truncated')
                or plans[0].get('predicted_final') is None):
            raise TimeoutError('Incomplete posterior scenario')
        return plans[0]['predicted_final']

    def _active_evaluate(self, budget, action):
        state = self.completed_state()
        pair = self._active_pair(state, action)
        support = self._active_support()
        if not pair or not support:
            return None, {'reason': 'no_sensitive_pair'}
        slots_before = min(self.operation_limit - 1 - self.operation_count,
                           self.llm_counter.safe_limit - self.llm_counter.call_count)
        slots_after = min(self.operation_limit - 1 - self.operation_count - 2,
                          self.llm_counter.safe_limit - self.llm_counter.call_count - 1)
        if budget < 3.0 or slots_after < 1:
            return None, {'reason': 'insufficient_resources'}
        deadline = time.monotonic() + self.active_seconds
        evidence = {'pair': pair, 'budget_before': budget, 'budget_after_scans': budget - 1.,
                    'scenario_count': 4, 'slots_before': slots_before, 'slots_after': slots_after}
        try:
            baseline = self._active_value(state, budget, slots_before, set(self.local_nodes), deadline, action)
            expected = 0.
            values = []
            for left in support:
                for right in support:
                    hypothetical = _as_copy.deepcopy(state)
                    for i, observed in zip(pair, (left, right)):
                        hypothetical[0][i] = {k: observed[k] for k in ('w', 'persona', 'comm_left')}
                    value = self._active_value(hypothetical, budget - 1., slots_after,
                                               set(self.local_nodes) | set(pair), deadline)
                    probability = left['probability'] * right['probability']
                    expected += probability * value
                    values.append({'probability': probability, 'value': value})
            if time.monotonic() >= deadline:
                raise TimeoutError('Active-scan total deadline')
        except TimeoutError:
            evidence.update(reason='incomplete_no_scan')
            return None, evidence
        gain = expected - baseline
        evidence.update(reason='purchase' if gain > self.active_margin else 'cost_not_recovered',
                        baseline_final=baseline, expected_after_scans=expected, expected_gain=gain,
                        scenarios=values)
        return (pair if gain > self.active_margin else None), evidence

    def _active_scan_step(self, budget):
        target = self._active_queue.pop(0)
        status = self._execute_planned_action(self._action_dict('scan', target), budget, None)
        success = bool(self.action_history and self.action_history[-1]['success'])
        if success:
            self._active_added += 1
        else:
            self._active_queue = []
        self.active_records.append(dict(event='extra_scan', operation=self.operation_count,
                                        node=target, success=success, additional_scans=self._active_added))
        return status

    def _learned_candidates(self, budget):
        cached = getattr(self, '_active_reference', None)
        if cached is not None and cached[0] == self.operation_count and cached[1] == budget:
            self._active_reference = None
            return cached[2]
        return _ActiveBase._learned_candidates(self, budget)

    def _planned_step(self):
        if not hasattr(self, '_active_prior'):
            self._active_init()
        if self.node_count != 50 or self.stopped or not self.scan_stop_reached:
            return super()._planned_step()
        budget = self._budget()
        if budget is None:
            return self._stop('Environment budget unavailable')
        if self.operation_count >= self.operation_limit - 1 or self.llm_counter.call_count >= self.llm_counter.safe_limit:
            return self._stop('Resource safety limit')
        if self._active_queue:
            return self._active_scan_step(budget)
        window = next((w for w in self.active_windows if budget <= w and w not in self._active_seen_windows), None)
        if self._active_added or window is None:
            return super()._planned_step()
        reference = _ActiveBase._learned_candidates(self, budget)
        self._active_reference = (self.operation_count, budget, reference)
        try:
            if reference and reference[0]['action'] == 'shield' and budget >= 7.0:
                self._active_seen_windows.add(window)
                pair, evidence = self._active_evaluate(budget, reference[0])
                self.active_records.append(dict(event='evaluate', operation=self.operation_count,
                                                window=window, **evidence))
                if pair:
                    decision = self.scout._run_chain('batch', {'budget': budget, 'node_count': self.node_count,
                                                             'unknown_ids': pair, 'frontier': pair})
                    if self.llm_counter.call_count >= self.llm_counter.safe_limit:
                        return self._stop('Active-scan LLM safety limit')
                    returned = decision.get('targets', []) if isinstance(decision, dict) else []
                    order = list(dict.fromkeys(i for i in returned if type(i) is int and i in pair)) if isinstance(returned, list) else []
                    # Purchase exactly two scans; the real scout owns their order.
                    self._active_queue = order + [i for i in pair if i not in order]
                    return self._active_scan_step(budget)
            return super()._planned_step()
        finally:
            self._active_reference = None


class ParticipantSquadModel(ActiveScan):
    pass
