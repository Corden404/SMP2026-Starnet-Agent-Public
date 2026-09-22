"""Concatenated after the byte-identical S44 module; public-state-only policy.

R0 exchanges a planned shield/first-communication pair and commits to the shield.
F1 may replace that shield only after observing the communication, with a fully
completed same-belief continuation comparison against the original shield.
No environment access occurs in any hypothetical branch. Other sizes use S44.
"""
import copy as _ff_copy
from types import SimpleNamespace as _FFNamespace

_FeedbackBase = ParticipantSquadModel


class FeedbackFirst(_FeedbackBase):
    feedback_mode = 'F1'
    feedback_seconds = 3.0
    feedback_margin = 0.5

    def __init__(self, host_env, person_list, llm):
        super().__init__(host_env, person_list, llm)
        self._feedback_init()

    def _feedback_init(self):
        self.feedback_records = []
        self._feedback_pending = None

    def _feedback_signature(self, budget):
        return (float(budget), self.operation_count, self.llm_counter.call_count,
                tuple((i, d['w'], d['persona'], d['comm_left'])
                      for i, d in sorted(self.local_nodes.items())),
                tuple(sorted(self.local_edges)), tuple(sorted(self.dead_nodes)),
                tuple(sorted(self.comm_effects.items())),
                tuple((p, tuple(sorted(v.items())))
                      for p, v in sorted(self.persona_observations.items())))

    def _feedback_slots(self):
        return min(self.operation_limit - 1 - self.operation_count,
                   self.llm_counter.safe_limit - self.llm_counter.call_count)

    def _feedback_state(self):
        return self.completed_state()

    def _feedback_shadow_next(self, shield, budget):
        """Ask exact S44 for its next numerical decision after a public shield.

        __init__/LLM/env are never invoked. State and mutable diagnostic lists
        are isolated; numerical coefficient cache remains the S44 cache.
        """
        shadow = _ff_copy.copy(self)
        shadow.env = None
        target = shield['target_node_1']
        shadow.local_nodes = {i: dict(d) for i, d in self.local_nodes.items() if i != target}
        shadow.local_edges = {e for e in self.local_edges if target not in e}
        shadow.dead_nodes = set(self.dead_nodes) | {target}
        shadow.comm_effects = dict(self.comm_effects)
        shadow.persona_observations = _ff_copy.deepcopy(self.persona_observations)
        shadow.planner_options = dict(self.planner_options)
        shadow.operation_count = self.operation_count + 1
        shadow.llm_counter = _FFNamespace(call_count=self.llm_counter.call_count + 1,
                                         safe_limit=self.llm_counter.safe_limit)
        shadow.search_records = []
        shadow.scan_diagnostic_records = []
        shadow.action_history = _ff_copy.deepcopy(self.action_history)
        shadow.decision_history = []
        shadow.scan_queue = list(self.scan_queue)
        shadow.feedback_records = []
        shadow._feedback_pending = None
        choices = _FeedbackBase._learned_candidates(shadow, budget - 5.0)
        last = shadow.search_records[-1]['plans'][0] if shadow.search_records and shadow.search_records[-1]['plans'] else {}
        if not last.get('search_completed') or last.get('search_truncated'):
            return None, 'shadow_search_incomplete'
        if not choices:
            return None, 'no_shadow_action'
        action = choices[0]
        node = action['target_node_1']
        if (action['action'] != 'comm' or node == target or node not in self.local_nodes
                or node in self.comm_effects or self.local_nodes[node]['comm_left'] != 3):
            return None, 'not_a_distinct_first_communication'
        return dict(action), 'eligible'

    def _feedback_original_action(self, pending, state):
        action = dict(pending['shield'])
        node = action['target_node_1']
        nodes, edges = state[:2]
        remaining = {i: d for i, d in nodes.items() if i != node}
        action['gain'] = score(remaining, {e for e in edges if node not in e})[0] - score(nodes, edges)[0]
        action['roi'] = action['gain'] / 5.0
        return action

    def _feedback_compare(self, state, budget, original, alternative):
        """Compare two complete continuations; timeout never accepts half a plan."""
        engine = RestrictedRollout(*state, SEARCH, self._feedback_slots(), eligible=self.local_nodes)
        means = {i: engine.effect(i, d, state[2], state[3])[0] for i, d in state[0].items()}
        deadline = time.monotonic() + self.feedback_seconds
        try:
            incumbent = engine.continuation(engine.initial, budget, engine.slots, means,
                                            False, (original,), deadline)
            challenger = engine.continuation(engine.initial, budget, engine.slots, means,
                                             False, (alternative,), deadline)
        except TimeoutError:
            return original, {'comparison': 'timeout_keep_original'}
        if incumbent is None or challenger is None:
            return original, {'comparison': 'incomplete_keep_original'}
        delta = challenger['score'] - incumbent['score']
        replace = delta >= self.feedback_margin
        return (alternative if replace else original), {
            'comparison': 'complete', 'incumbent_final': incumbent['score'],
            'challenger_final': challenger['score'], 'predicted_delta': delta,
            'replaced': replace}

    def _feedback_resolve(self, budget, pending):
        valid = (pending.get('ready') and budget >= 5.0 and self._feedback_slots() >= 1
                 and pending['shield']['target_node_1'] in self.local_nodes
                 and pending.get('signature') == self._feedback_signature(budget))
        if not valid:
            self.feedback_records.append({'event': 'pending_invalid', 'operation': self.operation_count})
            return _FeedbackBase._learned_candidates(self, budget)
        state = self._feedback_state()
        original = self._feedback_original_action(pending, state)
        event = {'event': 'resolve', 'operation': self.operation_count,
                 'shield_node': original['target_node_1'], 'probe_node': pending['probe_node'],
                 'observed_first_effect': self.comm_effects.get(pending['probe_node']),
                 'mode': self.feedback_mode}
        chosen = original
        if self.feedback_mode == 'R0':
            event['comparison'] = 'fixed_reorder_control'
        else:
            choices = _FeedbackBase._learned_candidates(self, budget)
            if choices and BudgetRollout.key(choices[0]) != BudgetRollout.key(original):
                chosen, comparison = self._feedback_compare(state, budget, original, choices[0])
                event.update(comparison)
            else:
                event['comparison'] = 'same_action_or_no_alternative'
        event['chosen'] = BudgetRollout.key(chosen)
        self.feedback_records.append(event)
        return [chosen]

    def _learned_candidates(self, budget):
        if not hasattr(self, '_feedback_pending'):
            self._feedback_init()
        if self.node_count != 50:
            return _FeedbackBase._learned_candidates(self, budget)
        pending, self._feedback_pending = self._feedback_pending, None
        if pending is not None:
            return self._feedback_resolve(budget, pending)
        choices = _FeedbackBase._learned_candidates(self, budget)
        if (not choices or choices[0]['action'] != 'shield' or budget < 7.0
                or self._feedback_slots() < 2):
            return choices
        shield = dict(choices[0])
        probe, reason = self._feedback_shadow_next(shield, budget)
        if probe is None:
            self.feedback_records.append({'event': 'abstain', 'operation': self.operation_count,
                                          'reason': reason})
            return choices
        # Its displayed gain must refer to the current, not post-shield, topology.
        state = self._feedback_state()
        _, influence = score(state[0], state[1])
        mean = BudgetRollout.effect(probe['target_node_1'], state[0][probe['target_node_1']],
                                    state[2], state[3])[0]
        probe['gain'] = influence[probe['target_node_1']] * mean
        probe['roi'] = probe['gain'] / 2.0
        if probe['gain'] <= 0.25:
            return choices
        self._feedback_pending = {'shield': shield, 'probe_node': probe['target_node_1'],
                                  'operation': self.operation_count, 'ready': False}
        self.feedback_records.append({'event': 'proposed', 'operation': self.operation_count,
                                      'shield_node': shield['target_node_1'],
                                      'probe_node': probe['target_node_1'], 'budget': budget})
        return [probe]

    def _execute_planned_action(self, action, budget, before):
        result = super()._execute_planned_action(action, budget, before)
        pending = getattr(self, '_feedback_pending', None)
        if pending is not None and not pending.get('ready'):
            success = (action['action'] == 'comm' and action['target_node_1'] == pending['probe_node']
                       and self.action_history and self.action_history[-1]['success']
                       and self.operation_count == pending['operation'] + 1)
            if success:
                pending['ready'] = True
                pending['signature'] = self._feedback_signature(budget - 2.0)
            else:
                self._feedback_pending = None
                self.feedback_records.append({'event': 'probe_failed', 'operation': self.operation_count})
        return result
