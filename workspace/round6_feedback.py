"""P2 feedback-lookahead variants derived directly from frozen D5."""

from __future__ import annotations

import time

from round6_common import load_d5_module


D5 = load_d5_module()
FeedbackBase = D5.ParticipantSquadModel


VARIANTS = {
    "L1": {"rollout_seconds": 3.0, "rollout_width": 2, "rollout_samples": 1,
           "rollout_learning": True, "rollout_combinations": False, "rollout_margin": 0.5},
    "U3": {"rollout_seconds": 3.0, "rollout_width": 2, "rollout_samples": 3,
           "rollout_learning": False, "rollout_combinations": False, "rollout_margin": 0.5},
    "L3": {"rollout_seconds": 3.0, "rollout_width": 2, "rollout_samples": 3,
           "rollout_learning": True, "rollout_combinations": False, "rollout_margin": 0.5},
}


class FeedbackD5(FeedbackBase):
    feedback_variant = None

    def __init__(self, host_env, person_list, llm):
        super().__init__(host_env, person_list, llm)
        self.feedback_records = []

    def _learned_candidates(self, budget):
        options = VARIANTS[self.feedback_variant]
        # Bypass only D5's final search adapter. Keep its frozen F0/L0 belief,
        # candidate and real-response update path unchanged.
        reference = D5.L0._learned_candidates(self, budget)
        if not reference:
            return reference
        slots = min(self.operation_limit - 1 - self.operation_count,
                    self.llm_counter.safe_limit - self.llm_counter.call_count)
        rollout = D5.FiniteRollout(self.local_nodes, self.local_edges,
                                   self.comm_effects, self.persona_observations,
                                   options, slots)
        started = time.monotonic()
        plans = rollout.choose(budget, baseline_action=reference[0])
        self.search_records.append({'operation': self.operation_count,
                                    'seconds': time.monotonic() - started, 'plans': plans})
        self.feedback_records.append({
            'operation': self.operation_count, 'variant': self.feedback_variant,
            'learning': options['rollout_learning'], 'samples': options['rollout_samples'],
            'evaluations': rollout.evaluations,
            'selected_baseline': bool(plans) and bool(plans[0].get('baseline')),
            'search_completed': bool(plans) and bool(plans[0].get('search_completed')),
            'search_truncated': bool(plans) and bool(plans[0].get('search_truncated')),
        })
        return [plans[0]['first']] if plans else reference


class L1(FeedbackD5):
    feedback_variant = "L1"


class U3(FeedbackD5):
    feedback_variant = "U3"


class L3(FeedbackD5):
    feedback_variant = "L3"


CLASSES = {"L1": L1, "U3": U3, "L3": L3}


def initialize_fast_feedback(model):
    model.search_records = []
    model.feedback_records = []
    return model
