import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace"))

from action_surrogate_experiment import (  # noqa: E402
    FEATURE_NAMES,
    action_key,
    candidate_universe,
    control_indices,
    feature_rows,
    ml_indices,
    public_state,
    rollout_values,
)


def sample_state():
    nodes = {
        1: {"w": -30.0, "persona": "暴力", "comm_left": 3},
        2: {"w": 10.0, "persona": "和平", "comm_left": 3},
        3: {"w": 2.0, "persona": "中立", "comm_left": 2},
        4: {"w": 8.0, "persona": "和平", "comm_left": 3},
    }
    return nodes, {(1, 2), (2, 3), (3, 4)}, {3: 16.0}, {"中立": {3: 16.0}}


def test_public_state_rejects_hidden_response():
    row = {
        "nodes": {"1": {"w": 1.0, "persona": "和平", "comm_left": 3, "r": 1.0}},
        "edges": [], "effects": {}, "observations": {},
    }
    try:
        public_state(row)
    except AssertionError as exc:
        assert "Hidden response" in str(exc)
    else:
        raise AssertionError("hidden response field was accepted")


def test_features_and_values_are_finite_and_aligned():
    state = sample_state()
    engine, candidates = candidate_universe(state, 20.0, 8)
    features = feature_rows(state, 20.0, 8, candidates)
    values = rollout_values(engine, state, 20.0, 8, candidates)
    assert len(candidates) == len(features) == len(values)
    assert len(candidates) >= 2
    assert all(len(row) == len(FEATURE_NAMES) for row in features)
    assert all(math.isfinite(value) for row in features for value in row)
    assert all(math.isfinite(value) for value in values)


def test_fixed_k_selectors_always_keep_fallback():
    row = {
        "actions": [
            {"key": ("comm", 1, None), "gain": 5.0, "roi": 2.5},
            {"key": ("cut", 1, 2), "gain": 6.0, "roi": 2.0},
            {"key": ("shield", 1, None), "gain": 8.0, "roi": 1.6},
            {"key": ("comm", 2, None), "gain": 4.0, "roi": 2.0},
        ]
    }
    prediction = np.asarray([-99.0, 1.0, 3.0, 2.0])
    for selected in (ml_indices(prediction, 3), control_indices(row, 3)):
        assert selected[0] == 0
        assert len(selected) == 3
        assert len(set(selected)) == 3


def test_action_key_ignores_nonidentity_fields():
    first = {"action": "comm", "target_node_1": 2, "target_node_2": None, "gain": 3.0}
    second = {**first, "gain": 999.0, "roi": 12.0}
    assert action_key(first) == action_key(second)
