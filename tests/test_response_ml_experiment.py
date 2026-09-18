import sys
from pathlib import Path

import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace"))

from response_ml_experiment import Dataset, Ridge, public_feature_rows, subset


def test_public_features_do_not_require_hidden_response():
    nodes = {
        1: {"w": 10.0, "persona": "和平", "comm_left": 3},
        2: {"w": -5.0, "persona": "暴力", "comm_left": 3},
        3: {"w": 0.0, "persona": "中立", "comm_left": 3},
    }
    names, rows = public_feature_rows(nodes, {(1, 2), (2, 3)})
    assert set(rows) == set(nodes)
    assert all(len(row) == len(names) for row in rows.values())
    assert all(np.isfinite(row).all() for row in rows.values())
    assert not any(name in {"r", "scale", "mode", "pool", "node_id"} for name in names)


def test_feature_rows_are_equivariant_to_node_relabeling():
    nodes = {
        1: {"w": 10.0, "persona": "和平", "comm_left": 3},
        2: {"w": -5.0, "persona": "暴力", "comm_left": 3},
        3: {"w": 0.0, "persona": "中立", "comm_left": 3},
    }
    _, original = public_feature_rows(nodes, {(1, 2), (2, 3)})
    mapping = {1: 30, 2: 10, 3: 20}
    relabeled_nodes = {mapping[i]: value for i, value in nodes.items()}
    relabeled_edges = {tuple(sorted((mapping[u], mapping[v]))) for u, v in {(1, 2), (2, 3)}}
    _, relabeled = public_feature_rows(relabeled_nodes, relabeled_edges)
    for old, new in mapping.items():
        np.testing.assert_allclose(original[old], relabeled[new], atol=1e-12)


def test_ridge_learns_visible_linear_signal_with_group_subset():
    x = np.asarray([[i, i % 2] for i in range(20)], dtype=float)
    y = 2.0 * x[:, 0] - 3.0 * x[:, 1] + 4.0
    data = Dataset(
        x=x,
        y=y,
        cases=np.asarray([f"case-{i // 2}" for i in range(20)]),
        nodes=np.arange(20),
        personas=np.asarray(["中立"] * 20),
        families=np.asarray(["test"] * 20),
        feature_names=["x", "flag"],
    )
    model = Ridge(alpha=1e-8).fit(subset(data, np.ones(20, dtype=bool)))
    np.testing.assert_allclose(model.predict(x), y, atol=1e-6)
