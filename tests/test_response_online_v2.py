import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace"))

from response_online_v2 import (  # noqa: E402
    ContextRidge,
    GlobalCalibrator,
    Matrix,
    feature_vector,
    generate_seed,
)


def test_seed_generation_is_deterministic_and_complete():
    spec = {"case": "unit-0000", "pool": "unit", "index": 0, "family": "ba", "mode": "independent", "n": 50, "generator_version": 2}
    first, _ = generate_seed(spec)
    second, _ = generate_seed(spec)
    assert first == second
    assert len(first["nodes"]) == 50
    assert {row["id"] for row in first["nodes"]} == set(range(1, 51))
    assert all("r" in row for row in first["nodes"])


def test_context_model_uses_public_observations():
    low = feature_vector("和平", 10.0, 4, 50, [("和平", 8.0)])
    high = feature_vector("和平", 10.0, 4, 50, [("和平", 32.0)])
    matrix = Matrix(np.vstack([low, high]), np.asarray([8.0, 32.0]), np.asarray(["a", "b"]))
    model = ContextRidge(1.0).fit(matrix)
    assert model.predict_one("和平", 10.0, 4, 50, [("和平", 32.0)]) > model.predict_one("和平", 10.0, 4, 50, [("和平", 8.0)])


def test_nonml_calibrator_moves_with_receipts():
    model = GlobalCalibrator(2.0)
    weak = model.predict_one("中立", 0.0, 0, 50, [("和平", 5.0), ("中立", 5.0)])
    strong = model.predict_one("中立", 0.0, 0, 50, [("和平", 35.0), ("中立", 30.0)])
    assert strong > weak
