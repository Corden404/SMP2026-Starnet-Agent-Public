import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace"))

from response_online_v3_run import PersonaCalibrator, specs  # noqa: E402


def test_fresh_pools_are_disjoint_and_balanced():
    rows = specs()
    assert len({row["case"] for row in rows}) == len(rows)
    assert {row["pool"] for row in rows} == {"robust_train", "tune", "confirm"}
    assert {row["mode"] for row in rows if row["pool"] == "confirm"} == {
        "independent", "degree_pos", "degree_neg", "weight_pos",
        "weight_neg", "persona_scale", "beta", "no_global"}


def test_persona_calibrator_prefers_same_persona_evidence():
    model = PersonaCalibrator()
    low = model.predict_one("和平", 0.0, 0, 50, [("和平", 8.0), ("暴力", 20.0)])
    high = model.predict_one("和平", 0.0, 0, 50, [("和平", 32.0), ("暴力", 2.0)])
    assert high > low
