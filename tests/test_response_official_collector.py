import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace"))

from response_official_collector import jobs, selected_targets  # noqa: E402
from response_online_v2 import materialize  # noqa: E402


def test_collection_jobs_cover_repeats_and_mechanisms():
    rows = jobs(materialize())
    assert len(rows) == 48
    assert len({row["case"] for row in rows}) == 42
    assert len({row["mode"] for row in rows}) == 7


def test_target_selection_is_deterministic_and_public():
    scans = {i: {"w": float(i), "persona": "中立", "neighbors": []} for i in range(1, 51)}
    first = selected_targets("test-0000", scans)
    assert first == selected_targets("test-0000", scans)
    assert len(first) == 37
    assert len(set(first)) == 37
