"""Minimal same-seed official-sandbox replay probe with zero LLM calls."""

import json
import math
import time

from api_client import RemoteStarNetEnv
from mechanism_round5 import SERVER, prediction
from round4_common import ROOT, digest, save


SEED_PATH = ROOT / "workspace/custom_seeds/my_test_network.json"
OUT = ROOT / "agent_notes/experiments/seed-replay-probe-20260918/result.json"


def normalized_scan(value):
    if value is None:
        return None
    result = dict(value)
    if "neighbors" in result:
        result["neighbors"] = sorted(result["neighbors"])
    return result


def run_session(seed, scan_order):
    env = RemoteStarNetEnv(SERVER, seed, timeout=30.0)
    initial_budget = env.get_remaining_budget()
    scans = {}
    for node_id in scan_order:
        scans[str(node_id)] = normalized_scan(env.scan_node(node_id))
    budget_after_scans = env.get_remaining_budget()
    before = scans["1"]["w"]
    response = env.communicate(1, 1)
    if response.get("status") != "success" or not isinstance(response.get("new_w"), (int, float)):
        raise RuntimeError(f"Unexpected communication response: {response!r}")
    budget_after_communication = env.get_remaining_budget()
    result = {
        "session_id": env.session_id,
        "scan_order": list(scan_order),
        "initial_budget": initial_budget,
        "scans": scans,
        "budget_after_scans": budget_after_scans,
        "communication": response,
        "first_effect": response["new_w"] - before,
        "budget_after_communication": budget_after_communication,
    }
    result["final_score"] = env.trigger_eval()
    return result


def main():
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    actions = [("scan_node", [i]) for i in (1, 2, 3, 4)] + [("communicate", [1, 1])]
    local = prediction(seed, actions)
    started = time.time()
    first = run_session(seed, (1, 2, 3, 4))
    second = run_session(seed, (4, 3, 2, 1))
    checks = {
        "different_session_ids": first["session_id"] != second["session_id"],
        "same_scans_after_normalization": first["scans"] == second["scans"],
        "same_initial_budget": first["initial_budget"] == second["initial_budget"],
        "same_budget_after_scans": first["budget_after_scans"] == second["budget_after_scans"],
        "same_communication_response": first["communication"] == second["communication"],
        "same_first_effect": math.isclose(first["first_effect"], second["first_effect"], abs_tol=1e-12),
        "same_budget_after_communication": first["budget_after_communication"] == second["budget_after_communication"],
        "same_final_score": first["final_score"] == second["final_score"],
        "response_matches_seed_prediction": math.isclose(
            first["first_effect"], seed["prompts"]["1"] * seed["nodes"][0]["r"], abs_tol=1e-12
        ),
        "settlement_matches_local_finite50_to_two_decimals": round(local["finite50"], 2)
        == first["final_score"],
    }
    result = {
        "schema": 1,
        "purpose": "Same custom-seed replay and branch isolation; no policy or LLM",
        "server": SERVER,
        "seed_sha256": digest(seed),
        "seed_path": str(SEED_PATH),
        "sessions": [first, second],
        "local_prediction": {
            "first_effect": seed["prompts"]["1"] * seed["nodes"][0]["r"],
            "finite50": local["finite50"],
            "rounded_finite50": round(local["finite50"], 2),
            "remaining_budget": local["budget"],
        },
        "checks": checks,
        "passed": all(checks.values()),
        "glm_http_calls": 0,
        "tokens": 0,
        "elapsed_seconds": time.time() - started,
    }
    save(OUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
