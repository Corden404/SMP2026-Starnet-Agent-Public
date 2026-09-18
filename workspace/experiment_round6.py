"""P0/P1 paired local runs against the exact frozen D5 implementation."""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import time

from benchmark import LoggedEnv
from experiment_round3 import fast_model
from experiment_round5 import FiniteEnv, PublicEnv
from round4_seeds import load_case
from round6_common import OUT, digest, fixed_dev_cases, load_d5_module, people, prepare_manifests, save
from round6_tail import TailD5, initialize_fast_tail


D5 = load_d5_module()


def run_case(case, variant, force=False):
    if variant not in ("D5", "T1"):
        raise ValueError(variant)
    spec, seed = load_case(case)
    if spec["pool"] != "dev":
        raise ValueError("P0/P1 only use the fixed development pool")
    target = OUT / "p0-p1" / f"{case}-{variant}.json"
    identity = {
        "schema": 1,
        "case": case,
        "variant": variant,
        "seed_sha256": digest(seed),
        "baseline": "D5 frozen zip",
        "selector": "deterministic first",
        "purpose": "round6 fixed development",
    }
    if target.exists() and not force:
        old = json.loads(target.read_text(encoding="utf-8"))
        if old["identity"] != identity:
            raise RuntimeError("Cached identity mismatch")
        return old

    raw = FiniteEnv(seed)
    logged = LoggedEnv(raw)
    model_class = D5.ParticipantSquadModel if variant == "D5" else TailD5
    model = fast_model(PublicEnv(logged), copy.deepcopy(people()), model_class)
    model.search_records = []
    if variant == "T1":
        initialize_fast_tail(model)
    states = []
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(seed["global_setting"]["max_api_calls"]):
            states.append(
                {
                    "budget": raw.get_remaining_budget(),
                    "operation_count": model.operation_count,
                    "logical_calls": model.llm_counter.call_count,
                    "known_responses": len(model.comm_effects),
                    "prefix_length": len(logged.actions),
                }
            )
            if model.step():
                break
        else:
            raise RuntimeError("Resource fuse reached")
    result = {
        "identity": identity,
        "case": case,
        "variant": variant,
        "spec": spec,
        "final_score": raw.evaluate(),
        "elapsed_seconds": time.monotonic() - started,
        "remaining_budget": raw.get_remaining_budget(),
        "stop_reason": model.stop_reason,
        "logical_calls": model.llm_counter.call_count,
        "operation_count": model.operation_count,
        "actions": logged.actions,
        "states": states,
        "search_records": model.search_records,
        "tail_records": getattr(model, "tail_records", []),
        "glm_calls": 0,
        "network_calls": 0,
    }
    save(target, result)
    return result


def reproduce_six():
    representatives = [fixed_dev_cases()[index] for index in (0, 4, 8, 12, 16, 20)]
    rows = []
    for item in representatives:
        result = run_case(item["case"], "D5")
        historical = json.loads(
            (OUT.parent / "round5/P3" / f"{item['case']}-D-search.json").read_text(encoding="utf-8")
        )
        old_actions = [(row["action"], list(row["args"])) for row in historical["actions"]]
        new_actions = [(row["action"], list(row["args"])) for row in result["actions"]]
        rows.append(
            {
                "case": item["case"],
                "family": item["family"],
                "actions_equal": new_actions == old_actions,
                "score_error": result["final_score"] - historical["final_score"],
                "remaining_budget": result["remaining_budget"],
                "stop_reason": result["stop_reason"],
            }
        )
    report = {
        "status": "PASSED"
        if all(row["actions_equal"] and abs(row["score_error"]) < 1e-8 for row in rows)
        else "FAILED",
        "rows": rows,
    }
    save(OUT / "baseline-reproduction.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "reproduce", "p1"), required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_manifests()
    if args.stage == "prepare":
        print(json.dumps({"status": "prepared", "cases": len(fixed_dev_cases())}))
        return
    if args.stage == "reproduce":
        print(json.dumps(reproduce_six(), ensure_ascii=False, indent=2))
        return
    for index, item in enumerate(fixed_dev_cases(), 1):
        for variant in ("D5", "T1"):
            result = run_case(item["case"], variant, args.force)
            print(
                json.dumps(
                    {
                        "completed": index,
                        "case": item["case"],
                        "variant": variant,
                        "score": result["final_score"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
