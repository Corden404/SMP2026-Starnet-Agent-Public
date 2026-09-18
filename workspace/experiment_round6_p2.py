"""Run the registered P2 L1/U3/L3 feedback ablation."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import time

from benchmark import LoggedEnv
from experiment_round3 import fast_model
from experiment_round5 import FiniteEnv, PublicEnv
from round4_seeds import load_case
from round6_common import OUT, digest, fixed_dev_cases, people, prepare_manifests, save
from round6_feedback import CLASSES, initialize_fast_feedback


def run_case(case, variant, force=False):
    if variant not in CLASSES:
        raise ValueError(variant)
    spec, seed = load_case(case)
    if spec["pool"] != "dev":
        raise ValueError("P2 only uses the fixed development pool")
    target = OUT / "p0-p1" / f"{case}-{variant}.json"
    identity = {
        "schema": 1,
        "case": case,
        "variant": variant,
        "seed_sha256": digest(seed),
        "baseline": "D5 frozen zip",
        "selector": "deterministic first",
        "purpose": "round6 fixed feedback ablation",
    }
    if target.exists() and not force:
        old = json.loads(target.read_text(encoding="utf-8"))
        if old["identity"] != identity:
            raise RuntimeError("Cached identity mismatch")
        return old

    raw = FiniteEnv(seed)
    logged = LoggedEnv(raw)
    model = fast_model(PublicEnv(logged), copy.deepcopy(people()), CLASSES[variant])
    initialize_fast_feedback(model)
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(seed["global_setting"]["max_api_calls"]):
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
        "search_records": model.search_records,
        "feedback_records": model.feedback_records,
        "glm_calls": 0,
        "network_calls": 0,
    }
    save(target, result)
    return result


def main():
    prepare_manifests()
    for index, item in enumerate(fixed_dev_cases(), 1):
        for variant in ("L1", "U3", "L3"):
            result = run_case(item["case"], variant)
            print(json.dumps({"completed": index, "case": item["case"], "variant": variant,
                              "score": result["final_score"]}), flush=True)


if __name__ == "__main__":
    main()
