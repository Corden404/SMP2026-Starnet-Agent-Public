"""Run fixed 42/46-scan diagnostics on the registered 24 cases."""

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
from round6_common import OUT, digest, fixed_dev_cases, people, save
from round6_scan import CLASSES, initialize_fast_scan


def run_case(case, variant):
    spec, seed_data = load_case(case)
    target = OUT / "p6" / f"{case}-{variant}.json"
    identity = {"schema": 1, "case": case, "variant": variant, "seed_sha256": digest(seed_data),
                "purpose": "late scan value diagnostic; not a submission candidate"}
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if old["identity"] != identity:
            raise RuntimeError("Cached identity mismatch")
        return old
    raw = FiniteEnv(seed_data)
    logged = LoggedEnv(raw)
    model = fast_model(PublicEnv(logged), copy.deepcopy(people()), CLASSES[variant])
    initialize_fast_scan(model)
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(seed_data["global_setting"]["max_api_calls"]):
            if model.step():
                break
        else:
            raise RuntimeError("Resource fuse reached")
    result = {
        "identity": identity, "case": case, "variant": variant, "spec": spec,
        "final_score": raw.evaluate(), "elapsed_seconds": time.monotonic() - started,
        "remaining_budget": raw.get_remaining_budget(), "stop_reason": model.stop_reason,
        "logical_calls": model.llm_counter.call_count, "operation_count": model.operation_count,
        "actions": logged.actions, "search_records": model.search_records,
        "scan_diagnostic_records": model.scan_diagnostic_records,
        "scans": sum(row["action"] == "scan_node" for row in logged.actions),
        "glm_calls": 0, "network_calls": 0,
    }
    save(target, result)
    return result


def main():
    for index, item in enumerate(fixed_dev_cases(), 1):
        for variant in ("S42", "S46"):
            result = run_case(item["case"], variant)
            print(json.dumps({"completed": index, "case": item["case"], "variant": variant,
                              "score": result["final_score"]}), flush=True)


if __name__ == "__main__":
    main()
