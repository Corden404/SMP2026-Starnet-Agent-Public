"""Checkpointed official-sandbox response collection with no LLM calls.

This reuses the validated Store/Journal/Gate/DurableEnv implementation without
modifying it.  Each session scans a complete materialized seed and obtains 37
first-response receipts within the 100-point budget.
"""

from __future__ import annotations

import argparse
import json
import math
import time

from recovery_env import DurableEnv, Gate, InfrastructureFailure, Transport
from recovery_runner import Health, SERVER
from recovery_store import Journal, RecoveryStop, Store, fingerprint
from response_online_v2 import OUT, digest, load_seed, materialize
from round4_common import save


COLLECTION = OUT / "official-collection"
JOURNAL = COLLECTION / "journal.sqlite"


def jobs(manifest: dict) -> list[dict]:
    tests = [row for row in manifest["entries"] if row["spec"]["pool"] == "test"]
    selected = tests[:42]  # 7 response mechanisms x 6 graph families.
    result = []
    for row in selected:
        repeats = 2 if row["spec"]["index"] < 6 else 1
        for repeat in range(repeats):
            result.append({
                "case": row["spec"]["case"],
                "repeat": repeat,
                "family": row["spec"]["family"],
                "mode": row["spec"]["mode"],
                "seed_sha256": row["seed_sha256"],
            })
    return result


def identity(job: dict) -> dict:
    return {
        "schema": 1,
        "purpose": "official first-response receipt collection",
        "server": SERVER,
        "job": job,
        "actions": "scan all nodes in id order; communicate prompt1 to deterministic 37-node sample; evaluate",
        "llm": "none",
        "recovery": "existing recovery_store/recovery_env components",
    }


def selected_targets(case: str, scans: dict[int, dict]) -> list[int]:
    # Hash ordering does not use hidden r and remains invariant across replay.
    return sorted(scans, key=lambda node: fingerprint(["official-label", case, node]))[:37]


def execute(job: dict, env: DurableEnv, journal: Journal) -> dict:
    seed = load_seed(job["case"])
    if digest(seed) != job["seed_sha256"]:
        raise RecoveryStop("SEED_IDENTITY_CHANGED", job["case"])
    scans = {}
    for node_id in sorted(row["id"] for row in seed["nodes"]):
        row = env.scan_node(node_id)
        if not isinstance(row, dict) or not isinstance(row.get("w"), (int, float)):
            raise RecoveryStop("SANDBOX_PROTOCOL_FAILURE", f"invalid scan {node_id}")
        scans[node_id] = row
    receipts = []
    by_id = {row["id"]: row for row in seed["nodes"]}
    for node_id in selected_targets(job["case"], scans):
        before = float(scans[node_id]["w"])
        response = env.communicate(node_id, 1)
        if response.get("status") != "success" or not isinstance(response.get("new_w"), (int, float)):
            raise RecoveryStop("SANDBOX_PROTOCOL_FAILURE", f"invalid communicate {node_id}")
        effect = float(response["new_w"]) - before
        expected = 15.0 * float(by_id[node_id]["r"])
        receipts.append({
            "node_id": node_id,
            "persona": scans[node_id]["persona"],
            "public_w": before,
            "degree": len(scans[node_id].get("neighbors", [])),
            "effect": effect,
            "seed_expected_effect": expected,
            "absolute_error": abs(effect - expected),
        })
    budget = env.get_remaining_budget()
    if not math.isclose(budget, 1.0, abs_tol=1e-9):
        raise RecoveryStop("SANDBOX_PROTOCOL_FAILURE", f"unexpected budget {budget}")
    final_score = env.trigger_eval()
    journal.assert_end()
    return {
        **job,
        "status": "COMPLETED",
        "session_id_recorded": True,
        "receipts": receipts,
        "remaining_budget": budget,
        "final_score": final_score,
        "max_absolute_response_error": max(row["absolute_error"] for row in receipts),
        "glm_http_calls": 0,
        "tokens": 0,
    }


def write_report(store: Store, all_jobs: list[dict]):
    records = []
    for job in all_jobs:
        tid = fingerprint(identity(job))
        row = store.execute("SELECT status,result FROM trials WHERE id=?", (tid,)).fetchone()
        if row and row["result"]:
            records.append(json.loads(row["result"]))
    wire = [dict(row) for row in store.execute("SELECT * FROM wire").fetchall()]
    sandbox = [row for row in wire if row["source"] == "sandbox"]
    transient = 0
    unknown = 0
    for row in sandbox:
        if row["status"] == "ENV_RESULT_UNKNOWN":
            unknown += 1
        if row["response"]:
            try:
                if json.loads(row["response"]).get("status") in (408, 429, 500, 502, 503, 504):
                    transient += 1
            except (TypeError, ValueError):
                pass
    receipts = [receipt for row in records for receipt in row["receipts"]]
    by_case = {}
    for row in records:
        by_case.setdefault(row["case"], []).append(row)
    repeat_checks = []
    for case, rows in by_case.items():
        if len(rows) > 1:
            first = [(x["node_id"], x["effect"]) for x in rows[0]["receipts"]]
            repeat_checks.append({"case": case, "identical_receipts": all(
                [(x["node_id"], x["effect"]) for x in row["receipts"]] == first for row in rows[1:]
            )})
    report = {
        "schema": 1,
        "target_sessions": len(all_jobs),
        "completed_sessions": len(records),
        "completed_unique_seeds": len(set(row["case"] for row in records)),
        "official_response_receipts": len(receipts),
        "max_absolute_response_error": max((row["absolute_error"] for row in receipts), default=None),
        "all_receipts_match_seed": bool(receipts) and all(row["absolute_error"] <= 1e-9 for row in receipts),
        "repeat_checks": repeat_checks,
        "all_repeats_identical": bool(repeat_checks) and all(row["identical_receipts"] for row in repeat_checks),
        "sandbox_http_attempts": len(sandbox),
        "transient_http_failures": transient,
        "unknown_result_failures": unknown,
        "circuit": store.get("circuit"),
        "glm_http_calls": 0,
        "tokens": 0,
        "records": records,
    }
    save(COLLECTION / "report.json", report)
    return report


def run():
    manifest = materialize()
    all_jobs = jobs(manifest)
    COLLECTION.mkdir(parents=True, exist_ok=True)
    store = Store(JOURNAL)
    gate = Gate(store)
    try:
        health = Health(store, gate)
        gate.recover(health.probe, health.canary)
        for index, job in enumerate(all_jobs, 1):
            tid = store.trial(identity(job))
            row = store.execute("SELECT status FROM trials WHERE id=?", (tid,)).fetchone()
            if row["status"] == "COMPLETED":
                continue
            while True:
                gate.check()
                aid = store.attempt(tid)
                journal = Journal(store, tid, aid)
                try:
                    env = DurableEnv(SERVER, load_seed(job["case"]), journal, Transport(store, gate, aid))
                    result = execute(job, env, journal)
                    result["attempt_id"] = aid
                    store.complete(tid, aid, result)
                    gate.completed()
                    report = write_report(store, all_jobs)
                    print(json.dumps({
                        "stage": "official-collection",
                        "completed": report["completed_sessions"],
                        "total": len(all_jobs),
                        "receipts": report["official_response_receipts"],
                    }), flush=True)
                    break
                except InfrastructureFailure as error:
                    store.end_attempt(aid, "INFRA_FAILURE", error.detail)
                    store.execute("UPDATE trials SET status='WAITING_SANDBOX' WHERE id=?", (tid,))
                    gate.outage(error.detail, trial_failure=True, retry_after=store.get("last_retry_after", 0))
                    write_report(store, all_jobs)
                    gate.recover(health.probe, health.canary)
                except BaseException as error:
                    store.end_attempt(aid, getattr(error, "code", "BLOCKED"), str(error))
                    store.execute("UPDATE trials SET status='BLOCKED' WHERE id=?", (tid,))
                    state = store.get("circuit", {})
                    state.update(phase="BLOCKED", reason=getattr(error, "code", type(error).__name__), detail=str(error), stopped_at=time.time())
                    store.put("circuit", state)
                    write_report(store, all_jobs)
                    raise
        final = write_report(store, all_jobs)
        print(json.dumps({key: value for key, value in final.items() if key != "records"}, ensure_ascii=False, indent=2))
    finally:
        store.close()


def status():
    if not JOURNAL.exists():
        print(json.dumps({"status": "NOT_STARTED"}))
        return
    store = Store(JOURNAL, lock=False)
    try:
        report = write_report(store, jobs(materialize()))
        print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "status"), nargs="?", default="status")
    args = parser.parse_args()
    run() if args.command == "run" else status()


if __name__ == "__main__":
    main()
