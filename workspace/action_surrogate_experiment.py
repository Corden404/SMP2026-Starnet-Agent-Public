"""Public-state action-search surrogate experiment.

The model is deliberately *not* a hidden response predictor.  It ranks legal
actions from the public state, then a fixed-size exact finite50 rollout reviews
the shortlist.  The greedy action is always included, so ML can only propose
extra candidates; it cannot remove the fallback.

Long-running labels and policy runs are written one case at a time and can be
resumed safely.  The frozen round-5 development split is used as follows:

* dev-108: training
* corr-72: hyperparameter selection
* select-54: untouched confirmation for this experiment

These are synthetic local scores, not leaderboard scores.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from benchmark import LoggedEnv
from experiment_round3 import fast_model
from experiment_round5 import F0, FiniteEnv, FiniteRollout, PublicEnv, score
from round4_common import ROOT, digest, file_hash, save
from round4_seeds import load_case


OUT = ROOT / "agent_notes/experiments/action-surrogate-20260918"
ROUND5 = ROOT / "agent_notes/experiments/round5"
LABEL_VERSION = "public-mean-one-step-greedy-continuation-v1"
FEATURE_VERSION = "action-public-v1"
TOP_K = 6
MARGIN = 0.5
TARGET_BUDGETS = (75.0, 35.0)
POOL_CASES = {
    "train": [f"dev-{i:03d}" for i in range(108)],
    "tune": [f"corr-{i:03d}" for i in range(72)],
    "confirm": [f"select-{i:03d}" for i in range(54)],
}
PERSONAS = ("和平", "中立", "暴力")
PRIORS = {"和平": 20.25, "中立": 15.0, "暴力": 3.0}


FEATURE_NAMES = (
    "kind_comm", "kind_cut", "kind_shield", "baseline_action",
    "budget_fraction", "budget_after_fraction", "slots_fraction",
    "node_fraction", "edge_density", "component_fraction",
    "base_score_scaled", "weight_mean_scaled", "weight_sd_scaled",
    "weight_min_scaled", "weight_max_scaled", "observed_fraction",
    "candidate_count_scaled", "comm_count_scaled", "cut_count_scaled",
    "shield_count_scaled", "gain_scaled", "roi_scaled", "cost_scaled",
    "overall_rank_fraction", "kind_rank_fraction", "gain_over_best",
    "target_weight_scaled", "target_degree_fraction", "target_influence_scaled",
    "target_comm_left", "target_peace", "target_neutral", "target_violent",
    "target_effect_scaled", "target_effect_observed",
    "other_weight_scaled", "other_degree_fraction", "other_influence_scaled",
    "other_peace", "other_neutral", "other_violent",
    "endpoint_weight_sum_scaled", "endpoint_weight_gap_scaled",
    "endpoint_degree_sum", "endpoint_influence_sum_scaled",
)


def action_key(action: dict) -> tuple:
    return action["action"], action["target_node_1"], action.get("target_node_2")


def public_state(row: dict) -> tuple[dict, set, dict, dict]:
    """Deserialize a saved public-only state and reject hidden fields."""
    nodes = {int(i): dict(data) for i, data in row["nodes"].items()}
    if any("r" in data for data in nodes.values()):
        raise AssertionError("Hidden response leaked into public state")
    edges = {tuple(sorted(edge)) for edge in row["edges"]}
    effects = {int(i): float(value) for i, value in row["effects"].items()}
    observations = {
        persona: {int(i): float(value) for i, value in values.items()}
        for persona, values in row["observations"].items()
    }
    return nodes, edges, effects, observations


def connected_components(nodes: dict, edges: set) -> int:
    adj = {i: set() for i in nodes}
    for u, v in edges:
        if u in adj and v in adj:
            adj[u].add(v)
            adj[v].add(u)
    remaining = set(nodes)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            for v in adj[stack.pop()] & remaining:
                remaining.remove(v)
                stack.append(v)
    return count


def candidate_universe(state: tuple, budget: float, slots: int) -> tuple[FiniteRollout, list[dict]]:
    """Return the exact fallback first, followed by public-belief candidates."""
    nodes, edges, effects, observations = state
    baseline_engine = FiniteRollout(nodes, edges, effects, {}, {}, slots)
    baseline = baseline_engine.candidates(state, budget)
    if not baseline:
        return FiniteRollout(nodes, edges, effects, observations, {}, slots), []
    engine = FiniteRollout(nodes, edges, effects, observations, {}, slots)
    learned = engine.candidates(state, budget)
    result = [baseline[0]]
    seen = {action_key(result[0])}
    for action in learned:
        if action_key(action) not in seen:
            result.append(action)
            seen.add(action_key(action))
    return engine, result


def state_slots(row: dict) -> int:
    return max(0, min(119 - int(row["operation_count"]), 120 - int(row["logical_calls"])))


def select_checkpoints(run: dict) -> list[tuple[str, dict]]:
    eligible = []
    for row in run["states"]:
        if row["operation_count"] < 50 or row["budget"] < 2 or state_slots(row) <= 0:
            continue
        state = public_state(row)
        _, candidates = candidate_universe(state, float(row["budget"]), state_slots(row))
        if len(candidates) >= 2:
            eligible.append(row)
    selected = []
    used = set()
    for target in TARGET_BUDGETS:
        if not eligible:
            break
        row = min(eligible, key=lambda item: (abs(float(item["budget"]) - target), -float(item["budget"])))
        identity = (row["operation_count"], row["prefix_length"])
        if identity not in used:
            selected.append((f"budget-{target:g}", row))
            used.add(identity)
    return selected


def feature_rows(state: tuple, budget: float, slots: int, candidates: list[dict]) -> list[list[float]]:
    nodes, edges, effects, observations = state
    base_score, influence = score(nodes, edges)
    degrees = {i: 0 for i in nodes}
    for u, v in edges:
        if u in degrees and v in degrees:
            degrees[u] += 1
            degrees[v] += 1
    n = len(nodes)
    weights = [float(data["w"]) for data in nodes.values()]
    counts = {kind: sum(a["action"] == kind for a in candidates) for kind in ("comm", "cut", "shield")}
    best_gain = max((abs(float(a["gain"])) for a in candidates), default=1.0)
    overall_ranks = {action_key(a): i for i, a in enumerate(sorted(candidates, key=lambda a: (-a["roi"], -a["gain"], action_key(a))))}
    kind_ranks = {}
    for kind in ("comm", "cut", "shield"):
        ordered = sorted((a for a in candidates if a["action"] == kind), key=lambda a: (-a["roi"], -a["gain"], action_key(a)))
        kind_ranks.update({action_key(a): (i, len(ordered)) for i, a in enumerate(ordered)})
    observed_ids = set(effects)
    observed_ids.update(i for values in observations.values() for i in values)
    components = connected_components(nodes, edges)

    def node_values(node_id: int | None) -> list[float]:
        if node_id is None or node_id not in nodes:
            return [0.0] * 9
        data = nodes[node_id]
        persona = data["persona"]
        mean = FiniteRollout.effect(node_id, data, effects, observations)[0]
        return [
            float(data["w"]) / 50.0,
            degrees[node_id] / max(1, n - 1),
            influence[node_id] / 5.0,
            float(data["comm_left"]) / 3.0,
            float(persona == "和平"), float(persona == "中立"), float(persona == "暴力"),
            mean / 30.0,
            float(node_id in observed_ids),
        ]

    result = []
    for action in candidates:
        kind = action["action"]
        key = action_key(action)
        u, v = action["target_node_1"], action.get("target_node_2")
        primary = node_values(u)
        secondary_full = node_values(v)
        secondary = secondary_full[:3] + secondary_full[4:7]
        rank, kind_total = kind_ranks[key]
        endpoint_weight_sum = (nodes[u]["w"] + (nodes[v]["w"] if v in nodes else 0.0)) / 100.0
        endpoint_weight_gap = abs(nodes[u]["w"] - (nodes[v]["w"] if v in nodes else nodes[u]["w"])) / 50.0
        endpoint_degree_sum = (degrees[u] + (degrees[v] if v in nodes else 0)) / max(1, 2 * (n - 1))
        endpoint_influence_sum = (influence[u] + (influence[v] if v in nodes else 0.0)) / 10.0
        row = [
            float(kind == "comm"), float(kind == "cut"), float(kind == "shield"), float(key == action_key(candidates[0])),
            budget / 100.0, (budget - float(action["cost"])) / 100.0, slots / 70.0,
            n / 50.0, len(edges) / max(1.0, n * (n - 1) / 2), components / max(1, n),
            base_score / 1000.0, statistics.fmean(weights) / 50.0,
            (statistics.pstdev(weights) if len(weights) > 1 else 0.0) / 50.0,
            min(weights) / 50.0, max(weights) / 50.0, len(observed_ids) / max(1, n),
            len(candidates) / 150.0, counts["comm"] / 50.0, counts["cut"] / 100.0, counts["shield"] / 50.0,
            float(action["gain"]) / 100.0, float(action["roi"]) / 50.0, float(action["cost"]) / 5.0,
            overall_ranks[key] / max(1, len(candidates) - 1), rank / max(1, kind_total - 1),
            float(action["gain"]) / max(1.0, best_gain),
            *primary,
            *secondary,
            endpoint_weight_sum, endpoint_weight_gap, endpoint_degree_sum, endpoint_influence_sum,
        ]
        if len(row) != len(FEATURE_NAMES) or not all(math.isfinite(x) for x in row):
            raise AssertionError((len(row), len(FEATURE_NAMES), key))
        result.append(row)
    return result


def rollout_values(engine: FiniteRollout, state: tuple, budget: float, slots: int, candidates: list[dict]) -> list[float]:
    nodes, _, effects, observations = state
    latent = {i: engine.effect(i, data, effects, observations)[0] for i, data in nodes.items()}
    values = []
    for action in candidates:
        result = engine.continuation(state, budget, slots, latent, False, (action,))
        if result is None:
            raise AssertionError(f"Illegal candidate {action_key(action)}")
        values.append(float(result["score"]))
    return values


def source_identity() -> dict:
    return {
        "label_version": LABEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "experiment_round5_sha256": file_hash(ROOT / "workspace/experiment_round5.py"),
        "submission_model_sha256": file_hash(ROOT / "workspace/team_submission/starnet_model.py"),
        "target_budgets": list(TARGET_BUDGETS),
        "hidden_response_features": False,
        "latent_for_labels": "posterior public mean only",
    }


def label_case(job: tuple[str, str]) -> dict:
    pool, case = job
    path = OUT / "labels" / pool / f"{case}.json"
    identity = {"case": case, "pool": pool, **source_identity()}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("identity") != identity:
            raise RuntimeError(f"Label cache identity mismatch: {path}")
        return {"case": case, "pool": pool, "cached": True, "states": len(old["states"]), "seconds": 0.0}
    run_path = ROUND5 / "P2" / f"{case}-F0.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    start = time.monotonic()
    rows = []
    for checkpoint, state_row in select_checkpoints(run):
        state = public_state(state_row)
        budget = float(state_row["budget"])
        slots = state_slots(state_row)
        engine, candidates = candidate_universe(state, budget, slots)
        features = feature_rows(state, budget, slots, candidates)
        values = rollout_values(engine, state, budget, slots, candidates)
        baseline = values[0]
        rows.append({
            "state_id": f"{case}/{checkpoint}",
            "checkpoint": checkpoint,
            "budget": budget,
            "slots": slots,
            "operation_count": state_row["operation_count"],
            "candidate_count": len(candidates),
            "features": features,
            "actions": [{"key": action_key(a), "gain": a["gain"], "roi": a["roi"], "cost": a["cost"]} for a in candidates],
            "values": values,
            "targets": [value - baseline for value in values],
            "oracle_index": int(np.argmax(values)),
        })
    result = {"identity": identity, "spec": run["spec"], "states": rows, "elapsed_seconds": time.monotonic() - start}
    save(path, result)
    return {"case": case, "pool": pool, "cached": False, "states": len(rows), "seconds": result["elapsed_seconds"]}


def run_labels(workers: int, limits: dict[str, int] | None = None) -> None:
    jobs = []
    for pool, cases in POOL_CASES.items():
        selected = cases[: limits.get(pool, len(cases))] if limits else cases
        jobs.extend((pool, case) for case in selected)
    completed = 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(label_case, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if completed == 1 or completed % 12 == 0 or completed == len(jobs):
                print(json.dumps({"stage": "labels", "completed": completed, "total": len(jobs), "last": result,
                                  "wall_seconds": time.monotonic() - started}, ensure_ascii=False), flush=True)


def load_rows(pool: str, limit: int | None = None) -> list[dict]:
    cases = POOL_CASES[pool][:limit]
    rows = []
    for case in cases:
        path = OUT / "labels" / pool / f"{case}.json"
        if not path.exists():
            raise RuntimeError(f"Missing label file: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["identity"] != {"case": case, "pool": pool, **source_identity()}:
            raise RuntimeError(f"Label identity mismatch: {path}")
        rows.extend(data["states"])
    return rows


def matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([features for row in rows for features in row["features"]], dtype=float)
    y = np.asarray([target for row in rows for target in row["targets"]], dtype=float)
    return x, y


def control_indices(row: dict, top_k: int = TOP_K) -> list[int]:
    """Strong non-ML control: fallback + best of each type + global ROI."""
    selected = [0]
    for kind in ("comm", "cut", "shield"):
        index = next((i for i, action in enumerate(row["actions"]) if action["key"][0] == kind), None)
        if index is not None and index not in selected:
            selected.append(index)
    for index in sorted(range(len(row["actions"])), key=lambda i: (-row["actions"][i]["roi"], -row["actions"][i]["gain"])):
        if index not in selected:
            selected.append(index)
        if len(selected) >= min(top_k, len(row["actions"])):
            break
    return selected[:top_k]


def ml_indices(prediction: np.ndarray, top_k: int = TOP_K) -> list[int]:
    selected = [0]
    for index in np.argsort(-prediction):
        index = int(index)
        if index not in selected:
            selected.append(index)
        if len(selected) >= min(top_k, len(prediction)):
            break
    return selected


def ranking_metrics(rows: list[dict], model) -> dict:
    records = []
    for row in rows:
        prediction = np.asarray(model.predict(np.asarray(row["features"], dtype=float)))
        values = np.asarray(row["values"], dtype=float)
        oracle = int(np.argmax(values))
        selections = {"ml": ml_indices(prediction), "control": control_indices(row)}
        record = {"state_id": row["state_id"], "candidates": len(values), "oracle": oracle,
                  "baseline_regret": float(values[oracle] - values[0])}
        for name, indices in selections.items():
            best = max(indices, key=lambda i: values[i])
            record[name] = {
                "indices": indices,
                "recall": oracle in indices,
                "reviewed_index": int(best),
                "regret": float(values[oracle] - values[best]),
                "gain_over_fallback": float(values[best] - values[0]),
            }
        records.append(record)
    result = {"states": len(records), "candidates": sum(r["candidates"] for r in records)}
    for name in ("ml", "control"):
        regrets = [r[name]["regret"] for r in records]
        result[name] = {
            "recall_at_k": statistics.fmean(float(r[name]["recall"]) for r in records),
            "mean_oracle_regret": statistics.fmean(regrets),
            "p95_oracle_regret": float(np.quantile(regrets, .95)),
            "mean_gain_over_fallback": statistics.fmean(r[name]["gain_over_fallback"] for r in records),
            "zero_regret_rate": statistics.fmean(float(abs(x) < 1e-9) for x in regrets),
        }
    result["mean_fallback_regret"] = statistics.fmean(r["baseline_regret"] for r in records)
    return result


def candidate_models() -> list[tuple[str, object]]:
    result = []
    for leaf in (2, 5, 10):
        for features in (.65, 1.0):
            name = f"extra-leaf{leaf}-features{features:g}"
            result.append((name, ExtraTreesRegressor(
                n_estimators=192, min_samples_leaf=leaf, max_features=features,
                n_jobs=-1, random_state=20260918,
            )))
    result.extend([
        ("hist-leaf15-l2", HistGradientBoostingRegressor(max_leaf_nodes=15, l2_regularization=1.0, random_state=20260918)),
        ("hist-leaf31-l4", HistGradientBoostingRegressor(max_leaf_nodes=31, l2_regularization=4.0, random_state=20260918)),
    ])
    return result


def train_model(limits: dict[str, int] | None = None) -> dict:
    train_rows = load_rows("train", limits.get("train") if limits else None)
    tune_rows = load_rows("tune", limits.get("tune") if limits else None)
    confirm_rows = load_rows("confirm", limits.get("confirm") if limits else None)
    x_train, y_train = matrix(train_rows)
    trials = []
    best = None
    for name, model in candidate_models():
        start = time.monotonic()
        model.fit(x_train, y_train)
        metrics = ranking_metrics(tune_rows, model)
        trial = {"name": name, "seconds": time.monotonic() - start, "tune": metrics}
        trials.append(trial)
        key = (metrics["ml"]["mean_oracle_regret"], -metrics["ml"]["recall_at_k"], name)
        if best is None or key < best[0]:
            best = (key, name, model)
        print(json.dumps({"stage": "fit", **trial}, ensure_ascii=False), flush=True)
    _, name, _ = best
    chosen = dict(candidate_models())[name]
    x_fit, y_fit = matrix(train_rows + tune_rows)
    chosen.fit(x_fit, y_fit)
    artifact = {
        "model": chosen,
        "name": name,
        "feature_names": FEATURE_NAMES,
        "feature_version": FEATURE_VERSION,
        "label_version": LABEL_VERSION,
        "top_k": TOP_K,
        "margin": MARGIN,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, OUT / "model.joblib")
    summary = {
        "identity": source_identity(),
        "split": {pool: len(POOL_CASES[pool][: limits.get(pool, len(POOL_CASES[pool]))]) if limits else len(POOL_CASES[pool]) for pool in POOL_CASES},
        "feature_names": FEATURE_NAMES,
        "top_k": TOP_K,
        "train_rows": len(train_rows),
        "tune_rows": len(tune_rows),
        "confirm_rows": len(confirm_rows),
        "training_candidates": len(y_train),
        "chosen_model": name,
        "trials": trials,
        "train_refit": ranking_metrics(train_rows + tune_rows, chosen),
        "confirm": ranking_metrics(confirm_rows, chosen),
        "hidden_response_features": False,
        "official_score": False,
    }
    save(OUT / "model-summary.json", summary)
    return summary


class SurrogatePolicy(F0):
    """Experimental policy; never imported by the submission package."""

    def _learned_candidates(self, budget):
        reference = super()._learned_candidates(budget)
        if not reference:
            return reference
        slots = min(self.operation_limit - 1 - self.operation_count,
                    self.llm_counter.safe_limit - self.llm_counter.call_count)
        state = (self.local_nodes, self.local_edges, self.comm_effects, self.persona_observations)
        engine, candidates = candidate_universe(state, budget, slots)
        if not candidates:
            return reference
        if self.surrogate_mode == "ml_topk":
            features = np.asarray(feature_rows(state, budget, slots, candidates), dtype=float)
            selected = ml_indices(self.surrogate_artifact["model"].predict(features), self.surrogate_top_k)
        elif self.surrogate_mode == "control_topk":
            row = {"actions": [{"key": action_key(a), "gain": a["gain"], "roi": a["roi"]} for a in candidates]}
            selected = control_indices(row, self.surrogate_top_k)
        else:
            raise ValueError(self.surrogate_mode)
        latent = {i: engine.effect(i, data, self.comm_effects, self.persona_observations)[0]
                  for i, data in self.local_nodes.items()}
        start = time.monotonic()
        values = {}
        for index in selected:
            result = engine.continuation(state, budget, slots, latent, False, (candidates[index],))
            if result is not None:
                values[index] = float(result["score"])
        fallback = 0
        chosen = max(values, key=values.get) if values else fallback
        if chosen != fallback and values[chosen] < values[fallback] + self.surrogate_margin:
            chosen = fallback
        self.surrogate_records.append({
            "operation": self.operation_count,
            "budget": budget,
            "candidate_count": len(candidates),
            "selected_indices": selected,
            "exact_values": values,
            "chosen_index": chosen,
            "fallback_index": fallback,
            "seconds": time.monotonic() - start,
        })
        return [candidates[chosen]]


def policy_identity(case: str, mode: str, artifact: dict) -> dict:
    spec, seed = load_case(case)
    return {
        "case": case,
        "mode": mode,
        "seed_sha256": digest(seed),
        "spec": spec,
        "model_name": artifact["name"],
        "feature_version": artifact["feature_version"],
        "label_version": artifact["label_version"],
        "top_k": artifact["top_k"],
        "margin": artifact["margin"],
        "policy_version": "fixed-k-exact-public-mean-v1",
    }


def run_policy_case(job: tuple[str, str]) -> dict:
    case, mode = job
    artifact = joblib.load(OUT / "model.joblib")
    identity = policy_identity(case, mode, artifact)
    path = OUT / "policy" / f"{case}-{mode}.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("identity") != identity:
            raise RuntimeError(f"Policy cache identity mismatch: {path}")
        return {"case": case, "mode": mode, "cached": True, "score": old["final_score"], "seconds": 0.0}
    spec, seed = load_case(case)
    people = json.loads((ROUND5 / "baseline/config.json").read_text(encoding="utf-8"))["person"]
    raw = FiniteEnv(seed)
    logged = LoggedEnv(raw)
    model = fast_model(PublicEnv(logged), people, SurrogatePolicy)
    model.surrogate_artifact = artifact
    model.surrogate_mode = mode
    model.surrogate_top_k = int(artifact["top_k"])
    model.surrogate_margin = float(artifact["margin"])
    model.surrogate_records = []
    start = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(seed["global_setting"]["max_api_calls"]):
            if model.step():
                break
        else:
            raise RuntimeError("Resource fuse reached")
    result = {
        "identity": identity,
        "spec": spec,
        "final_score": raw.evaluate(),
        "elapsed_seconds": time.monotonic() - start,
        "remaining_budget": raw.get_remaining_budget(),
        "stop_reason": model.stop_reason,
        "logical_calls": model.llm_counter.call_count,
        "actions": logged.actions,
        "search_records": model.surrogate_records,
        "glm_http_calls": 0,
        "tokens": 0,
    }
    save(path, result)
    return {"case": case, "mode": mode, "cached": False, "score": result["final_score"], "seconds": result["elapsed_seconds"]}


def run_policies(workers: int, limit: int | None = None) -> None:
    cases = POOL_CASES["confirm"][:limit]
    jobs = [(case, mode) for case in cases for mode in ("control_topk", "ml_topk")]
    completed = 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_policy_case, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if completed == 1 or completed % 8 == 0 or completed == len(jobs):
                print(json.dumps({"stage": "policy", "completed": completed, "total": len(jobs), "last": result,
                                  "wall_seconds": time.monotonic() - started}, ensure_ascii=False), flush=True)


def bootstrap_ci(rows: list[dict], field: str, baseline: str, draws: int = 5000) -> list[float]:
    rng = np.random.default_rng(20260918)
    families = sorted({row["family"] for row in rows})
    by_family = {family: [row for row in rows if row["family"] == family] for family in families}
    samples = []
    for _ in range(draws):
        deltas = []
        for family in families:
            group = by_family[family]
            indices = rng.integers(0, len(group), len(group))
            deltas.extend(group[int(i)][field] - group[int(i)][baseline] for i in indices)
        samples.append(statistics.fmean(deltas))
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def paired_summary(rows: list[dict], candidate: str, baseline: str) -> dict:
    deltas = [row[candidate] - row[baseline] for row in rows]
    ordered = sorted(deltas)
    worst_n = max(1, math.ceil(.1 * len(ordered)))
    return {
        "n": len(rows),
        "candidate": candidate,
        "baseline": baseline,
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "stratified_bootstrap95": bootstrap_ci(rows, candidate, baseline),
        "worst10_percent_mean": statistics.fmean(ordered[:worst_n]),
        "wins": sum(x > 1e-9 for x in deltas),
        "ties": sum(abs(x) <= 1e-9 for x in deltas),
        "losses": sum(x < -1e-9 for x in deltas),
        "by_family": {family: statistics.fmean(row[candidate] - row[baseline] for row in rows if row["family"] == family)
                      for family in sorted({row["family"] for row in rows})},
    }


def analyze_policies(limit: int | None = None) -> dict:
    cases = POOL_CASES["confirm"][:limit]
    rows = []
    runtimes = {"F0": [], "control_topk": [], "ml_topk": []}
    exact_calls = {"control_topk": [], "ml_topk": []}
    for case in cases:
        baseline = json.loads((ROUND5 / "P2" / f"{case}-F0.json").read_text(encoding="utf-8"))
        row = {"case": case, "family": baseline["spec"]["family"], "F0": baseline["final_score"]}
        runtimes["F0"].append(baseline["elapsed_seconds"])
        for mode in ("control_topk", "ml_topk"):
            result = json.loads((OUT / "policy" / f"{case}-{mode}.json").read_text(encoding="utf-8"))
            row[mode] = result["final_score"]
            runtimes[mode].append(result["elapsed_seconds"])
            exact_calls[mode].append(sum(len(record["exact_values"]) for record in result["search_records"]))
        rows.append(row)
    comparisons = {
        "control_vs_f0": paired_summary(rows, "control_topk", "F0"),
        "ml_vs_f0": paired_summary(rows, "ml_topk", "F0"),
        "ml_vs_control": paired_summary(rows, "ml_topk", "control_topk"),
    }
    gate = {
        "ml_vs_f0_positive_ci": comparisons["ml_vs_f0"]["stratified_bootstrap95"][0] > 0,
        "ml_vs_control_positive_ci": comparisons["ml_vs_control"]["stratified_bootstrap95"][0] > 0,
        "ml_worst10_nonnegative": comparisons["ml_vs_f0"]["worst10_percent_mean"] >= 0,
    }
    gate["passed"] = all(gate.values())
    summary = {
        "scope": "Synthetic local finite50 confirmation; not official leaderboard score",
        "cases": len(cases),
        "top_k": TOP_K,
        "margin": MARGIN,
        "comparisons": comparisons,
        "runtime": {name: {"mean_seconds": statistics.fmean(values), "p95_seconds": float(np.quantile(values, .95))}
                    for name, values in runtimes.items()},
        "exact_rollout_calls": {name: {"mean": statistics.fmean(values), "max": max(values)} for name, values in exact_calls.items()},
        "gate": gate,
        "official_sandbox_run_authorized": bool(gate["passed"]),
        "glm_calls": 0,
        "tokens": 0,
        "rows": rows,
    }
    save(OUT / "policy-summary.json", summary)
    return summary


def parse_limits(value: str | None) -> dict[str, int] | None:
    if not value:
        return None
    number = int(value)
    return {pool: min(number, len(cases)) for pool, cases in POOL_CASES.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("labels", "train", "policy", "analyze", "all"), default="all")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit-per-pool", type=int)
    args = parser.parse_args()
    limits = parse_limits(str(args.limit_per_pool) if args.limit_per_pool else None)
    if args.stage in ("labels", "all"):
        run_labels(args.workers, limits)
    if args.stage in ("train", "all"):
        summary = train_model(limits)
        print(json.dumps({"stage": "train-complete", "chosen": summary["chosen_model"], "confirm": summary["confirm"]}, ensure_ascii=False), flush=True)
    if args.stage in ("policy", "all"):
        run_policies(args.workers, limits.get("confirm") if limits else None)
    if args.stage in ("analyze", "all"):
        summary = analyze_policies(limits.get("confirm") if limits else None)
        print(json.dumps({"stage": "analysis-complete", "comparisons": summary["comparisons"], "gate": summary["gate"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
