"""Broad, checkpointed response-learning experiment on fully materialized seeds.

The policy sees only public scan/action receipts.  Hidden ``r`` values are used by
the local environment and as supervised labels, never by policy inference.
This experiment has its own generated train/tune/test/stress pools and does not
open the round-4 final pool.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from benchmark import LoggedEnv
from experiment_round3 import fast_model
from experiment_round5 import F0, FiniteEnv, PublicEnv
from round4_common import ROOT, digest, file_hash, save


OUT = ROOT / "agent_notes/experiments/response-online-v2-20260918"
SEEDS = OUT / "seeds"
RESULTS = OUT / "policy-results"
PERSONAS = ("和平", "中立", "暴力")
CURRENT_PRIORS = {"和平": 20.25, "中立": 15.0, "暴力": 3.0}
FAMILIES = ("ba", "ws", "er", "sbm", "disconnected", "tree")
MODES = (
    "independent",
    "degree_pos",
    "degree_neg",
    "weight_pos",
    "weight_neg",
    "persona_scale",
    "beta",
)
POOL_COUNTS = {"train": 480, "tune": 144, "test": 288, "stress": 144}
PREFIX_SIZES = (0, 1, 2, 4, 8, 16)
EPS = 1e-9
# Frozen when Tune144 began. Later resume-only serialization fixes must not
# invalidate already completed semantic trials.
RUNNER_ID = "e877f3de3e8374e94abb8b8cadba07d164e4d599d173bd8ca869c28248e2db56"


def stable_seed(value) -> int:
    return int(hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16], 16)


def graph_for(family: str, n: int, rng: random.Random) -> nx.Graph:
    seed = rng.randrange(2**32)
    if family == "ba":
        return nx.barabasi_albert_graph(n, rng.choice((1, 2, 3, 4)), seed=seed)
    if family == "ws":
        return nx.watts_strogatz_graph(n, rng.choice((2, 4, 6, 8)), rng.uniform(.06, .55), seed=seed)
    if family == "er":
        return nx.gnp_random_graph(n, rng.uniform(.035, .22), seed=seed)
    if family == "sbm":
        split = rng.randrange(18, 33)
        inside = rng.uniform(.12, .30)
        between = rng.uniform(.002, .045)
        return nx.stochastic_block_model([split, n - split], [[inside, between], [between, inside]], seed=seed)
    if family == "disconnected":
        split = rng.randrange(20, 31)
        left = nx.barabasi_albert_graph(split, rng.choice((1, 2, 3)), seed=seed)
        right = nx.watts_strogatz_graph(n - split, rng.choice((2, 4, 6)), rng.uniform(.08, .45), seed=seed + 1)
        return nx.disjoint_union(left, right)
    if hasattr(nx, "random_labeled_tree"):
        return nx.random_labeled_tree(n, seed=seed)
    return nx.random_tree(n, seed=seed)


def pool_specs() -> list[dict]:
    rows = []
    for pool, count in POOL_COUNTS.items():
        for index in range(count):
            family = FAMILIES[index % len(FAMILIES)]
            mode_choices = MODES + (("no_global",) if pool == "stress" else ())
            mode = mode_choices[(index // len(FAMILIES)) % len(mode_choices)]
            rows.append({
                "case": f"{pool}-{index:04d}",
                "pool": pool,
                "index": index,
                "family": family,
                "mode": mode,
                "n": 50,
                "generator_version": 2,
            })
    return rows


def generate_seed(spec: dict) -> tuple[dict, dict]:
    rng = random.Random(stable_seed(["response-online-v2", spec]))
    n = spec["n"]
    graph = graph_for(spec["family"], n, rng)
    mixes = ((.35, .35, .30), (.20, .20, .60), (.65, .25, .10), (.12, .63, .25))
    mix = mixes[(spec["index"] // 7) % len(mixes)]
    global_scale = rng.uniform(.42, 1.72)
    persona_scale = {p: global_scale for p in PERSONAS}
    if spec["mode"] == "persona_scale":
        persona_scale = {p: global_scale * rng.uniform(.68, 1.32) for p in PERSONAS}
    elif spec["mode"] == "no_global":
        persona_scale = {p: rng.uniform(.38, 1.78) for p in PERSONAS}

    base_ranges = {"和平": (.8, 1.8), "中立": (.5, 1.4), "暴力": (.05, .5)}
    weight_ranges = {"和平": (5, 30), "中立": (-12, 12), "暴力": (-50, -12)}
    personas = rng.choices(PERSONAS, mix, k=n)
    nodes = {}
    for i, persona in enumerate(personas):
        wlo, whi = weight_ranges[persona]
        rlo, rhi = base_ranges[persona]
        if spec["mode"] == "beta":
            unit = rng.betavariate(1.7, 3.2) if persona != "暴力" else rng.betavariate(2.5, 1.8)
            response = rlo + (rhi - rlo) * unit
        else:
            response = rng.uniform(rlo, rhi)
        nodes[i] = {
            "id": i,
            "w": rng.uniform(wlo, whi),
            "r": response * persona_scale[persona],
            "persona": persona,
            "comm_left": 3,
        }

    for persona in PERSONAS:
        ids = [i for i in nodes if nodes[i]["persona"] == persona]
        if spec["mode"].startswith("degree_"):
            values = sorted(nodes[i]["r"] for i in ids)
            ascending = spec["mode"] == "degree_pos"
            order = sorted(ids, key=lambda i: (graph.degree(i), i), reverse=not ascending)
            for node, value in zip(order, values):
                nodes[node]["r"] = value
        elif spec["mode"].startswith("weight_"):
            values = sorted(nodes[i]["r"] for i in ids)
            ascending = spec["mode"] == "weight_pos"
            order = sorted(ids, key=lambda i: (nodes[i]["w"], i), reverse=not ascending)
            for node, value in zip(order, values):
                nodes[node]["r"] = value

    shuffled = list(range(1, n + 1))
    rng.shuffle(shuffled)
    for i, node in nodes.items():
        node["id"] = shuffled[i]
        node["w"] = round(node["w"], 5)
        node["r"] = round(node["r"], 5)
    rows = sorted(nodes.values(), key=lambda row: row["id"])
    seed = {
        "global_setting": {"max_budget": 2 * n, "max_api_calls": 120},
        "original_total": sum(row["w"] for row in rows),
        "nodes": rows,
        "edges": sorted(sorted((shuffled[u], shuffled[v])) for u, v in graph.edges()),
        "prompts": {"1": 15.0, "2": 10.0, "3": -5.0},
    }
    latent = {
        "global_scale": global_scale,
        "persona_scales": persona_scale,
        "persona_mix": mix,
    }
    return seed, latent


def materialize() -> dict:
    entries = []
    for spec in pool_specs():
        seed, latent = generate_seed(spec)
        path = SEEDS / f"{spec['case']}.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if digest(old) != digest(seed):
                raise RuntimeError(f"Seed drift: {spec['case']}")
        else:
            save(path, seed)
        entries.append({"spec": spec, "seed_sha256": digest(seed), "latent": latent})
    if len({row["seed_sha256"] for row in entries}) != len(entries):
        raise AssertionError("Seed collision")
    manifest = {
        "schema": 2,
        "purpose": "Broad synthetic response learning; complete JSON seeds; not official hidden distribution",
        "counts": POOL_COUNTS,
        "entries": entries,
        "generator_sha256": file_hash(__file__),
        "round4_final_opened": False,
    }
    path = OUT / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if [(x["spec"], x["seed_sha256"]) for x in old["entries"]] != [
            (x["spec"], x["seed_sha256"]) for x in entries
        ]:
            raise RuntimeError("Manifest identity drift")
    save(path, manifest)
    return manifest


def load_seed(case: str) -> dict:
    return json.loads((SEEDS / f"{case}.json").read_text(encoding="utf-8"))


def degrees(seed: dict) -> dict[int, int]:
    result = {row["id"]: 0 for row in seed["nodes"]}
    for u, v in seed["edges"]:
        result[u] += 1
        result[v] += 1
    return result


def feature_vector(persona: str, w: float, degree: int, n: int, observations: list[tuple[str, float]]) -> np.ndarray:
    onehot = [float(persona == p) for p in PERSONAS]
    ratios = [min(3.0, max(.15, effect / CURRENT_PRIORS[p])) for p, effect in observations]
    same = [ratio for (p, _), ratio in zip(observations, ratios) if p == persona]
    global_mean = statistics.fmean(ratios) if ratios else 1.0
    global_median = statistics.median(ratios) if ratios else 1.0
    global_sd = statistics.pstdev(ratios) if len(ratios) > 1 else 0.0
    same_mean = statistics.fmean(same) if same else global_mean
    values = [
        1.0,
        *onehot,
        max(-2.0, min(2.0, w / 25.0)),
        degree / max(1, n - 1),
        math.log1p(len(ratios)) / math.log(51.0),
        float(bool(ratios)),
        global_mean - 1.0,
        global_median - 1.0,
        global_sd,
        math.log1p(len(same)) / math.log(18.0),
        same_mean - 1.0,
        *(flag * (global_mean - 1.0) for flag in onehot),
        *(flag * (same_mean - 1.0) for flag in onehot),
    ]
    return np.asarray(values, dtype=float)


FEATURE_NAMES = (
    "intercept", "persona_peace", "persona_neutral", "persona_violent",
    "weight_scaled", "degree_fraction", "observed_count_log", "has_observation",
    "global_ratio_mean", "global_ratio_median", "global_ratio_sd",
    "same_persona_count_log", "same_persona_ratio_mean",
    "peace_x_global", "neutral_x_global", "violent_x_global",
    "peace_x_same", "neutral_x_same", "violent_x_same",
)


@dataclass
class Matrix:
    x: np.ndarray
    y: np.ndarray
    cases: np.ndarray


def training_matrix(manifest: dict) -> Matrix:
    xs, ys, cases = [], [], []
    for entry in manifest["entries"]:
        spec = entry["spec"]
        if spec["pool"] != "train":
            continue
        seed = load_seed(spec["case"])
        by_id = {row["id"]: row for row in seed["nodes"]}
        deg = degrees(seed)
        order = sorted(by_id)
        random.Random(stable_seed(["prefix-order", spec["case"]])).shuffle(order)
        for prefix in PREFIX_SIZES:
            observed_ids = order[:prefix]
            observations = [(by_id[i]["persona"], 15.0 * by_id[i]["r"]) for i in observed_ids]
            targets = order[prefix:prefix + 12]
            for node_id in targets:
                row = by_id[node_id]
                xs.append(feature_vector(row["persona"], row["w"], deg[node_id], len(by_id), observations))
                ys.append(15.0 * row["r"])
                cases.append(spec["case"])
    return Matrix(np.vstack(xs), np.asarray(ys), np.asarray(cases))


class ContextRidge:
    kind = "context_ridge"

    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def fit(self, matrix: Matrix):
        self.mean = matrix.x.mean(axis=0)
        self.scale = matrix.x.std(axis=0)
        self.mean[0] = 0.0
        self.scale[self.scale < EPS] = 1.0
        self.scale[0] = 1.0
        z = (matrix.x - self.mean) / self.scale
        penalty = np.eye(z.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(z.T @ z + penalty, z.T @ matrix.y)
        prediction = z @ self.coef
        self.residual_sd = float(np.sqrt(np.mean(np.square(matrix.y - prediction))))
        return self

    def predict_matrix(self, x: np.ndarray) -> np.ndarray:
        return np.clip(((x - self.mean) / self.scale) @ self.coef, .05, 55.0)

    def predict_one(self, persona, w, degree, n, observations):
        x = feature_vector(persona, w, degree, n, observations)[None, :]
        return float(self.predict_matrix(x)[0])

    def to_dict(self):
        return {
            "kind": self.kind,
            "alpha": self.alpha,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "residual_sd": self.residual_sd,
            "feature_names": list(FEATURE_NAMES),
        }

    @classmethod
    def from_dict(cls, row):
        result = cls(row["alpha"])
        result.mean = np.asarray(row["mean"], dtype=float)
        result.scale = np.asarray(row["scale"], dtype=float)
        result.coef = np.asarray(row["coef"], dtype=float)
        result.residual_sd = float(row["residual_sd"])
        return result


class GlobalCalibrator:
    kind = "nonml_global_calibrator"

    def __init__(self, strength: float):
        self.strength = float(strength)
        self.residual_sd = 7.0

    def predict_one(self, persona, w, degree, n, observations):
        ratios = [min(2.7, max(.2, effect / CURRENT_PRIORS[p])) for p, effect in observations]
        if not ratios:
            return CURRENT_PRIORS[persona]
        center = .5 * statistics.fmean(ratios) + .5 * statistics.median(ratios)
        scale = (self.strength + len(ratios) * center) / (self.strength + len(ratios))
        return CURRENT_PRIORS[persona] * scale

    def to_dict(self):
        return {"kind": self.kind, "strength": self.strength, "residual_sd": self.residual_sd}


def fit_model(manifest: dict) -> tuple[ContextRidge, dict]:
    matrix = training_matrix(manifest)
    fold = np.asarray([stable_seed(["ridge-fold", case]) % 5 for case in matrix.cases])
    train = fold != 0
    valid = ~train
    scores = {}
    for alpha in (.1, 1.0, 10.0, 100.0, 1000.0):
        model = ContextRidge(alpha).fit(Matrix(matrix.x[train], matrix.y[train], matrix.cases[train]))
        pred = model.predict_matrix(matrix.x[valid])
        scores[str(alpha)] = {
            "mae": float(np.mean(np.abs(pred - matrix.y[valid]))),
            "rmse": float(np.sqrt(np.mean(np.square(pred - matrix.y[valid])))),
        }
    alpha = min(map(float, scores), key=lambda value: scores[str(value)]["rmse"])
    model = ContextRidge(alpha).fit(matrix)
    artifact = {
        "schema": 1,
        "model": model.to_dict(),
        "alpha_validation": scores,
        "train_examples": len(matrix.y),
        "train_cases": len(set(matrix.cases)),
        "train_label_mean": float(matrix.y.mean()),
        "train_label_sd": float(matrix.y.std()),
        "round4_final_opened": False,
    }
    save(OUT / "model.json", artifact)
    return model, artifact


def predictor_from_spec(spec: dict, model: ContextRidge):
    if spec["kind"] == "global":
        return GlobalCalibrator(spec["strength"])
    if spec["kind"] == "ridge":
        return model
    raise ValueError(spec)


def policy_class(name: str, predictor, min_observations: int, blend_full: int):
    class OnlineResponse(F0):
        response_variant = name
        response_predictor = predictor

        def _effect_estimate(self, i, data):
            if i in self.comm_effects:
                return self.comm_effects[i], 0.0
            observations = [
                (persona, effect)
                for persona, values in self.persona_observations.items()
                for effect in values.values()
            ]
            prior = CURRENT_PRIORS.get(data["persona"], 15.0)
            if len(observations) < min_observations:
                return prior, abs(prior) * .35
            degree = sum(i in edge for edge in self.local_edges)
            learned = self.response_predictor.predict_one(
                data["persona"], data["w"], degree, len(self.local_nodes), observations
            )
            weight = 1.0 if blend_full <= min_observations else min(
                1.0, (len(observations) - min_observations + 1) / (blend_full - min_observations + 1)
            )
            mean = (1.0 - weight) * prior + weight * learned
            return mean, max(abs(mean) * .1, self.response_predictor.residual_sd)

    OnlineResponse.__name__ = "Response_" + name.replace("-", "_")
    return OnlineResponse


VARIANTS = {
    "NM-global-s2": {"kind": "global", "strength": 2.0, "min_observations": 1, "blend_full": 4},
    "NM-global-s5": {"kind": "global", "strength": 5.0, "min_observations": 1, "blend_full": 6},
    "NM-global-s10": {"kind": "global", "strength": 10.0, "min_observations": 1, "blend_full": 8},
    "ML-context-m0": {"kind": "ridge", "min_observations": 0, "blend_full": 6},
    "ML-context-m2": {"kind": "ridge", "min_observations": 2, "blend_full": 7},
    "ML-context-m4": {"kind": "ridge", "min_observations": 4, "blend_full": 9},
}


def run_policy(seed: dict, spec: dict, variant: str, predictor=None, variant_spec=None) -> dict:
    if variant == "F0":
        model_cls = F0
        identity_variant = {"name": "F0"}
    else:
        model_cls = policy_class(
            variant,
            predictor,
            int(variant_spec["min_observations"]),
            int(variant_spec["blend_full"]),
        )
        identity_variant = {"name": variant, "spec": variant_spec, "predictor": predictor.to_dict()}
    identity = {
        "schema": 1,
        "case": spec["case"],
        "seed_sha256": digest(seed),
        "variant": identity_variant,
        "score": "finite50",
        "selector": "deterministic-first",
        "runner_sha256": RUNNER_ID,
    }
    path = RESULTS / spec["pool"] / f"{spec['case']}-{variant}.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["identity"] != identity:
            raise RuntimeError(f"Result identity mismatch: {path}")
        return old
    people = json.loads((ROOT / "agent_notes/experiments/round5/baseline/config.json").read_text(encoding="utf-8"))["person"]
    raw = FiniteEnv(seed)
    logged = LoggedEnv(raw)
    model = fast_model(PublicEnv(logged), people, model_cls)
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(seed["global_setting"]["max_api_calls"]):
            if model.step():
                break
        else:
            raise RuntimeError("Resource fuse reached")
    result = {
        "identity": identity,
        "case": spec["case"],
        "pool": spec["pool"],
        "family": spec["family"],
        "mode": spec["mode"],
        "variant": variant,
        "final_score": raw.evaluate(),
        "remaining_budget": raw.get_remaining_budget(),
        "actions": logged.actions,
        "logical_calls": model.llm_counter.call_count,
        "stop_reason": model.stop_reason,
        "elapsed_seconds": time.monotonic() - started,
        "glm_http_calls": 0,
    }
    save(path, result)
    return result


def bootstrap_interval(pairs: list[dict]) -> list[float]:
    groups = {}
    for row in pairs:
        groups.setdefault(row["family"], []).append(row["delta"])
    rng = random.Random(20260918)
    values = []
    for _ in range(4000):
        sample = []
        for rows in groups.values():
            sample.extend(rng.choice(rows) for _ in rows)
        values.append(statistics.fmean(sample))
    values.sort()
    return [values[100], values[3900]]


def compare(rows: list[dict], baseline: str, candidate: str) -> dict:
    table = {(row["case"], row["variant"]): row for row in rows}
    pairs = []
    for case in sorted({row["case"] for row in rows}):
        left, right = table[(case, baseline)], table[(case, candidate)]
        pairs.append({
            "case": case,
            "family": left["family"],
            "mode": left["mode"],
            "baseline": left["final_score"],
            "candidate": right["final_score"],
            "delta": right["final_score"] - left["final_score"],
        })
    deltas = [row["delta"] for row in pairs]
    ordered = sorted(deltas)
    tail = max(1, math.ceil(.1 * len(ordered)))
    trimmed = sorted(deltas, reverse=True)[2:]
    by_family = {
        family: statistics.fmean(row["delta"] for row in pairs if row["family"] == family)
        for family in FAMILIES
    }
    by_mode = {
        mode: statistics.fmean(row["delta"] for row in pairs if row["mode"] == mode)
        for mode in sorted({row["mode"] for row in pairs})
    }
    result = {
        "n": len(pairs),
        "baseline": baseline,
        "candidate": candidate,
        "mean_delta": statistics.fmean(deltas),
        "median_delta": statistics.median(deltas),
        "stratified_bootstrap95": bootstrap_interval(pairs),
        "worst10_percent_paired_delta": statistics.fmean(ordered[:tail]),
        "without_best_two_mean": statistics.fmean(trimmed),
        "wins": sum(value > 1e-9 for value in deltas),
        "losses": sum(value < -1e-9 for value in deltas),
        "ties": sum(abs(value) <= 1e-9 for value in deltas),
        "by_family": by_family,
        "by_mode": by_mode,
        "pairs": pairs,
    }
    result["development_gate"] = (
        result["mean_delta"] >= 5.0
        and result["stratified_bootstrap95"][0] > 0.0
        and result["without_best_two_mean"] > 0.0
        and result["worst10_percent_paired_delta"] >= -10.0
    )
    return result


def run_pool(manifest: dict, pool: str, variants: list[str], model: ContextRidge) -> tuple[list[dict], dict]:
    selected = [entry["spec"] for entry in manifest["entries"] if entry["spec"]["pool"] == pool]
    rows = []
    for index, spec in enumerate(selected, 1):
        seed = load_seed(spec["case"])
        for variant in variants:
            if variant == "F0":
                rows.append(run_policy(seed, spec, variant))
            else:
                config = VARIANTS[variant]
                predictor = predictor_from_spec(config, model)
                rows.append(run_policy(seed, spec, variant, predictor, config))
        if index % 12 == 0 or index == len(selected):
            print(json.dumps({"stage": pool, "completed": index, "total": len(selected)}), flush=True)
    comparisons = {variant: compare(rows, "F0", variant) for variant in variants if variant != "F0"}
    return rows, comparisons


def compact(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "pairs"}


def write_report(summary: dict):
    lines = [
        "# 在线响应学习大样本实验",
        "",
        "本轮生成并物化完整 seed JSON；策略推断仅使用公开扫描与游说回执。round4 final 池未读取。",
        "",
        "## 数据与模型",
        "",
        f"- seed 总数：{sum(POOL_COUNTS.values())}；节点总数：{50 * sum(POOL_COUNTS.values())}。",
        f"- 训练 seed：{summary['model']['train_cases']}；训练序列样本：{summary['model']['train_examples']}。",
        f"- 岭回归 alpha：{summary['model']['model']['alpha']}。",
        "- 机制覆盖：连续总体强度、正反 degree/weight 相关、persona 独立漂移、偏态响应；stress 另含无总体强度机制。",
        "",
        "## Tune144（相对 F0）",
        "",
        "| 方案 | 均分差 | 95%区间 | 胜/负/平 | 最差10% | 去最佳2例 | 过门槛 |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, row in summary["tune"].items():
        low, high = row["stratified_bootstrap95"]
        lines.append(
            f"| {name} | {row['mean_delta']:+.4f} | [{low:+.4f}, {high:+.4f}] | "
            f"{row['wins']}/{row['losses']}/{row['ties']} | {row['worst10_percent_paired_delta']:+.4f} | "
            f"{row['without_best_two_mean']:+.4f} | {'是' if row['development_gate'] else '否'} |"
        )
    lines += [
        "",
        "## 独立 Test288 / Stress144（相对 F0）",
        "",
        "| 数据池 | 方案 | 均分差 | 95%区间 | 胜/负/平 | 最差10% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pool in ("test", "stress"):
        for name, row in summary[pool].items():
            low, high = row["stratified_bootstrap95"]
            lines.append(
                f"| {pool} | {name} | {row['mean_delta']:+.4f} | [{low:+.4f}, {high:+.4f}] | "
                f"{row['wins']}/{row['losses']}/{row['ties']} | {row['worst10_percent_paired_delta']:+.4f} |"
            )
    lines += [
        "",
        "## 审计结论",
        "",
        f"- Tune 选出的 ML 方案：{summary['selection']['best_ml']}；非ML方案：{summary['selection']['best_nonml']}。",
        f"- 是否允许进入真实 GLM 闭环：{'是' if summary['selection']['promote_to_glm'] else '否'}。",
        "- 本报告分数是完整 JSON seed 上的 finite50 配对分数，不等同官网榜单均分。",
        "- 官方沙盒采集和 GLM 闭环另行写入独立审计文件。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("generate", "train", "tune", "test", "all"), default="all")
    args = parser.parse_args()
    started = time.monotonic()
    manifest = materialize()
    if args.stage == "generate":
        print(json.dumps({"counts": POOL_COUNTS, "nodes": 50 * sum(POOL_COUNTS.values())}))
        return
    if (OUT / "model.json").exists() and args.stage not in ("train", "all"):
        model_artifact = json.loads((OUT / "model.json").read_text(encoding="utf-8"))
        model = ContextRidge.from_dict(model_artifact["model"])
    else:
        model, model_artifact = fit_model(manifest)
    if args.stage == "train":
        print(json.dumps(model_artifact, ensure_ascii=False, indent=2))
        return

    tune_variants = ["F0", *VARIANTS]
    _, tune = run_pool(manifest, "tune", tune_variants, model)
    save(OUT / "tune-summary.json", {name: compact(row) for name, row in tune.items()})
    if args.stage == "tune":
        print(json.dumps({name: compact(row) for name, row in tune.items()}, ensure_ascii=False, indent=2))
        return

    best_ml = max((name for name in VARIANTS if name.startswith("ML-")), key=lambda name: tune[name]["mean_delta"])
    best_nonml = max((name for name in VARIANTS if name.startswith("NM-")), key=lambda name: tune[name]["mean_delta"])
    selected = ["F0", best_ml, best_nonml]
    _, test = run_pool(manifest, "test", selected, model)
    _, stress = run_pool(manifest, "stress", selected, model)
    promote = bool(tune[best_ml]["development_gate"] and test[best_ml]["development_gate"])
    summary = {
        "schema": 1,
        "model": model_artifact,
        "tune": {name: compact(row) for name, row in tune.items()},
        "test": {name: compact(row) for name, row in test.items()},
        "stress": {name: compact(row) for name, row in stress.items()},
        "selection": {
            "best_ml": best_ml,
            "best_nonml": best_nonml,
            "promote_to_glm": promote,
            "gate": "mean>=5, stratified bootstrap lower>0, without-best-two>0, worst10>=-10 on tune and test",
        },
        "audit": {
            "seed_count": sum(POOL_COUNTS.values()),
            "node_count": 50 * sum(POOL_COUNTS.values()),
            "round4_final_opened": False,
            "glm_http_calls": 0,
            "remote_sandbox_sessions": 0,
            "elapsed_seconds": time.monotonic() - started,
            "source_sha256": file_hash(__file__),
        },
    }
    save(OUT / "summary.json", summary)
    write_report(summary)
    print(json.dumps({
        "selection": summary["selection"],
        "tune": summary["tune"],
        "test": summary["test"],
        "stress": summary["stress"],
        "audit": summary["audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
