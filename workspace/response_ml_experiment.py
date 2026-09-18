"""Offline response-learning experiment on the frozen round-4 development pools.

This module never calls the remote sandbox or a language model.  It trains only
on features available after the submission's full scan and compares response
estimators under the same finite50 scorer, deterministic selector, budget, and
action legality rules.  The sealed ``final`` pool is intentionally unsupported.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from analyze_round5 import paired
from benchmark import LoggedEnv
from experiment_round3 import fast_model
from experiment_round5 import F0, FiniteEnv, PublicEnv
from round4_common import ROOT, digest, file_hash, save
from round4_seeds import load_case, specs


OUT = ROOT / "agent_notes/experiments/response-ml-20260918"
POOLS = {"dev": 108, "corr": 72, "select": 54}
PERSONAS = ("和平", "中立", "暴力")
CURRENT_PRIORS = {"和平": 20.25, "中立": 15.0, "暴力": 3.0}
EPS = 1e-12


def _rank_fraction(values: dict[int, float], ids: list[int]) -> dict[int, float]:
    if len(ids) <= 1:
        return {i: 0.5 for i in ids}
    ordered = sorted(ids, key=lambda i: (values[i], i))
    return {i: rank / (len(ordered) - 1) for rank, i in enumerate(ordered)}


def public_feature_rows(nodes: dict[int, dict], edges: set[tuple[int, int]]):
    """Return deterministic public-only features for every currently known node."""
    ids = sorted(nodes)
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    graph.add_edges_from((u, v) for u, v in edges if u in nodes and v in nodes and u != v)
    n = max(1, len(ids))
    weights = {i: float(nodes[i]["w"]) for i in ids}
    degrees = {i: graph.degree(i) for i in ids}
    global_w = np.asarray([weights[i] for i in ids], dtype=float)
    w_mean = float(global_w.mean())
    w_sd = float(global_w.std()) or 1.0
    degree_values = np.asarray([degrees[i] for i in ids], dtype=float)
    degree_mean = float(degree_values.mean())
    degree_sd = float(degree_values.std()) or 1.0
    density = nx.density(graph) if n > 1 else 0.0
    clustering = nx.clustering(graph)
    components = {}
    for component in nx.connected_components(graph):
        fraction = len(component) / n
        component_w = statistics.mean(weights[i] for i in component)
        for i in component:
            components[i] = (fraction, component_w)

    # Pure-numpy PageRank iteration avoids an optional SciPy dependency.
    rank = {i: 1.0 / n for i in ids}
    damping = 0.85
    for _ in range(60):
        nxt = {i: (1.0 - damping) / n for i in ids}
        dangling = sum(rank[i] for i in ids if degrees[i] == 0)
        shared = damping * dangling / n
        for i in ids:
            nxt[i] += shared
        for i in ids:
            if degrees[i]:
                contribution = damping * rank[i] / degrees[i]
                for j in graph.neighbors(i):
                    nxt[j] += contribution
        if max(abs(nxt[i] - rank[i]) for i in ids) < 1e-12:
            rank = nxt
            break
        rank = nxt

    degree_rank = {}
    weight_rank = {}
    for persona in PERSONAS:
        members = [i for i in ids if nodes[i]["persona"] == persona]
        degree_rank.update(_rank_fraction(degrees, members))
        weight_rank.update(_rank_fraction(weights, members))
    persona_fraction = {
        persona: sum(nodes[i]["persona"] == persona for i in ids) / n for persona in PERSONAS
    }

    base_names = [
        "w",
        "w_z",
        "w_rank_persona",
        "degree_fraction",
        "degree_z",
        "degree_rank_persona",
        "pagerank_scaled",
        "clustering",
        "component_fraction",
        "component_w_z",
        "neighbor_w_z",
        "neighbor_peace_fraction",
        "neighbor_neutral_fraction",
        "neighbor_violent_fraction",
        "graph_density",
        "graph_mean_degree_fraction",
        "graph_w_mean_scaled",
        "graph_w_sd_scaled",
        "graph_peace_fraction",
        "graph_neutral_fraction",
        "graph_violent_fraction",
    ]
    names = [f"persona_{p}" for p in PERSONAS] + base_names
    names += [f"persona_{p}_x_{name}" for p in PERSONAS for name in base_names]
    rows = {}
    for i in ids:
        neighbors = list(graph.neighbors(i))
        if neighbors:
            neighbor_w = statistics.mean(weights[j] for j in neighbors)
            neighbor_fractions = [
                sum(nodes[j]["persona"] == persona for j in neighbors) / len(neighbors)
                for persona in PERSONAS
            ]
        else:
            neighbor_w = w_mean
            neighbor_fractions = [0.0, 0.0, 0.0]
        component_fraction, component_w = components[i]
        numeric = [
            weights[i] / 50.0,
            (weights[i] - w_mean) / w_sd,
            weight_rank[i],
            degrees[i] / max(1, n - 1),
            (degrees[i] - degree_mean) / degree_sd,
            degree_rank[i],
            rank[i] * n,
            clustering[i],
            component_fraction,
            (component_w - w_mean) / w_sd,
            (neighbor_w - w_mean) / w_sd,
            *neighbor_fractions,
            density,
            degree_mean / max(1, n - 1),
            w_mean / 50.0,
            w_sd / 50.0,
            *(persona_fraction[p] for p in PERSONAS),
        ]
        one_hot = [float(nodes[i]["persona"] == persona) for persona in PERSONAS]
        interactions = [flag * value for flag in one_hot for value in numeric]
        rows[i] = np.asarray(one_hot + numeric + interactions, dtype=float)
    return names, rows


@dataclass
class Dataset:
    x: np.ndarray
    y: np.ndarray
    cases: np.ndarray
    nodes: np.ndarray
    personas: np.ndarray
    families: np.ndarray
    feature_names: list[str]


def build_dataset(pools: tuple[str, ...]) -> Dataset:
    if any(pool not in POOLS for pool in pools):
        raise ValueError("Only dev, corr, and select pools may be loaded")
    rows = []
    labels = []
    cases = []
    node_ids = []
    personas = []
    families = []
    names = None
    selected = [spec for spec in specs() if spec["pool"] in pools]
    for spec in selected:
        frozen_spec, seed = load_case(spec["case"])
        public_nodes = {
            row["id"]: {key: row[key] for key in ("w", "persona", "comm_left")}
            for row in seed["nodes"]
        }
        edges = {tuple(sorted(edge)) for edge in seed["edges"]}
        feature_names, features = public_feature_rows(public_nodes, edges)
        names = names or feature_names
        if names != feature_names:
            raise AssertionError("Feature schema drift")
        prompt = float(seed["prompts"]["1"])
        for row in seed["nodes"]:
            rows.append(features[row["id"]])
            labels.append(prompt * float(row["r"]))
            cases.append(spec["case"])
            node_ids.append(row["id"])
            personas.append(row["persona"])
            families.append(frozen_spec["family"])
    return Dataset(
        x=np.vstack(rows),
        y=np.asarray(labels, dtype=float),
        cases=np.asarray(cases),
        nodes=np.asarray(node_ids, dtype=int),
        personas=np.asarray(personas),
        families=np.asarray(families),
        feature_names=names or [],
    )


class PersonaMean:
    kind = "persona_mean"

    def fit(self, data: Dataset):
        self.means = {
            persona: float(data.y[data.personas == persona].mean()) for persona in PERSONAS
        }
        self.residual_sd = float(
            np.sqrt(np.mean([(y - self.means[p]) ** 2 for y, p in zip(data.y, data.personas)]))
        )
        return self

    def predict(self, x, personas):
        return np.asarray([self.means[p] for p in personas], dtype=float)

    def to_dict(self):
        return {"kind": self.kind, "means": self.means, "residual_sd": self.residual_sd}


class Ridge:
    kind = "ridge"

    def __init__(self, alpha=10.0):
        self.alpha = float(alpha)

    def fit(self, data: Dataset):
        self.mean = data.x.mean(axis=0)
        self.scale = data.x.std(axis=0)
        self.scale[self.scale < EPS] = 1.0
        z = (data.x - self.mean) / self.scale
        design = np.column_stack((np.ones(len(z)), z))
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0.0
        self.coef = np.linalg.solve(design.T @ design + penalty, design.T @ data.y)
        fitted = design @ self.coef
        self.residual_sd = float(np.sqrt(np.mean((data.y - fitted) ** 2)))
        return self

    def predict(self, x, personas=None):
        z = (np.asarray(x, dtype=float) - self.mean) / self.scale
        design = np.column_stack((np.ones(len(z)), z))
        return np.clip(design @ self.coef, 0.05, 60.0)

    def to_dict(self):
        return {
            "kind": self.kind,
            "alpha": self.alpha,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.coef.tolist(),
            "residual_sd": self.residual_sd,
        }


class BoostedStumps:
    kind = "boosted_stumps"

    def __init__(self, rounds=80, learning_rate=0.05, quantiles=12, min_leaf=60):
        self.rounds = int(rounds)
        self.learning_rate = float(learning_rate)
        self.quantiles = int(quantiles)
        self.min_leaf = int(min_leaf)

    def fit(self, data: Dataset):
        x, y = data.x, data.y
        self.initial = float(y.mean())
        pred = np.full(len(y), self.initial, dtype=float)
        candidates = []
        for feature in range(x.shape[1]):
            thresholds = np.unique(
                np.quantile(x[:, feature], np.linspace(0.05, 0.95, self.quantiles))
            )
            for threshold in thresholds:
                left = x[:, feature] <= threshold
                count = int(left.sum())
                if self.min_leaf <= count <= len(y) - self.min_leaf:
                    candidates.append((feature, float(threshold), left))
        self.stumps = []
        for _ in range(self.rounds):
            residual = y - pred
            best = None
            best_loss = math.inf
            for feature, threshold, left in candidates:
                right = ~left
                left_value = float(residual[left].mean())
                right_value = float(residual[right].mean())
                loss = float(
                    np.square(residual[left] - left_value).sum()
                    + np.square(residual[right] - right_value).sum()
                )
                if loss < best_loss:
                    best_loss = loss
                    best = (feature, threshold, left_value, right_value, left)
            if best is None:
                break
            feature, threshold, left_value, right_value, left = best
            left_value *= self.learning_rate
            right_value *= self.learning_rate
            pred += np.where(left, left_value, right_value)
            self.stumps.append((feature, threshold, left_value, right_value))
        self.residual_sd = float(np.sqrt(np.mean((y - pred) ** 2)))
        return self

    def predict(self, x, personas=None):
        x = np.asarray(x, dtype=float)
        pred = np.full(len(x), self.initial, dtype=float)
        for feature, threshold, left_value, right_value in self.stumps:
            pred += np.where(x[:, feature] <= threshold, left_value, right_value)
        return np.clip(pred, 0.05, 60.0)

    def to_dict(self):
        return {
            "kind": self.kind,
            "rounds": self.rounds,
            "learning_rate": self.learning_rate,
            "quantiles": self.quantiles,
            "min_leaf": self.min_leaf,
            "initial": self.initial,
            "stumps": self.stumps,
            "residual_sd": self.residual_sd,
        }


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    if len(actual) < 2:
        return 0.0
    left, right = _ranks(actual), _ranks(predicted)
    if left.std() < EPS or right.std() < EPS:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def response_metrics(data: Dataset, prediction: np.ndarray):
    error = np.asarray(prediction) - data.y
    by_seed_spearman = []
    top10_overlap = []
    ndcg10 = []
    for case in sorted(set(data.cases)):
        mask = data.cases == case
        actual, predicted = data.y[mask], prediction[mask]
        by_seed_spearman.append(_spearman(actual, predicted))
        k = min(10, len(actual))
        actual_top = set(np.argsort(actual)[-k:])
        predicted_top = set(np.argsort(predicted)[-k:])
        top10_overlap.append(len(actual_top & predicted_top) / k)
        order = np.argsort(predicted)[::-1][:k]
        ideal = np.argsort(actual)[::-1][:k]
        gains = np.maximum(actual, 0.0)
        discount = 1.0 / np.log2(np.arange(2, k + 2))
        dcg = float(np.sum(gains[order] * discount))
        idcg = float(np.sum(gains[ideal] * discount))
        ndcg10.append(dcg / idcg if idcg else 1.0)
    denominator = float(np.sum(np.square(data.y - data.y.mean())))
    return {
        "n_nodes": len(data.y),
        "n_seeds": len(set(data.cases)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "r2": 1.0 - float(np.sum(np.square(error))) / denominator,
        "mean_seed_spearman": statistics.mean(by_seed_spearman),
        "mean_top10_overlap": statistics.mean(top10_overlap),
        "mean_ndcg10": statistics.mean(ndcg10),
        "by_persona_mae": {
            persona: float(np.mean(np.abs(error[data.personas == persona])))
            for persona in PERSONAS
        },
    }


def _fold(case: str, folds=5) -> int:
    return int(hashlib.sha256(case.encode()).hexdigest()[:8], 16) % folds


def subset(data: Dataset, mask: np.ndarray) -> Dataset:
    return Dataset(
        x=data.x[mask],
        y=data.y[mask],
        cases=data.cases[mask],
        nodes=data.nodes[mask],
        personas=data.personas[mask],
        families=data.families[mask],
        feature_names=data.feature_names,
    )


def select_ridge_alpha(train: Dataset):
    alphas = (0.1, 1.0, 10.0, 100.0, 1000.0)
    results = {}
    fold_ids = np.asarray([_fold(case) for case in train.cases])
    for alpha in alphas:
        prediction = np.zeros(len(train.y), dtype=float)
        for fold in range(5):
            model = Ridge(alpha).fit(subset(train, fold_ids != fold))
            prediction[fold_ids == fold] = model.predict(train.x[fold_ids == fold])
        results[str(alpha)] = response_metrics(train, prediction)
    selected = min(alphas, key=lambda alpha: results[str(alpha)]["rmse"])
    return selected, results


def current_prior_prediction(data: Dataset):
    return np.asarray([CURRENT_PRIORS[persona] for persona in data.personas], dtype=float)


def model_class(name: str, predictor, online: bool):
    class LearnedResponse(F0):
        response_predictor = predictor
        response_variant = name

        def _ml_cache(self):
            cached = getattr(self, "_response_ml_cache", None)
            if cached is None:
                _, features = public_feature_rows(self.local_nodes, self.local_edges)
                ids = sorted(features)
                matrix = np.vstack([features[i] for i in ids])
                personas = [self.local_nodes[i]["persona"] for i in ids]
                values = self.response_predictor.predict(matrix, personas)
                cached = dict(zip(ids, map(float, values)))
                self._response_ml_cache = cached
            return cached

        def _effect_estimate(self, i, data):
            if i in self.comm_effects:
                return self.comm_effects[i], 0.0
            base = self._ml_cache()[i]
            if not online or not self.comm_effects:
                return base, self.response_predictor.residual_sd
            cache = self._ml_cache()
            ratios = []
            same_persona = []
            for node, actual in self.comm_effects.items():
                if node not in cache or cache[node] <= EPS:
                    continue
                ratio = min(3.0, max(0.2, actual / cache[node]))
                ratios.append(ratio)
                if self.local_nodes.get(node, {}).get("persona") == data["persona"]:
                    same_persona.append(ratio)
            global_ratio = (4.0 + sum(ratios)) / (4.0 + len(ratios))
            calibrated = (2.0 * global_ratio + sum(same_persona)) / (2.0 + len(same_persona))
            return base * calibrated, self.response_predictor.residual_sd

    LearnedResponse.__name__ = "Response_" + name.replace("-", "_")
    return LearnedResponse


def run_policy(case: str, variant: str, predictor=None, online=False):
    spec, seed = load_case(case)
    if spec["pool"] != "select":
        raise ValueError("Policy comparison is restricted to select")
    people = json.loads(
        (ROOT / "agent_notes/experiments/round5/baseline/config.json").read_text(encoding="utf-8")
    )["person"]
    model_cls = F0 if variant == "F0" else model_class(variant, predictor, online)
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
    return {
        "case": case,
        "variant": variant,
        "spec": spec,
        "final_score": raw.evaluate(),
        "elapsed_seconds": time.monotonic() - started,
        "actions": logged.actions,
        "remaining_budget": raw.get_remaining_budget(),
        "stop_reason": model.stop_reason,
        "logical_calls": model.llm_counter.call_count,
        "glm_http_calls": 0,
        "tokens": 0,
    }


def markdown_report(summary: dict) -> str:
    lines = [
        "# 节点游说响应 ML 对照实验",
        "",
        "本实验只使用 dev+corr 训练，在 select54 上比较策略；final108 未读取。所有策略使用相同 finite50 评分、完整扫描、确定性选择器、预算与动作约束；GLM 和远程沙盒调用均为 0。",
        "",
        "## 响应预测",
        "",
        "| 模型 | MAE | RMSE | R² | seed内Spearman | Top10重合 | NDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["response_validation"].items():
        lines.append(
            f"| {name} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} | "
            f"{row['mean_seed_spearman']:.4f} | {row['mean_top10_overlap']:.4f} | {row['mean_ndcg10']:.4f} |"
        )
    lines += [
        "",
        "## select54 策略分数",
        "",
        "| 方案 | 平均差 | 95%配对区间 | 胜/负/平 | 最差10%均值 | 去掉最佳2例 | 动作变化seed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["policy_comparison"].items():
        deltas = [pair["delta"] for pair in row["pairs"]]
        wins = sum(value > 1e-9 for value in deltas)
        losses = sum(value < -1e-9 for value in deltas)
        ties = len(deltas) - wins - losses
        low, high = row["stratified_bootstrap95"]
        lines.append(
            f"| {name} | {row['mean_delta']:+.4f} | [{low:+.4f}, {high:+.4f}] | "
            f"{wins}/{losses}/{ties} | {row['worst10_percent_paired_delta']:+.4f} | "
            f"{row['without_best_two_mean']:+.4f} | {row['divergent_seeds']}/{row['n']} |"
        )
    lines += [
        "",
        "## 图族分解",
        "",
        "| 方案 | ba | ws | er | sbm | disconnected | tree |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["policy_comparison"].items():
        by = row["by_family"]
        lines.append(
            f"| {name} | {by['ba']:+.3f} | {by['ws']:+.3f} | {by['er']:+.3f} | "
            f"{by['sbm']:+.3f} | {by['disconnected']:+.3f} | {by['tree']:+.3f} |"
        )
    lines += [
        "",
        "## 审计",
        "",
        f"- 训练seed：{summary['audit']['train_seeds']}；训练节点：{summary['audit']['train_nodes']}。",
        f"- 验证seed：{summary['audit']['select_seeds']}；验证节点：{summary['audit']['select_nodes']}。",
        f"- 选择的岭回归 alpha：{summary['audit']['ridge_alpha']}。",
        "- 特征不包含 r、scale、mode、pool、节点随机ID；策略执行时只读取公开扫描状态。",
        "- final 池未加载；结果仅描述人工开发分布，不等价于隐藏榜单收益。",
    ]
    return "\n".join(lines) + "\n"


def main():
    started = time.monotonic()
    train = build_dataset(("dev", "corr"))
    select = build_dataset(("select",))
    ridge_alpha, ridge_cv = select_ridge_alpha(train)
    models = {
        "persona-mean": PersonaMean().fit(train),
        "ridge": Ridge(ridge_alpha).fit(train),
        "boosted-stumps": BoostedStumps().fit(train),
    }
    validation = {"current-prior": response_metrics(select, current_prior_prediction(select))}
    for name, model in models.items():
        validation[name] = response_metrics(select, model.predict(select.x, select.personas))

    variants = {
        "ML-ridge-static": (models["ridge"], False),
        "ML-ridge-online": (models["ridge"], True),
        "ML-stumps-static": (models["boosted-stumps"], False),
        "ML-stumps-online": (models["boosted-stumps"], True),
    }
    result_dir = OUT / "select"
    rows = []
    for index, spec in enumerate([item for item in specs() if item["pool"] == "select"], 1):
        case = spec["case"]
        baseline_path = result_dir / f"{case}-F0.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            baseline = run_policy(case, "F0")
            save(baseline_path, baseline)
        rows.append(baseline)
        for variant, (model, online) in variants.items():
            path = result_dir / f"{case}-{variant}.json"
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                row = run_policy(case, variant, model, online)
                save(path, row)
            rows.append(row)
        print(json.dumps({"completed": index, "total": 54, "case": case}), flush=True)

    comparisons = {name: paired(rows, baseline="F0", candidate=name) for name in variants}
    summary = {
        "schema": 1,
        "scope": "dev+corr supervised training; select54 deterministic finite50 policy comparison; final pool sealed",
        "response_validation": validation,
        "ridge_cv": ridge_cv,
        "models": {name: model.to_dict() for name, model in models.items()},
        "feature_names": train.feature_names,
        "policy_comparison": comparisons,
        "audit": {
            "train_pools": ["dev", "corr"],
            "validation_pool": "select",
            "final_loaded": False,
            "train_seeds": len(set(train.cases)),
            "train_nodes": len(train.y),
            "select_seeds": len(set(select.cases)),
            "select_nodes": len(select.y),
            "ridge_alpha": ridge_alpha,
            "glm_http_calls": 0,
            "remote_sandbox_sessions": 0,
            "elapsed_seconds": time.monotonic() - started,
            "source_hashes": {
                name: file_hash(ROOT / "workspace" / name)
                for name in (
                    "response_ml_experiment.py",
                    "experiment_round5.py",
                    "round4_seeds.py",
                    "team_submission/starnet_model.py",
                )
            },
            "dataset_hash": digest(
                {
                    "train_cases": sorted(set(train.cases)),
                    "select_cases": sorted(set(select.cases)),
                    "features": train.feature_names,
                }
            ),
        },
    }
    save(OUT / "summary.json", summary)
    (OUT / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    compact = {
        "response_validation": validation,
        "policy_comparison": {
            name: {key: value for key, value in row.items() if key != "pairs"}
            for name, row in comparisons.items()
        },
        "audit": summary["audit"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
