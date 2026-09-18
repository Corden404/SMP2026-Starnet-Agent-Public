"""Robust response follow-up with fresh tune/confirm pools."""
from __future__ import annotations

import contextlib
import io
import json
import random
import statistics
import time

import numpy as np

from benchmark import LoggedEnv
from experiment_round3 import fast_model
from experiment_round5 import F0, FiniteEnv, PublicEnv
from response_online_v2 import (
    CURRENT_PRIORS, FAMILIES, Matrix, ContextRidge, compare, degrees,
    feature_vector, generate_seed, materialize as materialize_v2,
    load_seed as load_v2_seed, stable_seed,
)
from round4_common import ROOT, digest, file_hash, save

OUT = ROOT / "agent_notes/experiments/response-online-v3-20260918"
SEEDS = OUT / "seeds"
RESULTS = OUT / "policy-results"
PERSONAS = ("和平", "中立", "暴力")
MODES = ("independent", "degree_pos", "degree_neg", "weight_pos",
         "weight_neg", "persona_scale", "beta", "no_global")
COUNTS = {"robust_train": 240, "tune": 144, "confirm": 288}
RUNNER_ID = "response-online-v3-robust-persona-gating-v1"


def specs():
    rows = []
    for pool, count in COUNTS.items():
        for index in range(count):
            rows.append({"case": f"{pool}-{index:04d}", "pool": pool,
                "index": index, "family": FAMILIES[index % len(FAMILIES)],
                "mode": MODES[(index // len(FAMILIES)) % len(MODES)],
                "n": 50, "generator_version": 3})
    return rows


def materialize():
    entries = []
    for spec in specs():
        seed, latent = generate_seed(spec)
        path = SEEDS / f"{spec['case']}.json"
        if path.exists():
            old = json.loads(path.read_text(encoding="utf-8"))
            if digest(old) != digest(seed):
                raise RuntimeError(f"Seed drift: {spec['case']}")
        else:
            save(path, seed)
        entries.append({"spec": spec, "seed_sha256": digest(seed), "latent": latent})
    result = {"schema": 1, "purpose": "Robust online response follow-up with fresh confirmation pool",
        "counts": COUNTS, "entries": entries, "source_sha256": file_hash(__file__),
        "v2_test_or_stress_used_for_training": False, "round4_final_opened": False}
    save(OUT / "manifest.json", result)
    return result


def load_seed(case):
    return json.loads((SEEDS / f"{case}.json").read_text(encoding="utf-8"))


def append_examples(xs, ys, cases, case, seed):
    by_id = {row["id"]: row for row in seed["nodes"]}
    deg = degrees(seed)
    order = sorted(by_id)
    random.Random(stable_seed(["v3-prefix", case])).shuffle(order)
    for prefix in (0, 1, 2, 4, 8, 16):
        observed = [(by_id[i]["persona"], 15.0 * by_id[i]["r"]) for i in order[:prefix]]
        for node_id in order[prefix:prefix + 12]:
            row = by_id[node_id]
            xs.append(feature_vector(row["persona"], row["w"], deg[node_id], len(by_id), observed))
            ys.append(15.0 * row["r"])
            cases.append(case)


def fit_robust_model(manifest):
    xs, ys, cases = [], [], []
    for entry in materialize_v2()["entries"]:
        if entry["spec"]["pool"] == "train":
            case = entry["spec"]["case"]
            append_examples(xs, ys, cases, case, load_v2_seed(case))
    for entry in manifest["entries"]:
        if entry["spec"]["pool"] == "robust_train":
            case = entry["spec"]["case"]
            append_examples(xs, ys, cases, case, load_seed(case))
    matrix = Matrix(np.vstack(xs), np.asarray(ys), np.asarray(cases))
    model = ContextRidge(100.0).fit(matrix)
    artifact = {"schema": 1, "model": model.to_dict(),
        "train_cases": len(set(matrix.cases)), "train_examples": len(matrix.y),
        "includes_v2_train": 480, "includes_fresh_robust_train": 240,
        "includes_no_global_training": True, "v2_test_or_stress_used": False,
        "round4_final_opened": False}
    save(OUT / "model.json", artifact)
    return model, artifact


class PersonaCalibrator:
    kind = "nonml_persona_calibrator"
    residual_sd = 7.0

    def predict_one(self, persona, w, degree, n, observations):
        ratios = [(p, min(2.8, max(.15, effect / CURRENT_PRIORS[p]))) for p, effect in observations]
        same = [ratio for p, ratio in ratios if p == persona]
        if same:
            scale = (3.0 + sum(same)) / (3.0 + len(same))
        else:
            means = [statistics.fmean(r for p, r in ratios if p == name)
                     for name in PERSONAS if any(p == name for p, _ in ratios)]
            if len(means) >= 2 and max(means) - min(means) <= .45:
                global_mean = statistics.fmean(r for _, r in ratios)
                scale = (5.0 + len(ratios) * global_mean) / (5.0 + len(ratios))
            else:
                scale = 1.0
        return CURRENT_PRIORS[persona] * scale

    def to_dict(self):
        return {"kind": self.kind, "same_persona_strength": 3.0,
                "global_strength": 5.0, "consistency_range": .45}


def policy_class(name, predictor, config):
    class RobustOnline(F0):
        response_variant = name
        response_predictor = predictor

        def _effect_estimate(self, i, data):
            if i in self.comm_effects:
                return self.comm_effects[i], 0.0
            observations = [(persona, effect) for persona, values in self.persona_observations.items()
                            for effect in values.values()]
            prior = CURRENT_PRIORS.get(data["persona"], 15.0)
            same = [(p, effect) for p, effect in observations if p == data["persona"]]
            selected = observations
            if config["gate"] == "persona":
                if len(same) < config["min_same"]:
                    return prior, abs(prior) * .35
                selected = same
            elif config["gate"] == "consistency":
                groups = [statistics.fmean(effect / CURRENT_PRIORS[p]
                          for p, effect in observations if p == persona)
                          for persona in PERSONAS if any(p == persona for p, _ in observations)]
                inconsistent = len(groups) >= 2 and max(groups) - min(groups) > config["range"]
                if inconsistent:
                    if len(same) < config["min_same"]:
                        return prior, abs(prior) * .35
                    selected = same
                elif len(observations) < config["min_observations"]:
                    return prior, abs(prior) * .35
            elif len(observations) < config["min_observations"]:
                return prior, abs(prior) * .35
            degree = sum(i in edge for edge in self.local_edges)
            learned = predictor.predict_one(data["persona"], data["w"], degree,
                                            len(self.local_nodes), selected)
            weight = min(1.0, len(selected) / config["blend_full"])
            mean = (1.0 - weight) * prior + weight * learned
            return mean, max(abs(mean) * .1, getattr(predictor, "residual_sd", 7.0))

    RobustOnline.__name__ = "Robust_" + name.replace("-", "_")
    return RobustOnline


VARIANTS = {
    "NM-persona-shrink": {"kind": "nonml", "gate": "all", "min_observations": 1, "blend_full": 1},
    "ML-robust-all": {"kind": "ml", "gate": "all", "min_observations": 2, "blend_full": 8},
    "ML-robust-persona": {"kind": "ml", "gate": "persona", "min_same": 1, "blend_full": 4},
    "ML-robust-consistency": {"kind": "ml", "gate": "consistency", "min_same": 1,
                              "min_observations": 2, "range": .55, "blend_full": 6},
}


def run_one(seed, spec, variant, predictor=None, config=None):
    if variant == "F0":
        cls, identity_variant = F0, {"name": "F0"}
    else:
        cls = policy_class(variant, predictor, config)
        identity_variant = {"name": variant, "config": config, "predictor": predictor.to_dict()}
    identity = {"schema": 1, "case": spec["case"], "seed_sha256": digest(seed),
                "variant": identity_variant, "score": "finite50",
                "selector": "deterministic-first", "runner": RUNNER_ID}
    path = RESULTS / spec["pool"] / f"{spec['case']}-{variant}.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["identity"] != identity:
            raise RuntimeError(f"Identity mismatch: {path}")
        return old
    people = json.loads((ROOT / "agent_notes/experiments/round5/baseline/config.json").read_text(encoding="utf-8"))["person"]
    raw = FiniteEnv(seed)
    logged = LoggedEnv(raw)
    model = fast_model(PublicEnv(logged), people, cls)
    started = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(seed["global_setting"]["max_api_calls"]):
            if model.step():
                break
        else:
            raise RuntimeError("Resource fuse reached")
    result = {"identity": identity, "case": spec["case"], "pool": spec["pool"],
        "family": spec["family"], "mode": spec["mode"], "variant": variant,
        "final_score": raw.evaluate(), "remaining_budget": raw.get_remaining_budget(),
        "actions": logged.actions, "logical_calls": model.llm_counter.call_count,
        "stop_reason": model.stop_reason, "elapsed_seconds": time.monotonic() - started,
        "glm_http_calls": 0}
    save(path, result)
    return result


def run_pool(manifest, pool, variants, model):
    chosen = [entry["spec"] for entry in manifest["entries"] if entry["spec"]["pool"] == pool]
    rows = []
    for index, spec in enumerate(chosen, 1):
        seed = load_seed(spec["case"])
        for variant in variants:
            if variant == "F0":
                rows.append(run_one(seed, spec, variant))
            else:
                config = VARIANTS[variant]
                predictor = PersonaCalibrator() if config["kind"] == "nonml" else model
                rows.append(run_one(seed, spec, variant, predictor, config))
        if index % 12 == 0 or index == len(chosen):
            print(json.dumps({"stage": pool, "completed": index, "total": len(chosen)}), flush=True)
    return rows, {name: compare(rows, "F0", name) for name in variants if name != "F0"}


def compact(row):
    return {key: value for key, value in row.items() if key != "pairs"}


def write_report(summary):
    lines = ["# 在线响应稳健性终检", "",
        "V2 显示 no_global 机制失败后，本轮新增独立 robust_train，并使用全新的 tune/confirm；V2 test/stress 未进入训练。",
        "", "| 数据池 | 方案 | 均分差 | 95%区间 | 胜/负/平 | 最差10% | 过门槛 |",
        "|---|---|---:|---:|---:|---:|:---:|"]
    for pool in ("tune", "confirm"):
        for name, row in summary[pool].items():
            low, high = row["stratified_bootstrap95"]
            lines.append(f"| {pool} | {name} | {row['mean_delta']:+.4f} | [{low:+.4f}, {high:+.4f}] | "
                         f"{row['wins']}/{row['losses']}/{row['ties']} | "
                         f"{row['worst10_percent_paired_delta']:+.4f} | "
                         f"{'是' if row['development_gate'] else '否'} |")
    lines += ["", f"最佳 ML：{summary['selection']['best_ml']}；最佳非ML：{summary['selection']['best_nonml']}。",
        f"进入 GLM 闭环：{'是' if summary['selection']['promote_to_glm'] else '否'}。",
        "分数均为完整自定义 seed 的 finite50 配对分数，不等于官网榜单均分。"]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    started = time.monotonic()
    manifest = materialize()
    model, model_artifact = fit_robust_model(manifest)
    _, tune = run_pool(manifest, "tune", ["F0", *VARIANTS], model)
    best_ml = max((name for name in VARIANTS if name.startswith("ML-")),
                  key=lambda name: tune[name]["mean_delta"])
    best_nonml = "NM-persona-shrink"
    _, confirm = run_pool(manifest, "confirm", ["F0", best_ml, best_nonml], model)
    promote = bool(tune[best_ml]["development_gate"] and confirm[best_ml]["development_gate"])
    summary = {"schema": 1, "model": model_artifact,
        "tune": {name: compact(row) for name, row in tune.items()},
        "confirm": {name: compact(row) for name, row in confirm.items()},
        "selection": {"best_ml": best_ml, "best_nonml": best_nonml,
            "promote_to_glm": promote,
            "gate": "mean>=5, bootstrap lower>0, without-best-two>0, worst10>=-10 on tune and confirm"},
        "audit": {"fresh_seed_count": sum(COUNTS.values()),
            "fresh_node_count": 50 * sum(COUNTS.values()),
            "v2_test_or_stress_used_for_training": False, "round4_final_opened": False,
            "glm_http_calls": 0, "remote_sandbox_sessions": 0,
            "elapsed_seconds": time.monotonic() - started,
            "source_sha256": file_hash(__file__)}}
    save(OUT / "summary.json", summary)
    write_report(summary)
    print(json.dumps({"selection": summary["selection"], "tune": summary["tune"],
                      "confirm": summary["confirm"], "audit": summary["audit"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
