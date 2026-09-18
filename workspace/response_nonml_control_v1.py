"""Supplemental non-ML online controls for response_ml_experiment."""

import json

import numpy as np

from analyze_round5 import paired
from response_ml_experiment import (
    CURRENT_PRIORS,
    OUT,
    PersonaMean,
    build_dataset,
    run_policy,
)
from round4_common import save
from round4_seeds import specs


class CurrentPrior:
    """The submitted hand-written persona priors, with no fitted node features."""

    def fit(self, data):
        self.residual_sd = float(np.sqrt(np.mean([
            (y - CURRENT_PRIORS[persona]) ** 2
            for y, persona in zip(data.y, data.personas)
        ])))
        return self

    def predict(self, x, personas):
        return np.asarray([CURRENT_PRIORS[persona] for persona in personas], dtype=float)


def main():
    train = build_dataset(("dev", "corr"))
    variants = {
        "NM-current-global-online": CurrentPrior().fit(train),
        "NM-persona-global-online": PersonaMean().fit(train),
    }
    result_dir = OUT / "select"
    rows = []
    for index, spec in enumerate([item for item in specs() if item["pool"] == "select"], 1):
        case = spec["case"]
        baseline = json.loads((result_dir / f"{case}-F0.json").read_text(encoding="utf-8"))
        rows.append(baseline)
        for variant, model in variants.items():
            path = result_dir / f"{case}-{variant}.json"
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                row = run_policy(case, variant, model, online=True)
                save(path, row)
            rows.append(row)
        print(json.dumps({"completed": index, "total": 54, "case": case}), flush=True)
    comparisons = {name: paired(rows, baseline="F0", candidate=name) for name in variants}
    save(
        OUT / "nonml-controls.json",
        {
            "scope": "Same global online calibration as ML variants, without learned node features",
            "comparisons": comparisons,
            "train_seeds": len(set(train.cases)),
            "train_nodes": len(train.y),
            "glm_http_calls": 0,
            "remote_sandbox_sessions": 0,
        },
    )
    print(json.dumps({
        name: {key: value for key, value in row.items() if key != "pairs"}
        for name, row in comparisons.items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
