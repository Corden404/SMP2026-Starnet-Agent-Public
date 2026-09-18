"""Round 6 paired summaries using D5 as the only main baseline."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict

from round6_common import OUT, save


def paired(rows, baseline="D5", candidate="T1"):
    index = {(row["case"], row["variant"]): row for row in rows}
    pairs = []
    for case in sorted({row["case"] for row in rows}):
        if (case, baseline) not in index or (case, candidate) not in index:
            continue
        before, after = index[(case, baseline)], index[(case, candidate)]
        delta = after["final_score"] - before["final_score"]
        pairs.append(
            {
                "case": case,
                "family": before["spec"]["family"],
                "delta": delta,
                "baseline": before["final_score"],
                "candidate": after["final_score"],
                "tail_selected": sum(
                    record.get("status") == "selected" for record in after.get("tail_records", [])
                ),
                "tail_abstained": sum(
                    record.get("status") == "abstain" for record in after.get("tail_records", [])
                ),
            }
        )
    if not pairs:
        return {"n": 0}
    groups = defaultdict(list)
    for row in pairs:
        groups[row["family"]].append(row["delta"])
    rng = random.Random(20260918)
    boot = []
    for _ in range(4000):
        sample = [value for group in groups.values() for value in rng.choices(group, k=len(group))]
        boot.append(statistics.mean(sample))
    boot.sort()
    deltas = [row["delta"] for row in pairs]
    ordered = sorted(deltas)
    tail = max(1, math.ceil(len(deltas) * 0.1))
    return {
        "n": len(pairs),
        "baseline": baseline,
        "candidate": candidate,
        "mean_delta": statistics.mean(deltas),
        "bootstrap95": [boot[100], boot[3900]],
        "win_tie_loss": [
            sum(value > 1e-8 for value in deltas),
            sum(abs(value) <= 1e-8 for value in deltas),
            sum(value < -1e-8 for value in deltas),
        ],
        "worst10_mean": statistics.mean(ordered[:tail]),
        "without_best_two_mean": statistics.mean(ordered[:-2]) if len(ordered) > 2 else None,
        "by_family": {name: statistics.mean(values) for name, values in groups.items()},
        "tail_selected_total": sum(row["tail_selected"] for row in pairs),
        "tail_abstained_total": sum(row["tail_abstained"] for row in pairs),
        "pairs": pairs,
        "scope": "fixed historical development cases; not independent confirmation or leaderboard score",
    }


def main():
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in (OUT / "p0-p1").glob("*.json")]
    result = paired(rows)
    save(OUT / "p1-analysis.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
