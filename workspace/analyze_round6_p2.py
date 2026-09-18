"""Summarize P2 while reusing the registered D5 paired statistic."""

import json

from analyze_round6 import paired
from round6_common import OUT, save


def main():
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in (OUT / "p0-p1").glob("*.json")]
    results = {f"{candidate}_vs_D5": paired(rows, "D5", candidate)
               for candidate in ("L1", "U3", "L3")}
    results["L3_vs_U3"] = paired(rows, "U3", "L3")
    save(OUT / "p2-analysis.json", results)
    compact = {name: {key: value for key, value in result.items() if key != "pairs"}
               for name, result in results.items()}
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
