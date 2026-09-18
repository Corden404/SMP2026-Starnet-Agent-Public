"""Summarize the P6 diagnostic against existing D5 runs."""

import json

from analyze_round6 import paired
from round6_common import OUT, save


def main():
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in (OUT / "p0-p1").glob("*-D5.json")]
    rows += [json.loads(path.read_text(encoding="utf-8")) for path in (OUT / "p6").glob("*.json")]
    results = {f"{variant}_vs_D5": paired(rows, "D5", variant) for variant in ("S42", "S46")}
    save(OUT / "p6-analysis.json", results)
    print(json.dumps({name: {key: value for key, value in result.items() if key != "pairs"}
                      for name, result in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
