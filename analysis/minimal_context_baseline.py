import argparse
import json
from collections import defaultdict

parser = argparse.ArgumentParser(
    description="minimal-context baseline: for each results jsonl (typically the "
    "cmake_raw / no-reference condition, one file per model), reports the "
    "attempt_num==0 install rate (a zero-shot/single-attempt proxy, since every "
    "sc26 config runs with k=5 repair built in) alongside the final within-k=5 "
    "install rate. Feeds results.tex's 'Minimal context baseline' paragraph."
)
parser.add_argument("results", nargs="+", help="one or more results jsonl paths")
ARGS = parser.parse_args()

for path in ARGS.results:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        print(f"{path}: empty")
        continue

    by_pkg: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pkg[row["pkg_name"]].append(row)

    model = rows[0]["model"]
    n = len(by_pkg)
    attempt0 = sum(
        1
        for attempts in by_pkg.values()
        if any(r["attempt_num"] == 0 and r["status"] == "install" for r in attempts)
    )
    final = sum(
        1
        for attempts in by_pkg.values()
        if any(r["status"] == "install" for r in attempts)
    )

    print(
        f"{model:30s} n={n:3d}  attempt-0: {attempt0}/{n} = {attempt0 / n:.1%}"
        f"   final (k<=5): {final}/{n} = {final / n:.1%}"
    )
