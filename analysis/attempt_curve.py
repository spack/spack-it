import argparse
import json
from collections import defaultdict

parser = argparse.ArgumentParser(
    description="cumulative install rate by attempt number, one line per input "
    "file -- e.g. an audit on/off comparison, or a k=N plateau study. Feeds any "
    "results.tex claim of the form 'X% after attempt N' or 'the largest gains "
    "occurred between attempts M and M+1'."
)
parser.add_argument("results", nargs="+", help="one or more results jsonl paths")
parser.add_argument(
    "--max-attempt",
    type=int,
    default=None,
    help="cap the attempt index shown (0-indexed); omit to use each file's own max_attempts",
)
ARGS = parser.parse_args()

for path in ARGS.results:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        print(f"{path}: empty")
        continue

    by_pkg: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_pkg[row["pkg_name"]].append(row)
    n = len(by_pkg)

    if ARGS.max_attempt is not None:
        max_attempt = ARGS.max_attempt
    else:
        max_attempt = max(r["max_attempts"] for r in rows if r.get("max_attempts")) - 1

    first_success: dict[str, int] = {}
    for pkg, attempts in by_pkg.items():
        for r in sorted(attempts, key=lambda x: x["attempt_num"] if x["attempt_num"] is not None else -1):
            if r["status"] == "install":
                first_success[pkg] = r["attempt_num"]
                break

    cumulative = [
        sum(1 for a in first_success.values() if a <= k) / n for k in range(max_attempt + 1)
    ]
    print(f"{path} (n={n}):", [f"{v:.1%}" for v in cumulative])
