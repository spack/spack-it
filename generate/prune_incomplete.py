import argparse
import json
import os
from pathlib import Path

from generate.util import load_completed_pkgs

parser = argparse.ArgumentParser(
    description="delete every row in a results jsonl belonging to a package that "
    "never satisfies load_completed_pkgs' criteria (never logged --success_status "
    "and never reached its last attempt) -- leftover partial attempts from an "
    "interrupted run, and workflow_fail entries. Run this before a --resume rerun "
    "so those packages start clean instead of mixing stale rows in with a fresh "
    "attempt sequence (which would otherwise create duplicate pkg_name/attempt_num "
    "rows with conflicting outcomes)."
)
parser.add_argument("--results", required=True, help="path to a results jsonl file")
parser.add_argument(
    "--success_status",
    type=str,
    default="install",
    choices=["load", "concretize", "install", "test"],
    help="must match the --success_status the results jsonl was generated with",
)
parser.add_argument(
    "--max_attempts",
    type=int,
    default=5,
    help="must match the --max_attempts the results jsonl was generated with",
)
parser.add_argument(
    "--yes",
    action="store_true",
    help="skip the confirmation prompt and prune immediately",
)
ARGS = parser.parse_args()

results_path = Path(ARGS.results)
if not results_path.exists():
    raise SystemExit(f"{results_path} does not exist")

completed = load_completed_pkgs(ARGS.results, ARGS.success_status, ARGS.max_attempts)

rows = []
with results_path.open() as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

kept, pruned = [], []
for row in rows:
    if row.get("pkg_name") in completed:
        kept.append(row)
    else:
        pruned.append(row)

pruned_pkgs = sorted({row.get("pkg_name") for row in pruned})
if not pruned_pkgs:
    print(f"nothing to prune in {results_path}: every package present is completed")
    raise SystemExit(0)

print(
    f"{len(pruned)} row(s) across {len(pruned_pkgs)} incomplete package(s) will be "
    f"deleted from {results_path}:"
)
for name in pruned_pkgs:
    print(f"  {name}")

if not ARGS.yes:
    answer = input(f"delete these {len(pruned)} row(s)? [y/N] ").strip().lower()
    if answer != "y":
        print("aborted, no changes made")
        raise SystemExit(1)

# write-to-temp + atomic rename so a crash mid-write can't leave a truncated
# results file behind
tmp_path = results_path.with_suffix(results_path.suffix + ".tmp")
with tmp_path.open("w") as f:
    for row in kept:
        f.write(json.dumps(row) + "\n")
os.replace(tmp_path, results_path)

print(f"pruned {len(pruned)} row(s), {len(kept)} row(s) remain in {results_path}")
