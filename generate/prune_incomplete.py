import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(
    description="delete leftover rows from interrupted runs in one or more results "
    "jsonl files. Groups rows by (run_id, pkg_name) and drops any group that never "
    "logged --success_status and never reached its last attempt -- this covers both "
    "a package with no completed run at all, and a stale partial run_id left behind "
    "by a crash that a later --resume invocation (a new run_id) went on to finish. "
    "Run this before a --resume rerun so packages start clean instead of mixing "
    "stale rows in with a fresh attempt sequence (which would otherwise create "
    "duplicate pkg_name/attempt_num rows with conflicting outcomes)."
)
parser.add_argument(
    "--results",
    required=True,
    nargs="+",
    help="path(s) to one or more results jsonl files",
)
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


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def interrupted_row_indices(rows: list[dict]) -> set[int]:
    """indices of rows belonging to a (run_id, pkg_name) group that never reached
    --success_status and never reached its last attempt -- i.e. a run cut off
    mid-execution, whether or not a later run_id for the same package went on to
    finish it."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(row.get("run_id"), row.get("pkg_name"))].append(i)

    pruned: set[int] = set()
    for idxs in groups.values():
        terminal = any(
            rows[i].get("status") == ARGS.success_status
            or rows[i].get("attempt_num") == ARGS.max_attempts - 1
            for i in idxs
        )
        if not terminal:
            pruned.update(idxs)
    return pruned


plans = []  # (path, rows, pruned_idxs)
total_pruned_rows = 0
for results_arg in ARGS.results:
    results_path = Path(results_arg)
    if not results_path.exists():
        raise SystemExit(f"{results_path} does not exist")

    rows = load_rows(results_path)
    pruned_idxs = interrupted_row_indices(rows)
    if not pruned_idxs:
        print(f"nothing to prune in {results_path}: every (run_id, pkg_name) group is terminal")
        continue

    pruned_pkgs = sorted({rows[i].get("pkg_name") for i in pruned_idxs})
    print(
        f"{results_path}: {len(pruned_idxs)} row(s) across {len(pruned_pkgs)} "
        f"interrupted run(s) will be deleted:"
    )
    for name in pruned_pkgs:
        print(f"  {name}")

    plans.append((results_path, rows, pruned_idxs))
    total_pruned_rows += len(pruned_idxs)

if not plans:
    raise SystemExit(0)

if not ARGS.yes:
    answer = input(
        f"delete these {total_pruned_rows} row(s) across {len(plans)} file(s)? [y/N] "
    ).strip().lower()
    if answer != "y":
        print("aborted, no changes made")
        raise SystemExit(1)

# write-to-temp + atomic rename per file so a crash mid-write can't leave a
# truncated results file behind
for results_path, rows, pruned_idxs in plans:
    kept = [row for i, row in enumerate(rows) if i not in pruned_idxs]
    tmp_path = results_path.with_suffix(results_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    os.replace(tmp_path, results_path)
    print(f"pruned {len(pruned_idxs)} row(s) from {results_path}, {len(kept)} row(s) remain")
