import argparse

from generate.util import load_completed_pkgs

parser = argparse.ArgumentParser(
    description="check how many packages a results jsonl has already completed "
    "against a target --samples count, without running anything"
)
parser.add_argument("--results", required=True, help="path to a results jsonl file")
parser.add_argument(
    "--samples", type=int, required=True, help="target number of completed packages"
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
ARGS = parser.parse_args()

completed = load_completed_pkgs(ARGS.results, ARGS.success_status, ARGS.max_attempts)
remaining = max(ARGS.samples - len(completed), 0)

print(f"completed: {len(completed)} / {ARGS.samples}")
print(f"remaining: {remaining}")
if remaining and completed:
    print(f"already done (will be skipped by --resume): {sorted(completed)}")

raise SystemExit(0 if remaining == 0 else 1)
