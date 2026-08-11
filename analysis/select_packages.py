import argparse
import fnmatch
import json
import random
from pathlib import Path

parser = argparse.ArgumentParser(
    description="pick a fixed --pkg_list for generate.py from packages that have "
    "historically completed extraction without a workflow_fail, so the rerun's task "
    "set avoids names known to be unfetchable/unparseable rather than discovering "
    "that live during the ablation matrix. Reads jsonl results only -- no spack "
    "environment needed."
)
parser.add_argument(
    "--sources",
    nargs="+",
    default=["results/main_results/all.jsonl", "results/ground.jsonl"],
    help="historical results jsonl files to scan for workflow_fail history",
)
parser.add_argument(
    "--ignore",
    default="data/ignore.txt",
    help="same ignore file generate.py uses (one name/glob pattern per line, # "
    "comments allowed); excluded here too so the list's count isn't inflated by "
    "names that'll just get filtered out at run time",
)
parser.add_argument("-n", type=int, default=40, help="how many packages to select")
parser.add_argument(
    "--seed", type=int, default=None, help="reproducible sampling of the selection itself"
)
parser.add_argument("--out", default="data/pkg_list.txt")
ARGS = parser.parse_args()

seen: set[str] = set()
tainted: set[str] = set()

for source in ARGS.sources:
    path = Path(source)
    if not path.exists():
        print(f"warning: source {source} not found, skipping")
        continue
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = row.get("pkg_name")
            if not name:
                continue
            seen.add(name)
            if row.get("status") == "workflow_fail":
                tainted.add(name)

clean = seen - tainted
print(f"seen: {len(seen)}, tainted by workflow_fail: {len(tainted)}, clean: {len(clean)}")

ignore_patterns = []
ignore_path = Path(ARGS.ignore)
if ignore_path.exists():
    for line in ignore_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ignore_patterns.append(line)
else:
    print(f"warning: ignore file {ARGS.ignore} not found, nothing will be excluded")


def is_ignored(name: str) -> bool:
    return any(fnmatch.fnmatch(name.lower(), pat.lower()) for pat in ignore_patterns)


candidates = sorted(name for name in clean if not is_ignored(name))
print(f"candidates: {len(candidates)}")

if len(candidates) < ARGS.n:
    print(
        f"warning: only {len(candidates)} candidates available, fewer than "
        f"requested -n {ARGS.n}; writing all of them"
    )

rng = random.Random(ARGS.seed)
selected = sorted(rng.sample(candidates, min(ARGS.n, len(candidates))))

with open(ARGS.out, "w") as f:
    for name in selected:
        f.write(name + "\n")

print(f"wrote {len(selected)} package names to {ARGS.out}")
