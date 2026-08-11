import argparse
import fnmatch
import json
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser(
    description="find packages that installed successfully under some (model, "
    "reference-config) combinations but not others -- these are the packages that "
    "actually discriminate between ablation configs, as opposed to ones that always "
    "or never install regardless of context. Reads jsonl results only, no spack "
    "environment needed."
)
parser.add_argument(
    "--sources",
    nargs="+",
    default=["results/main_results/all.jsonl", "results/ground.jsonl"],
    help="historical results jsonl files to scan",
)
parser.add_argument(
    "--ignore",
    default="data/ignore.txt",
    help="same ignore file generate.py uses (one name/glob pattern per line, # "
    "comments allowed)",
)
parser.add_argument(
    "-n", type=int, default=None, help="limit to the top N most-discriminating packages"
)
parser.add_argument(
    "--out", default=None, help="optional: write just the package names (one per "
    "line) here, e.g. for reuse as a --pkg_list"
)
ARGS = parser.parse_args()

INSTALL_STATUSES = {"install", "test", "test_fail"}


def config_key(row: dict) -> str:
    if row.get("similar_recipe"):
        ref = f"similar{row['similar_recipe']}"
    elif row.get("random_recipe"):
        ref = f"random{row['random_recipe']}"
    elif row.get("random_buildsys_recipe"):
        ref = f"randombuildsys{row['random_buildsys_recipe']}"
    else:
        ref = "none"

    if row.get("raw_buildsys"):
        buildsys = "raw"
    elif row.get("distilled_cmake"):
        buildsys = "distilled"
    else:
        buildsys = "none"

    return f"{row.get('model')}/{ref}_{buildsys}"


# pkg_name -> config_key -> installed (bool, True if ANY row for that group reached install)
outcomes: dict[str, dict[str, bool]] = defaultdict(dict)

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
            status = row.get("status")
            # ground-truth/baseline eval rows (model: null) aren't ablation runs
            if not name or status == "workflow_fail" or row.get("ground_truth_eval"):
                continue

            key = config_key(row)
            installed = status in INSTALL_STATUSES
            # a group counts as installed if ANY row in it reached install
            outcomes[name][key] = outcomes[name].get(key, False) or installed

ignore_patterns = []
ignore_path = Path(ARGS.ignore)
if ignore_path.exists():
    for line in ignore_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ignore_patterns.append(line)


def is_ignored(name: str) -> bool:
    return any(fnmatch.fnmatch(name.lower(), pat.lower()) for pat in ignore_patterns)


discriminating = []
for name, by_config in outcomes.items():
    if is_ignored(name):
        continue
    installed_by = sorted(k for k, ok in by_config.items() if ok)
    not_installed_by = sorted(k for k, ok in by_config.items() if not ok)
    if installed_by and not_installed_by:
        discriminating.append((name, installed_by, not_installed_by))

# most balanced split first -- these show the effect most clearly
discriminating.sort(key=lambda t: min(len(t[1]), len(t[2])), reverse=True)

if ARGS.n:
    discriminating = discriminating[: ARGS.n]

print(f"packages seen: {len(outcomes)}, discriminating: {len(discriminating)}\n")

for name, installed_by, not_installed_by in discriminating:
    print(f"{name}  (installed: {len(installed_by)}, failed: {len(not_installed_by)})")
    print(f"  installed by:     {', '.join(installed_by)}")
    print(f"  not installed by: {', '.join(not_installed_by)}")

if ARGS.out:
    with open(ARGS.out, "w") as f:
        for name, _, _ in discriminating:
            f.write(name + "\n")
    print(f"\nwrote {len(discriminating)} package names to {ARGS.out}")
