import argparse
import fnmatch
import pickle
from pathlib import Path

parser = argparse.ArgumentParser(
    description="rank-stability sweep for the reference-package retrieval affinity "
    "score (total_score = w_d * dep_overlap + w_b * variant_overlap, currently "
    "hardcoded 0.6/0.4 in generate/util.py's find_similar_packages). For a grid of "
    "(w_d, w_b) pairs, recomputes which reference package(s) each target package "
    "would retrieve, using each target's own ground-truth CMake-detected "
    "dependencies/variants as the query -- so results are independent of any "
    "model's distillation quality and isolate the retrieval formula itself. Reports "
    "what fraction of target packages keep the same top-k selection as the current "
    "0.6/0.4 baseline as weights are perturbed. Needs `spack python` to run, since "
    "unpickling --input requires extraction.package_schema.Package (which imports "
    "spack.fetch_strategy). Do NOT confuse these weights with the separate "
    "0.6/0.2/0.1/0.1 weights in generate/eval.py used for the S_d evaluation metric "
    "-- different formula, different purpose, don't conflate the two in writeups."
)
parser.add_argument(
    "--input", required=True, help="pickled {name: Package} dict, same as generate.py --input"
)
parser.add_argument(
    "--pkg_list",
    default=None,
    help="optional file of exact package names (one per line, # comments allowed) "
    "to use as the query set, e.g. the ablation's data/pkg_list.txt; defaults to "
    "every eligible (cmake, non-py-*) package in --input for more statistical power",
)
parser.add_argument(
    "--ignore",
    default="data/ignore.txt",
    help="same ignore file generate.py uses, applied when --pkg_list is omitted",
)
parser.add_argument(
    "--k",
    type=int,
    default=1,
    help="top-k reference packages to compare for selection agreement (1 for "
    "--similar_recipe 1, 2 for --similar_recipe 2)",
)
parser.add_argument(
    "--baseline_wd",
    type=float,
    default=0.6,
    help="the currently-deployed dependency weight; w_b is derived as 1 - w_d",
)
parser.add_argument(
    "--grid",
    type=str,
    default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    help="comma-separated w_d values to sweep (w_b = 1 - w_d at each point)",
)
ARGS = parser.parse_args()

with open(ARGS.input, "rb") as f:
    pkgs = pickle.load(f)

ignore_patterns = []
ignore_path = Path(ARGS.ignore)
if ignore_path.exists():
    for line in ignore_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ignore_patterns.append(line)


def is_ignored(name: str) -> bool:
    return any(fnmatch.fnmatch(name.lower(), pat.lower()) for pat in ignore_patterns)


if ARGS.pkg_list:
    query_names = [
        line.strip()
        for line in Path(ARGS.pkg_list).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
else:
    query_names = [
        name
        for name, pkg in pkgs.items()
        if not name.startswith("py-")
        and "cmake" in pkg.build_systems
        and not is_ignored(name)
    ]

print(f"query set: {len(query_names)} package(s)")

# same indexing find_similar_packages (generate/util.py) uses internally, rebuilt
# here to keep this analysis script decoupled from generate.util's unrelated
# requests/jinja2 imports
build_system_index: dict[str, list[str]] = {}
dep_names_by_pkg: dict[str, frozenset] = {}
variant_names_by_pkg: dict[str, frozenset] = {}
for name, pkg in pkgs.items():
    for bs in pkg.build_systems:
        build_system_index.setdefault(bs, []).append(name)
    dep_names_by_pkg[name] = frozenset(d.pkg_name for d in pkg.dependencies)
    variant_names_by_pkg[name] = frozenset(v.name for v in pkg.variants)

CANDIDATES = build_system_index.get("cmake", [])


def top_k(target: str, w_d: float, w_b: float, k: int) -> tuple[frozenset, float]:
    """returns (set of top-k reference names, that top score) -- the score is
    used to flag degenerate zero-overlap cases where every weight ties"""
    deps = dep_names_by_pkg[target]
    variants = variant_names_by_pkg[target]

    scored = []
    for name in CANDIDATES:
        if name == target or target.lower() in name.lower():
            continue
        dep_score = len(deps & dep_names_by_pkg[name])
        var_score = len(variants & variant_names_by_pkg[name])
        scored.append((name, w_d * dep_score + w_b * var_score))

    scored.sort(key=lambda entry: entry[1], reverse=True)
    top = scored[:k]
    top_score = top[0][1] if top else 0.0
    return frozenset(name for name, _ in top), top_score


grid = [float(x) for x in ARGS.grid.split(",")]
baseline_wd = ARGS.baseline_wd
baseline_wb = 1 - baseline_wd

baseline_selections: dict[str, frozenset] = {}
zero_overlap = 0
for name in query_names:
    if name not in pkgs:
        continue
    selection, score = top_k(name, baseline_wd, baseline_wb, ARGS.k)
    baseline_selections[name] = selection
    if score == 0.0:
        zero_overlap += 1

skipped = len(query_names) - len(baseline_selections)
if skipped:
    print(f"warning: {skipped} query name(s) not found in --input, skipped")
if zero_overlap:
    print(
        f"note: {zero_overlap}/{len(baseline_selections)} query package(s) have zero "
        "dependency+variant overlap with every candidate at baseline weights -- their "
        "top-k selection is a tie broken by candidate order, not a weight effect, and "
        "will trivially 'agree' at every weight swept below"
    )

print(f"\nbaseline: w_d={baseline_wd:.2f}, w_b={baseline_wb:.2f}, k={ARGS.k}")
print(f"{'w_d':>5} {'w_b':>5} {'agree':>8} {'n':>5}")
for w_d in grid:
    w_b = 1 - w_d
    agree = sum(
        1
        for name, baseline_sel in baseline_selections.items()
        if top_k(name, w_d, w_b, ARGS.k)[0] == baseline_sel
    )
    n = len(baseline_selections)
    pct = 100 * agree / n if n else 0.0
    print(f"{w_d:>5.2f} {w_b:>5.2f} {pct:>7.1f}% {n:>5}")
