"""
Live-detects Autotools/Makefile candidate packages ahead of generation, so the
generation runs later can just read a pre-verified --pkg_list instead of
detecting (and possibly auto-skipping mismatches) inline during generate.py.

Unlike select_packages.py, this needs a real spack environment -- it does a
live fetch per candidate (same fetch_and_expand + detect_build_systems calls
generate.py makes at runtime) since Spack's own declared build_system variant
(pkg.build_systems, used only as a cheap pre-filter below) doesn't guarantee
what the live file-clue heuristic will find in the actual source tree.

Run with: spack python analysis/detect_buildsys.py --input data/packages-<ts>.pkl

Each detected package is also written to data/metadata_cache/ (the same cache
generate.py reads via load_extraction_cache), including the raw build-system
file dump generate.py's --raw_buildsys now reads for Autotools/Makefile, so
the later generation run gets a free cache hit and never re-fetches/re-detects.
"""

import argparse
import fnmatch
import pickle
import random
from pathlib import Path

from extraction.repository import (
    detect_build_systems,
    fetch_and_expand,
    format_raw_build_files,
    get_build_files,
)
from generate.util import load_ignore_patterns, load_pkg_list, save_extraction_cache

parser = argparse.ArgumentParser(
    description="fetch candidates and live-detect their build system, writing "
    "verified --pkg_list files for generate.py to read later"
)
parser.add_argument(
    "--input", required=True, help="pickled list of Packages (from extraction/dump.py)"
)
parser.add_argument("--ignore", default="data/ignore.txt")
parser.add_argument(
    "--exclude_pkg_list",
    default="data/pkg_list.txt",
    help="package names already used elsewhere (e.g. the main CMake study) to "
    "exclude from consideration, so the same name can't land in two studies",
)
parser.add_argument("-n", type=int, default=40, help="target count per bucket")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out_autotools", default="data/pkg_list_autotools.txt")
parser.add_argument("--out_makefile", default="data/pkg_list_makefile.txt")
ARGS = parser.parse_args()

with open(ARGS.input, "rb") as f:
    pkgs = pickle.load(f)

ignore_patterns = load_ignore_patterns(ARGS.ignore)
excluded_names = set(load_pkg_list(ARGS.exclude_pkg_list))


def is_ignored(name: str) -> bool:
    return any(fnmatch.fnmatch(name.lower(), pat.lower()) for pat in ignore_patterns)


# cheap pre-filter: Spack's own declared build_system variant. Only a
# prediction of what the live detector will find, but keeps this from live-
# fetching the whole corpus (mostly cmake) just to discover ~80 packages.
candidates = [
    pkg
    for pkg in pkgs.values()
    if not pkg.name.startswith("py-")
    and pkg.name not in excluded_names
    and not is_ignored(pkg.name)
    and ("autotools" in pkg.build_systems or "makefile" in pkg.build_systems)
]
print(f"candidates after pre-filter: {len(candidates)}")

rng = random.Random(ARGS.seed)
rng.shuffle(candidates)

autotools_found: list[str] = []
makefile_found: list[str] = []
fetch_failures: list[str] = []
other_detected: dict[str, int] = {}
attempted = 0

for pkg in candidates:
    if len(autotools_found) >= ARGS.n and len(makefile_found) >= ARGS.n:
        break

    attempted += 1
    stage, version = fetch_and_expand(pkg)
    if stage is None:
        fetch_failures.append(pkg.name)
        continue

    with stage:
        path = Path(stage.path) / "spack-src"
        build_sys, features = detect_build_systems(path)

        is_autotools = build_sys in ("autotools", "autoreconf")
        is_makefile = build_sys == "makefile"

        if not (
            (is_autotools and len(autotools_found) < ARGS.n)
            or (is_makefile and len(makefile_found) < ARGS.n)
        ):
            other_detected[build_sys] = other_detected.get(build_sys, 0) + 1
            continue

        # capture raw build files now, while the fetched source is still on disk --
        # matches generate.py's own dispatch so a later --raw_buildsys run gets a
        # real cache hit instead of a stale raw_files: null. cmake_parsed (the
        # AST-parsed form) has no autotools/makefile equivalent, stays None.
        raw_files = get_build_files(path, "autotools" if is_autotools else "makefile")
        raw_dump = format_raw_build_files(raw_files, repo_root=path) or None

    if is_autotools:
        autotools_found.append(pkg.name)
    else:
        makefile_found.append(pkg.name)

    save_extraction_cache(
        pkg.name,
        {
            "version": str(version),
            "build_sys": build_sys,
            "features": features,
            "cmake_parsed": None,
            "raw_files": raw_dump,
        },
    )
    print(
        f"[{attempted}] {pkg.name}: {build_sys}  "
        f"(autotools {len(autotools_found)}/{ARGS.n}, makefile {len(makefile_found)}/{ARGS.n})"
    )

print()
print(f"attempted: {attempted}, fetch failures: {len(fetch_failures)}")
print(f"autotools found: {len(autotools_found)}/{ARGS.n}")
print(f"makefile found: {len(makefile_found)}/{ARGS.n}")
if other_detected:
    print(f"other build systems seen along the way (bucket already full or off-target): {other_detected}")
if fetch_failures:
    print(f"fetch failures: {fetch_failures}")

Path(ARGS.out_autotools).write_text("\n".join(sorted(autotools_found)) + "\n")
Path(ARGS.out_makefile).write_text("\n".join(sorted(makefile_found)) + "\n")
print(f"wrote {ARGS.out_autotools}, {ARGS.out_makefile}")
