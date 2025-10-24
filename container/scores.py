import argparse

import spack

from extraction.package import get_pkg_obj
from generate.eval import cmake_args_score, deps_score

parser = argparse.ArgumentParser()
parser.add_argument("--pkg_name", help="package name to analyze")
parser.add_argument("--score", choices=["cmake", "deps"])
ARGS = parser.parse_args()

builtin = spack.repo.PATH.get_repo("builtin")
experiment = spack.repo.PATH.get_repo("experiment")

orig = get_pkg_obj(ARGS.pkg_name, builtin)
new = get_pkg_obj(ARGS.pkg_name, experiment)

if ARGS.score == "cmake":
    score, extra = cmake_args_score(orig, new)
    print(f"{score},{extra}")
if ARGS.score == "deps":
    print(deps_score(orig, new))
