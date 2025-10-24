import argparse
import functools
import pickle
import time

from extraction.package_schema import Package
from generate.container import BuilderContainer, SpackError, Stage
from generate.util import ResultsStore

####### CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", help="filename to a pickled file that has a list of Packages"
)
parser.add_argument(
    "--results",
    type=str,
    help="where to store results of each run",
)
parser.add_argument(
    "--success_status",
    type=str,
    default="concretize",
    choices=["load", "concretize", "install", "test"],
)
parser.add_argument(
    "--after",
    type=str,
    help="restart the ground truth eval after the specified package name",
)

ARGS = parser.parse_args()
# load pkgs into the global namespace for use in all packages
with open(ARGS.input, "rb") as f:
    pkgs = pickle.load(f)


run_id = str(int(time.time()))
results = ResultsStore(run_id=run_id, filepath=ARGS.results, ground_truth_eval=True)


def process_pkg(pkg: Package):
    status = None

    with BuilderContainer(namespace="builtin") as ctr:
        stages = [
            Stage("load", functools.partial(ctr.load_pkg, pkg.name)),
            Stage("concretize", functools.partial(ctr.concretize_pkg, pkg.name)),
            Stage("install", functools.partial(ctr.install_pkg, pkg.name)),
            Stage("test", functools.partial(ctr.test_pkg, pkg.name)),
        ]

        for stage in stages:
            try:
                stage.action()
                status = stage.name

                results.log(
                    pkg_name=pkg.name,
                    status=status,
                )

                if status == ARGS.success_status:
                    return
            except SpackError as exc:
                status = f"{stage.name}_fail"
                results.log(
                    pkg_name=pkg.name,
                    status=status,
                    message=str(exc),
                )

                break


def main():
    pkgs_list = list(pkgs.values())

    found = not bool(
        ARGS.after
    )  # true if ARGS.after is not set, so we start right away

    for pkg in pkgs_list:
        if not found:
            if pkg.name == ARGS.after:
                found = True
            continue  # skip until the match is seen (and skip the match itself)
        try:
            process_pkg(pkg)
        except Exception as exc:
            results.log(pkg_name=pkg.name, status="workflow_fail", message=str(exc))


if __name__ == "__main__":
    main()
