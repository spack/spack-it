import argparse
import functools
import time
from pathlib import Path

from generate.container import BuilderContainer, SpackError, Stage
from generate.util import ResultsStore

parser = argparse.ArgumentParser(
    description="install and setup breakpoints for spack packages to test"
)
parser.add_argument("dir", type=Path, help="directory containing package .py files")
parser.add_argument(
    "--results",
    type=str,
    help="where to store results of each run",
)
ARGS = parser.parse_args()
TEST_DIR = ARGS.dir

if not TEST_DIR.is_dir():
    raise ValueError(f"{TEST_DIR} is not a valid directory")

files = [f.name for f in TEST_DIR.glob("*.py")]


run_id = str(int(time.time()))
results = ResultsStore(run_id=run_id, filepath=ARGS.results, variant_testing=True)


for fname in files:
    scores = {
        "dependency_score": None,
        "variants_score": None,
        "variants_extras": None,
    }
    with BuilderContainer() as ctr:
        pkg_path = TEST_DIR / fname
        pkg_name = Path(fname).stem  # e.g., "libpressio-adios2"
        pkg_recipe = pkg_path.read_text()

        ctr.write_pkg(pkg_name, pkg_recipe)

        stages = [
            Stage("load", functools.partial(ctr.load_pkg, pkg_name)),
            Stage("concretize", functools.partial(ctr.concretize_pkg, pkg_name)),
            Stage("install", functools.partial(ctr.install_pkg, pkg_name)),
        ]

        for stage in stages:
            try:
                stage.action()
                status = stage.name

                # once the package has been loaded, we can get the scores for the package class
                if status == "load":
                    try:
                        scores["dependency_score"] = ctr.deps_score(pkg_name)
                    except Exception as exc:
                        print(f"error getting dep score: {exc}")
                    try:
                        scores["variants_score"], scores["variants_extras"] = (
                            ctr.cmake_args_score(pkg_name)
                        )
                    except Exception as exc:
                        print(f"error getting cmake score: {exc}")
                # TODO fill this in with num_tokens later
                results.log(pkg_name=pkg_name, status=status, **scores)
            except SpackError as exc:
                status = f"{stage.name}_fail"
                results.log(pkg_name=pkg_name, status=status, message=str(exc))

                break
