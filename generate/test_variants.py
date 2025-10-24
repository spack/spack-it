import argparse
import ast
import functools
import random
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
    default="results/ground.jsonl",
    help="where to store results of each run",
)
ARGS = parser.parse_args()
TEST_DIR = ARGS.dir

if not TEST_DIR.is_dir():
    raise ValueError(f"{TEST_DIR} is not a valid directory")

files = [f.name for f in TEST_DIR.glob("*.py")]


run_id = str(int(time.time()))
results = ResultsStore(run_id=run_id, filepath=ARGS.results, variant_testing=True)


def sample_variants(source_code: str, n: int = 5):
    """
    Parse a Spack package definition and return a dictionary
    containing N randomly selected variants (only those with
    explicit boolean defaults) and their default values.

    Args:
        source_code (str): Python source code as a string.
        n (int): Number of variants to sample (default: 5).

    Returns:
        dict[str, bool]: Mapping of variant names to default values (True/False only).
    """
    tree = ast.parse(source_code)
    variants: dict[str, bool] = {}

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "variant"
        ):
            name = None
            default_value = None

            # First argument: variant name
            if node.args and isinstance(node.args[0], ast.Constant):
                name = node.args[0].value

            # Keyword: default (must be True/False)
            for kw in node.keywords:
                if kw.arg == "default":
                    v = kw.value
                    if isinstance(v, ast.Constant) and isinstance(v.value, bool):
                        default_value = v.value
                    elif isinstance(v, ast.Name) and v.id in ("True", "False"):
                        default_value = v.id == "True"
                    break

            # Only keep variants with explicit boolean defaults
            if name is not None and isinstance(default_value, bool):
                variants[name] = default_value

    # Randomly sample up to N variants
    sample_size = min(n, len(variants))
    sampled_keys = random.sample(list(variants.keys()), sample_size)
    return {k: variants[k] for k in sampled_keys}


for fname in files:
    with BuilderContainer() as ctr:
        pkg_path = TEST_DIR / fname
        pkg_name = Path(fname).stem  # e.g., "libpressio-adios2"
        pkg_recipe = pkg_path.read_text()

        ctr.write_pkg(pkg_name, pkg_recipe)

        for var, var_default in sample_variants(pkg_recipe).items():
            # Flip the default:
            # - default False => enable with +var
            # - default True  => disable with ~var
            modifier = "+" if (var_default is False) else "~"
            spec = f"{pkg_name} {modifier}{var}"

            stages = [
                Stage("load", functools.partial(ctr.load_pkg, pkg_name)),
                Stage("concretize", functools.partial(ctr.concretize_pkg, spec)),
                Stage("install", functools.partial(ctr.install_pkg, spec)),
            ]

            for stage in stages:
                try:
                    stage.action()
                    status = stage.name

                    results.log(pkg_name=pkg_name, status=status, spec=spec)
                except SpackError as exc:
                    status = f"{stage.name}_fail"
                    results.log(
                        pkg_name=pkg_name, status=status, message=str(exc), spec=spec
                    )

                    break
