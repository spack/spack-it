import argparse
from pathlib import Path

from generate.container import BuilderContainer, SpackError

parser = argparse.ArgumentParser(
    description="install and setup breakpoints for spack packages to test"
)
parser.add_argument("dir", type=Path, help="directory containing package .py files")
ARGS = parser.parse_args()
TEST_DIR = ARGS.dir

if not TEST_DIR.is_dir():
    raise ValueError(f"{TEST_DIR} is not a valid directory")

files = [f.name for f in TEST_DIR.glob("*.py")]

with BuilderContainer() as ctr:
    for fname in files:
        pkg_path = TEST_DIR / fname
        pkg_name = Path(fname).stem  # e.g., "libpressio-adios2"
        pkg_recipe = pkg_path.read_text()

        try:
            ctr.write_pkg(pkg_name, pkg_recipe)
            ctr.load_pkg(pkg_name)
            ctr.install_pkg(pkg_name)
            print(f"{pkg_name} installed")
        except SpackError as exc:
            error = str(exc)
            print(f"{pkg_name} error: {error}")
    # sleep the container for a few hours so we can go in and debug
    ctr.exec("sleep 18000")
