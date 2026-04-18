import argparse
import os
import re
from pathlib import Path


def detect_build_systems(path) -> tuple[str, list]:
    # TODO is there any way to improve this to get more features

    # pulled from https://github.com/spack/spack/blob/4c87f02a8dfe870f6477b70a2624da098a013e16/lib/spack/spack/cmd/create.py#L783-L836
    # patterns to detect build systems and features
    clues = [
        # cmake
        (re.compile(r"/CMakeLists\.txt$"), "cmake"),
        (re.compile(r"/.*\.cmake$"), "cmake"),
        # autotools
        (re.compile(r"/configure$"), "autotools"),
        (re.compile(r"/config\.(guess|sub)$"), "autotools"),
        (re.compile(r"/install-sh$"), "autotools"),
        (re.compile(r"/missing$"), "autotools"),
        # autoreconf
        (re.compile(r"/configure\.(in|ac)$"), "autoreconf"),
        (re.compile(r"/Makefile\.am$"), "autoreconf"),
        # make
        (re.compile(r"/(GNU)?[Mm]akefile$"), "makefile"),
        (re.compile(r"/makefile$"), "makefile"),
        # python
        (re.compile(r"/pyproject\.toml$"), "python"),
        (re.compile(r"/setup\.(py|cfg)$"), "python"),
        (re.compile(r"/requirements\.txt$"), "python"),
        (re.compile(r"/Pipfile(\.lock)?$"), "python"),
        # cuda
        (re.compile(r"/.*\.(cu|cuh)$"), "cuda"),
    ]

    primary = None
    features = []

    # scan files under the directory
    for file in path.rglob("*"):
        if file.is_file():
            # normalize path to unix style
            rel_path = "/" + str(file.relative_to(path)).replace("\\", "/")
            # test each clue against the relative path
            for pattern, system in clues:
                if pattern.search(rel_path):
                    if system in {"python", "cuda"}:
                        if system not in features:
                            features.append(system)
                    elif primary is None:
                        primary = system

    return primary, features


def _expand_path(path: str) -> Path:
    return Path(os.path.expandvars(path)).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect build systems in a directory")
    parser.add_argument("path", help="Directory to scan")
    args = parser.parse_args()

    expanded_path = _expand_path(args.path)
    primary, features = detect_build_systems(expanded_path)
    print((primary, features))


if __name__ == "__main__":
    main()
