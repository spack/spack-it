# helper functions that extract information from a repository's files

import re
from pathlib import Path
from typing import Union

import spack

import extraction.package_schema
from extraction.cmake import EXCLUDE_DIRS


def get_build_files(
    repo_root, build_system: str, depth: Union[int, str] = "shallowest"
):
    """
    Find and return the contents of build system files in a repository.

    Args:
        repo_root (str or Path): Path to the root of the repository.
        build_system (str): Required. One of "cmake", "make", "bazel", "meson", or "all".
        depth (int or str): Optional. One of:
            - "shallowest": Only return files at the minimum depth (default).
            - -1: Return all matching files regardless of depth.
            - int >= 0: Return files at depth <= given value.

    Returns:
        dict[str, str]: Mapping of absolute file paths to file contents.
    """
    build_patterns = {
        "cmake": ["**/CMakeLists.txt", "**/*.cmake"],
        # "make": ["**/Makefile"],
        # "bazel": ["**/BUILD", "**/BUILD.bazel"],
        # "meson": ["**/meson.build"],
        # TODO extend here as needed
    }

    if build_system == "all":
        patterns = [p for plist in build_patterns.values() for p in plist]
    else:
        patterns = build_patterns.get(build_system.lower())
        if not patterns:
            raise ValueError(f"Unsupported build system: {build_system!r}")

    repo_root = Path(repo_root).resolve()
    build_files_paths = []
    excluded_dirs = {"test", "tests", "example", "examples", "samples", "demo", "demos"}

    for pattern in patterns:
        for path in repo_root.rglob(pattern):
            if path.is_file():
                rel_parts = path.relative_to(repo_root).parts
                if any(part.lower() in excluded_dirs for part in rel_parts):
                    continue
                build_files_paths.append(path.resolve())

    if not build_files_paths:
        return {}

    path_depths = {
        path: len(path.relative_to(repo_root).parts) for path in build_files_paths
    }

    if depth == "shallowest":
        min_depth = min(path_depths.values())
        selected_paths = [p for p, d in path_depths.items() if d == min_depth]
    elif isinstance(depth, int):
        if depth == -1:
            selected_paths = list(path_depths.keys())
        else:
            selected_paths = [p for p, d in path_depths.items() if d <= depth]
    else:
        raise ValueError(f"Unsupported depth option: {depth!r}")

    build_files = {}
    for path in selected_paths:
        try:
            content = path.read_text(errors="ignore")
            build_files[str(path)] = content
        except Exception as e:
            print(f"Failed to read {path}: {e}")

    return build_files


def build_tree(
    path: Path,
    prefix: str = "",
    max_depth: int = -1,
    current_depth: int = 0,
) -> str:
    """
    Create a simple tree view of the files and folders in a directory.

    Args:
        path: Root directory to render.
        prefix: Indentation (managed internally during recursion).
        max_depth: -1 for no limit. 0 prints only the immediate entries under `path`.
        current_depth: Tracks current depth (internal).
    """
    # If we've gone deeper than allowed, stop.
    if max_depth != -1 and current_depth > max_depth:
        return ""

    # Filter out excluded names
    try:
        entries = sorted(
            [p for p in path.iterdir() if p.name not in EXCLUDE_DIRS],
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except PermissionError:
        return f"{prefix}└── [Permission denied]\n"
    except FileNotFoundError:
        return f"{prefix}└── [Not found]\n"

    tree_str = ""
    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        tree_str += f"{prefix}{connector}{entry.name}\n"

        if entry.is_dir():
            # Only descend if we haven't reached the depth limit
            if max_depth == -1 or current_depth < max_depth:
                extension = "    " if i == len(entries) - 1 else "│   "
                tree_str += build_tree(
                    entry,
                    prefix=prefix + extension,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                )

    return tree_str


def fetch_and_expand(
    pkg,
) -> tuple[spack.stage.Stage, extraction.package_schema.Version]:
    """fetches a package and expands it into a temporary directory
    returns a spack stage that can be used to call other extraction functions
    """

    version = get_best_version(pkg)

    try:
        stage = spack.stage.Stage(version.fetcher)
        stage.create()
        version.fetcher.fetch()
        version.fetcher.expand()
        return stage, version
    except Exception as e:
        print(version.fetcher, e)
        stage.destroy()
        return None, None


def get_best_version(pkg) -> extraction.package_schema.Version:
    """gets 'highest-priority' version for our purposes of evaluation"""
    # first gets the preferred version
    # then gets the next package that isn't a develop release or deprecated
    # finally gets the first package in the list (assumed to be ordered)
    selected_version = next(
        (v for v in pkg.versions if v.preferred),
        next(
            (
                v
                for v in pkg.versions
                if not v.isdevelop and not v.deprecated and not v.is_prerelease
            ),
            pkg.versions[0] if pkg.versions else None,
        ),
    )

    selected_version.submodules = False

    if selected_version is None or not selected_version.fetcher:
        print(f"could not find suitable version or fetcher for {pkg.name}")
        return

    return selected_version


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
