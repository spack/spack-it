import json
import os
import random
import shutil
import socket
import subprocess
import tempfile
import time
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import requests
import spack
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from extraction.package_schema import Package

# in the right order
STATUSES = [
    "load",
    "load_fail",
    "concretize",
    "concretize_fail",
    "install",
    "install_fail",
    "test",
    "test_fail",
    "workflow_fail",
]


class GenerateException(Exception):
    pass


EXTRACTION_CACHE_DIR = Path("data/metadata_cache")


def load_extraction_cache(pkg_name: str) -> dict | None:
    """
    Returns the cached {version, build_sys, features, cmake_parsed} for a
    package, or None if nothing is cached yet. This is the fetch + build-system
    detection + CMake parse result, which is identical across every ablation
    config/model for a given package, so it's cached once regardless of how
    many configs reuse it. Does NOT include CMake distillation -- that's an
    LLM call whose output depends on the model, so it's always computed fresh.

    Caveat: keyed on pkg_name only, so if `--input` is regenerated with
    different version selections for the same package names, stale entries
    won't be detected -- clear `data/metadata_cache/` when switching inputs.
    """
    path = EXTRACTION_CACHE_DIR / f"{pkg_name}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def save_extraction_cache(pkg_name: str, data: dict) -> None:
    EXTRACTION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = EXTRACTION_CACHE_DIR / f"{pkg_name}.json"
    with path.open("w") as f:
        json.dump(data, f)


def _read_lines(path: str, missing_msg: str) -> list[str]:
    """shared reader for the one-name/pattern-per-line files below;
    '#' comments and blank lines are skipped, missing files warn and return []"""
    file = Path(path)
    if not file.exists():
        print(missing_msg.format(path=path))
        return []

    lines = []
    for line in file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def load_ignore_patterns(path: str) -> list[str]:
    """
    Reads fnmatch patterns (one per line) from an ignore file, e.g.:
        dealii
        roc*
        # comments and blank lines are skipped
    Returns [] (with a warning) rather than crashing a whole run if the file
    is missing -- an absent ignore file just means nothing gets excluded.
    """
    return _read_lines(
        path, "warning: ignore file {path} not found, no packages will be excluded"
    )


def load_pkg_list(path: str) -> list[str]:
    """
    Reads exact package names (one per line) from a task-list file, used to fix
    the exact set of packages a run operates on instead of randomly sampling
    --samples packages from --input via --seed.
    Returns [] (with a warning) rather than crashing a whole run if the file
    is missing.
    """
    return _read_lines(
        path, "warning: pkg_list file {path} not found, no packages will be selected"
    )


def load_completed_pkgs(results_path: str) -> set[str]:
    """
    Scan an existing results jsonl and return the set of package names that
    already made it through `generate_handler` at least once (i.e., have any
    logged status other than "workflow_fail", which is only written when
    `pipeline()` catches a `GenerateException` before a package is counted
    toward `--samples`). Used to resume an interrupted/incomplete run without
    re-spending LLM calls and build attempts on packages already given their
    full `--max_attempts` budget.
    """
    path = Path(results_path)
    if not path.exists():
        return set()

    completed: set[str] = set()
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
            if name and row.get("status") != "workflow_fail":
                completed.add(name)

    return completed


class ArtifactStore:
    """
    Lazily creates a unique run directory once, then
    lets you drop files in it as soon as you have them.
    """

    def __init__(self, run_id: str, root: str = "artifacts"):
        self.run_id = run_id

        self.base = Path(root) / self.run_id
        self.base.mkdir(parents=True, exist_ok=True)

    def save(self, subject: str, name: str, content: str):
        """
        Write one file immediately.
        `subject` plays the same role as your pkg_name folder.
        """
        subject_dir = self.base / subject
        subject_dir.mkdir(parents=True, exist_ok=True)

        file_path = subject_dir / name
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return file_path  # handy for logs / tests / REPL exploration


class ResultsStore:
    def __init__(
        self,
        run_id: str,
        filepath: str,
        model: str = None,
        max_attempts: int = None,
        ground_truth_eval: bool = False,
        # FEATURES
        raw_buildsys: bool = False,
        distilled_cmake: bool = False,
        tree: bool = False,
        random_recipe: bool = False,
        random_buildsys_recipe: bool = False,
        similar_recipe: bool = False,
        audit: bool = False,
        variant_testing: bool = False,
        rag: bool = False,
    ):
        self.filepath = Path(filepath)
        self.run_id = run_id
        self.model = model
        self.max_attempts = max_attempts

        host_platform = spack.platforms.host()
        host_os = host_platform.default_operating_system()
        host_target = host_platform.default_target()
        self.arch = f"{host_platform}-{host_os}-{host_target}"
        self.hostname = socket.gethostname()
        self.ground_truth_eval = ground_truth_eval

        self.features = {
            "raw_buildsys": raw_buildsys,
            "distilled_cmake": distilled_cmake,
            "tree": tree,
            # if there is no inclusion of a recipe, that is detected
            "no_recipe": not any(
                [
                    random_recipe,
                    random_buildsys_recipe,
                    similar_recipe,
                ]
            ),
            "random_recipe": random_recipe,
            "random_buildsys_recipe": random_buildsys_recipe,
            "similar_recipe": similar_recipe,
            "audit": audit,
            "variant_testing": variant_testing,
            "rag": rag,
        }

    def log(
        self,
        # RUN METADATA
        pkg_name: str,
        status: str,
        attempt_num: int = None,
        message: str = None,
        references: dict = {},
        num_tokens: int = None,
        spec: str = None,
        # SCORES
        dependency_score: float = None,
        variants_score: float = None,
        variants_extras: int = None,
    ):
        if status not in STATUSES:
            raise ValueError(f"status {status} not in list of statuses")

        run_entry = {
            "run_id": self.run_id,
            "timestamp": time.time(),
            "arch": self.arch,
            "hostname": self.hostname,
            "model": self.model,
            "ground_truth_eval": self.ground_truth_eval,
            "max_attempts": self.max_attempts,
            "pkg_name": pkg_name,
            "spec": spec,
            "status": status,
            "attempt_num": attempt_num,
            "message": message,
            "num_tokens": num_tokens,
            # SCORES
            "dependency_score": dependency_score,
            "variants_score": variants_score,
            "variants_extras": variants_extras,
            **self.features,
        }

        for name, ref in references.items():
            # example: ref_random_buildsys: pkg_name
            run_entry[f"ref_{name}"] = ref["pkg"]

        with self.filepath.open("a") as f:
            # this is a jsonl output (one JSON object per line)
            f.write(json.dumps(run_entry) + "\n")


# `pkgs` is loaded once per process and never mutated afterwards, so the indexes
# below are cached per `id(pkgs)` instead of being rebuilt on every lookup call.
# Across a few hundred package runs, find_similar_packages/get_random_recipe are
# each called repeatedly against the same multi-thousand-entry corpus, so
# rescanning it and rebuilding per-package sets every time is pure waste.
_build_system_index_cache: dict[int, dict[str, list[str]]] = {}
_dep_variant_names_cache: dict[int, tuple[dict[str, frozenset], dict[str, frozenset]]] = {}


def _get_build_system_index(pkgs: dict[str, Package]) -> dict[str, list[str]]:
    key = id(pkgs)
    if key not in _build_system_index_cache:
        index: dict[str, list[str]] = {}
        for name, pkg in pkgs.items():
            for build_system in pkg.build_systems:
                index.setdefault(build_system, []).append(name)
        _build_system_index_cache[key] = index
    return _build_system_index_cache[key]


def _get_dep_variant_names(
    pkgs: dict[str, Package],
) -> tuple[dict[str, frozenset], dict[str, frozenset]]:
    key = id(pkgs)
    if key not in _dep_variant_names_cache:
        dep_names = {}
        variant_names = {}
        for name, pkg in pkgs.items():
            dep_names[name] = frozenset(dep.pkg_name for dep in pkg.dependencies)
            variant_names[name] = frozenset(var.name for var in pkg.variants)
        _dep_variant_names_cache[key] = (dep_names, variant_names)
    return _dep_variant_names_cache[key]


def get_random_recipe(
    pkgs: dict[str, Package], build_system: str = None, avoid: str = None
) -> tuple[str, str]:
    if build_system is not None:
        names = _get_build_system_index(pkgs).get(build_system, [])
        filtered_pkgs = [pkgs[name] for name in names]
    else:
        filtered_pkgs = list(pkgs.values())

    # Filter out the package to avoid
    if avoid is not None:
        filtered_pkgs = [pkg for pkg in filtered_pkgs if pkg.name != avoid]

    random_pkg = random.choice(filtered_pkgs)
    return random_pkg.name, random_pkg.recipe


def find_similar_packages(
    pkgs: dict[str, Package],
    pkg_name: str,
    build_system: str,
    dependencies: list[str],
    variants: list[str],
    num_similar_refs: int,
) -> list[tuple[str, str, float]]:
    """
    Finds the packages most similar to `pkg_name` by symbolic overlap of
    dependencies and variants, restricted to packages sharing `build_system`.

    Replaces the former Neo4j/Cypher GraphRAG lookup: since `pkgs` already
    holds every field the old graph query touched, the same
    dep_score/var_score/total_score computation can run directly in memory.

    returns a list of (name, recipe, total_score) tuples, sorted by
    descending total_score, of at most `num_similar_refs` entries.
    """
    dependencies = set(dependencies)
    variants = set(variants)

    dep_names_by_pkg, variant_names_by_pkg = _get_dep_variant_names(pkgs)
    candidate_names = _get_build_system_index(pkgs).get(build_system, [])

    scored = []
    for name in candidate_names:
        if name == pkg_name or pkg_name.lower() in name.lower():
            continue

        dep_score = len(dependencies & dep_names_by_pkg[name])
        var_score = len(variants & variant_names_by_pkg[name])
        total_score = 0.6 * dep_score + 0.4 * var_score

        scored.append((name, pkgs[name].recipe, total_score))

    scored.sort(key=lambda entry: entry[2], reverse=True)
    return scored[:num_similar_refs]


# TEMPLATE/PROMPT HANDLING
# render_template is called multiple times per generation attempt, so the
# Environment (and its template parse cache) is built once per TEMPLATE_DIR
# instead of re-scanning the filesystem and re-parsing templates every call.
_template_env_cache: dict[str, Environment] = {}


def _get_template_env(template_dir: str) -> Environment:
    if template_dir not in _template_env_cache:
        _template_env_cache[template_dir] = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
    return _template_env_cache[template_dir]


def render_template(template: str, params: dict) -> str:
    template_dir = os.getenv("TEMPLATE_DIR")
    env = _get_template_env(template_dir)

    template_file = f"{template}.txt"

    try:
        template = env.get_template(template_file)
    except TemplateNotFound:
        raise ValueError(f"Template '{template}' does not exist in {template_dir}")

    return template.render(**params)


# LLM HANDLING
class RateLimiter:
    def __init__(self, max_calls, period, log=print):
        self.max_calls = max_calls
        self.period = period  # in seconds
        self.calls = deque()
        self._log = log

    def wait(self):
        now = time.time()
        while self.calls and now - self.calls[0] > self.period:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            self._log(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
            self.calls.popleft()
        self.calls.append(time.time())


def call_llm(prompt: str, model: str, log=print) -> tuple[int, str]:
    # rturns the number of tokens and the response
    headers = {
        "Content-Type": "application/json",
    }

    api_key = os.getenv("LLM_API_KEY")

    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }

    response = requests.post(
        f"{os.getenv('LLM_API_URL')}/v1/chat/completions",
        headers=headers,
        json=data,
    )

    if response.status_code != 200:
        log(response.text)
        raise GenerateException(f"model http error: {response.status_code}")

    res = response.json()
    return res["usage"]["prompt_tokens"], res["choices"][0]["message"]["content"]


def extract_distilled_cmake(distilled):
    # leverages cmake_distilled to extract the detected variants and dependencies...
    detected_dependencies = []
    detected_variants = []
    in_summary = False
    # it's probably more performant to expect that they would only be on the bottom three lines
    # but don't want to run into parsing issues yet...
    for line in distilled.splitlines():
        if line.strip() == "SUMMARY":
            in_summary = True
            continue
        if in_summary:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "variants":
                detected_variants = tokens[1:]
            elif tokens[0] == "dependencies":
                detected_dependencies = tokens[1:]

    return detected_dependencies, detected_variants


@dataclass(kw_only=True)
class GitPackage:
    name: str
    url: str
    branch: str


def load_git_repos():
    # repos = {
    #     "libigl": "https://github.com/libigl/libigl.git",
    #     "draco": "https://github.com/google/draco.git",
    #     "lethe": "https://github.com/chaos-polymtl/lethe.git",
    #     "cryptominisat": "https://github.com/msoos/cryptominisat.git",
    #     "minisat": "https://github.com/niklasso/minisat.git",
    #     "febio": "https://github.com/febiosoftware/FEBio.git",
    #     "stp": "https://github.com/stp/stp.git",
    #     "fastchem": "https://github.com/NewStrangeWorlds/FastChem.git",
    #     "incompact3d": "https://github.com/xcompact3d/Incompact3d.git",
    #     "opensees": "https://github.com/OpenSees/OpenSees.git",
    #     "opm-simulators": "https://github.com/OPM/opm-simulators.git",
    #     "bpftrace": "https://github.com/bpftrace/bpftrace.git",
    #     "actor-framework": "https://github.com/actor-framework/actor-framework.git",
    #     "meshlib": "https://github.com/MeshInspector/MeshLib.git",
    #     "meshoptimizer": "https://github.com/zeux/meshoptimizer.git",
    #     "highs": "https://github.com/ERGO-Code/HiGHS.git",
    #     "libmpc": "https://github.com/nicolapiccinelli/libmpc.git",
    #     "umt": "https://github.com/LLNL/UMT.git",
    #     "units": "https://github.com/LLNL/units.git",
    #     "scaleuprom": "https://github.com/LLNL/scaleupROM.git",
    #     "zero-rk": "https://github.com/LLNL/zero-rk.git",
    #     "smith": "https://github.com/LLNL/smith.git",
    #     "spheral": "https://github.com/LLNL/spheral.git",
    #     "continuationsolvers": "https://github.com/LLNL/ContinuationSolvers.git",
    #     "tribol": "https://github.com/LLNL/Tribol.git",
    #     "saltatlas": "https://github.com/LLNL/saltatlas.git",
    #     "ygm": "https://github.com/LLNL/ygm.git",
    #     "exaconstit": "https://github.com/LLNL/ExaConstit.git",
    #     "mgmol": "https://github.com/LLNL/mgmol.git",
    #     "polyclipper": "https://github.com/LLNL/PolyClipper.git",
    #     "exadis": "https://github.com/LLNL/exadis.git",
    #     "matred": "https://github.com/LLNL/matred.git",
    #     "krowkee": "https://github.com/LLNL/krowkee.git",
    #     "parelag": "https://github.com/LLNL/parelag.git",
    #     "ampe": "https://github.com/LLNL/AMPE.git",
    #     "dr-evt": "https://github.com/LLNL/dr_evt.git",
    #     "snls": "https://github.com/LLNL/SNLS.git",
    #     "havoqgt": "https://github.com/LLNL/havoqgt.git",
    #     "polytope": "https://github.com/LLNL/polytope.git",
    #     "gridkit": "https://github.com/ORNL/GridKit.git",
    #     "smoothg": "https://github.com/LLNL/smoothG.git",
    #     "tripoll": "https://github.com/LLNL/tripoll.git",
    #     "spify": "https://github.com/LLNL/spify.git",
    #     "psuade": "https://github.com/LLNL/psuade.git",
    #     "adapt": "https://github.com/LLNL/ADAPT.git",
    #     "psuade-lite": "https://github.com/LLNL/psuade-lite.git",
    #     "perroht": "https://github.com/LLNL/Perroht.git",
    # }

    repos = {
        "draco": ("https://github.com/google/draco.git", "main"),
        "fastchem": ("https://github.com/NewStrangeWorlds/FastChem.git", "master"),
        "actor-framework": (
            "https://github.com/actor-framework/actor-framework.git",
            "main",
        ),
        "units": ("https://github.com/LLNL/units.git", "main"),
        "continuationsolvers": (
            "https://github.com/LLNL/ContinuationSolvers.git",
            "master",
        ),
        "ygm": ("https://github.com/LLNL/ygm.git", "master"),
        "mgmol": ("https://github.com/LLNL/mgmol.git", "release"),
        "polyclipper": ("https://github.com/LLNL/PolyClipper.git", "master"),
        "matred": ("https://github.com/LLNL/matred.git", "master"),
        "ampe": ("https://github.com/LLNL/AMPE.git", "release"),
        "tripoll": ("https://github.com/LLNL/tripoll.git", "main"),
        "psuade": ("https://github.com/LLNL/psuade.git", "3.0.0"),
        "adapt": ("https://github.com/LLNL/ADAPT.git", "release"),
        "psuade-lite": ("https://github.com/LLNL/psuade-lite.git", "main"),
        "perroht": ("https://github.com/LLNL/Perroht.git", "main"),
    }

    return [
        GitPackage(name=name, url=url, branch=branch)
        for name, (url, branch) in repos.items()
    ]


class GitCloneStage(AbstractContextManager):
    """Minimal context manager that shallow-clones a Git repo into a temp dir."""

    def __init__(self, url, depth=1):
        self.url = url
        self.depth = depth
        self.path = None

    def __enter__(self):
        # Create a temporary directory for this stage
        self.path = Path(tempfile.mkdtemp(prefix="git-stage-"))

        # Shallow clone (depth=1)
        result = subprocess.run(
            ["git", "clone", "--depth", str(self.depth), self.url, str(self.path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise GenerateException(f"git clone failed:\n{result.stderr.strip()}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up temporary directory after use
        if self.path and self.path.exists():
            shutil.rmtree(self.path, ignore_errors=True)
