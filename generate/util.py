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


def get_random_recipe(
    pkgs: list[Package], build_system: str = None, avoid: str = None
) -> tuple[str, str]:
    filtered_pkgs = list(pkgs.values())

    # Filter by build system if provided
    if build_system is not None:
        filtered_pkgs = [
            pkg for pkg in filtered_pkgs if build_system in pkg.build_systems
        ]

    # Filter out the package to avoid
    if avoid is not None:
        filtered_pkgs = [pkg for pkg in filtered_pkgs if pkg.name != avoid]

    random_pkg = random.choice(filtered_pkgs)
    return random_pkg.name, random_pkg.recipe


# TEMPLATE/PROMPT HANDLING
def render_template(template: str, params: dict) -> str:
    env = Environment(
        loader=FileSystemLoader(os.getenv("TEMPLATE_DIR")),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template_file = f"{template}.txt"

    try:
        template = env.get_template(template_file)
    except TemplateNotFound:
        raise ValueError(
            f"Template '{template}' does not exist in {os.getenv('TEMPLATE_DIR')}"
        )

    return template.render(**params)


# LLM HANDLING
class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period  # in seconds
        self.calls = deque()

    def wait(self):
        now = time.time()
        while self.calls and now - self.calls[0] > self.period:
            self.calls.popleft()
        if len(self.calls) >= self.max_calls:
            sleep_time = self.period - (now - self.calls[0])
            print(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
            self.calls.popleft()
        self.calls.append(time.time())


def call_llm(prompt: str, model: str) -> tuple[int, str]:
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
        print(response.text)
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
