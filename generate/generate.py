import argparse
import fnmatch
import functools
import pickle
import random
import time
from pathlib import Path

from extraction.cmake import parse_repo as parse_cmake
from extraction.package_schema import Package
from extraction.repository import (
    build_tree,
    detect_build_systems,
    fetch_and_expand,
    format_raw_build_files,
    get_build_files,
)
from generate.container import BuilderContainer, SpackError, Stage
from generate.ui import ProgressUI
from generate.util import (
    ArtifactStore,
    GenerateException,
    GitCloneStage,
    RateLimiter,
    ResultsStore,
    call_llm,
    extract_distilled_cmake,
    find_similar_packages,
    get_random_recipe,
    load_completed_pkgs,
    load_extraction_cache,
    load_git_repos,
    load_ignore_patterns,
    load_pkg_list,
    render_template,
    save_extraction_cache,
)
#from rag.retrieve import load_index_from_cache, retrieve_chunks

####### CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", help="filename to a pickled file that has a list of Packages"
)
parser.add_argument(
    "--samples",
    type=int,
    default=None,
    help="number of experiments to run; defaults to 5, or to the length of "
    "--pkg_list when that's given",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="fix the package shuffle order for reproducible/consistent task sets across "
    "ablation runs; ignored when --pkg_list is given, since that already fixes the "
    "exact task set",
)
parser.add_argument(
    "--pkg_list",
    type=str,
    default=None,
    help="path to a file of package names (one per line, # comments allowed) to use "
    "as the exact task set, instead of randomly sampling --samples packages from "
    "--input via --seed",
)
parser.add_argument(
    "--model",
    type=str,
    help="the model to use",
)
parser.add_argument(
    "--results",
    type=str,
    default=None,
    help="where to store results of each run; if omitted, derived automatically "
    "from --model and the active ablation flags under --results_dir",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="results/rerun_2026",
    help="directory used for the auto-derived --results filename",
)
parser.add_argument(
    "--max_attempts",
    type=int,
    default=5,
    help="the max tries to attempt when generating a package",
)
parser.add_argument(
    "--success_status",
    type=str,
    default="concretize",
    choices=["load", "concretize", "install", "test"],
)
parser.add_argument(
    "--audit",
    action="store_true",
    help="enable spack package audit checks between each step",
)
parser.add_argument(
    "--rag", action="store_true", help="use RAG as a retrieval technique"
)
parser.add_argument(
    "--git_repos", action="store_true", help="generate packages for a list of git repos"
)
parser.add_argument(
    "--git_repos_file",
    type=str,
    default=None,
    help="path to a 'name,url,branch' CSV file (one repo per line, '#' comments "
    "and blank lines skipped) to use as the --git_repos repo list instead of the "
    "hardcoded default in generate/util.py:load_git_repos(), e.g. "
    "data/contamination_git_repos.txt. Ignored unless --git_repos is also set.",
)

# CATEGORY: build system information
parser.add_argument(
    "--raw_buildsys",
    action="store_true",
    help="include raw build system files in the LLM prompt",
)
parser.add_argument(
    "--distilled_cmake",
    action="store_true",
    help="distill cmake metadata using an LLM rather than giving it raw JSON",
)

# CATEGORY: directory tree information
parser.add_argument(
    "--tree",
    action="store_true",
    help="include directory tree in the LLM prompt",
)
parser.add_argument(
    "--tree_depth",
    type=int,
    default=3,
    help="if the directory tree is included, choose the depth. -1 means all levels",
)

# CATEGORY: reference recipe inclusion
parser.add_argument(
    "--random_recipe",
    type=int,
    default=0,
    help="include random spack recipe in the LLM prompt",
)
parser.add_argument(
    "--random_buildsys_recipe",
    type=int,
    default=0,
    help="include random spack recipe in the LLM prompt (from the same buildsystem as the target repo)",
)
parser.add_argument(
    "--similar_recipe",
    type=int,
    default=0,
    help="include similar spack recipe in the LLM prompt (symbolic overlap of "
    "dependencies/variants against the local package corpus; requires --distilled_cmake)",
)
parser.add_argument(
    "--baseline",
    action="store_true",
    help="option for experiment where we let the llm do all the research with no parsed data included",
)
parser.add_argument(
    "--target_buildsys",
    type=str,
    default=None,
    help="only continue generation for packages whose live-detected build system "
    "(detect_build_systems, requires a real fetch, so this can't be pre-verified from "
    "the input pickle alone) matches this value; a mismatch is treated as workflow_fail "
    "so pipeline() auto-skips it and keeps sampling toward --samples. 'autotools' also "
    "accepts a live detection of 'autoreconf' (same GNU Autotools family, no separate "
    "parser distinction). Also widens pipeline()'s default eligibility filter (which "
    "otherwise only samples packages with 'cmake' in pkg.build_systems) to this value.",
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="skip packages already completed in --results (per load_completed_pkgs) and "
    "only run enough additional packages to reach --samples total",
)
parser.add_argument(
    "--ignore",
    type=str,
    default="data/ignore.txt",
    help="path to a file of package names/glob patterns (fnmatch, case-insensitive, "
    "one per line, # comments allowed) to always exclude from --samples selection, "
    "regardless of --resume/--seed",
)

ARGS = parser.parse_args()

if ARGS.similar_recipe and not ARGS.distilled_cmake:
    parser.error("--similar_recipe requires --distilled_cmake")

IGNORE_PATTERNS = load_ignore_patterns(ARGS.ignore)
PKG_LIST = load_pkg_list(ARGS.pkg_list) if ARGS.pkg_list else None

if ARGS.samples is None:
    ARGS.samples = len(PKG_LIST) if PKG_LIST is not None else 5


def _target_buildsys_mismatch(detected: str) -> bool:
    """True if --target_buildsys is set and `detected` doesn't satisfy it. 'autotools'
    also accepts a live detection of 'autoreconf' (same GNU Autotools family)."""
    if not ARGS.target_buildsys:
        return False
    if detected == ARGS.target_buildsys:
        return False
    if ARGS.target_buildsys == "autotools" and detected == "autoreconf":
        return False
    return True


def _config_slug() -> str:
    """matches the none/none_raw/similar1_distilled/... naming used across the
    ablation matrix, derived from the active flags rather than typed by hand"""
    if ARGS.similar_recipe:
        ref = f"similar{ARGS.similar_recipe}"
    elif ARGS.random_recipe:
        ref = f"random{ARGS.random_recipe}"
    elif ARGS.random_buildsys_recipe:
        ref = f"randombuildsys{ARGS.random_buildsys_recipe}"
    else:
        ref = "none"

    if ARGS.raw_buildsys:
        buildsys = "raw"
    elif ARGS.distilled_cmake:
        buildsys = "distilled"
    else:
        buildsys = "none"

    return f"{ref}_{buildsys}"


def _next_available(path: Path) -> Path:
    """path.jsonl, path_2.jsonl, path_3.jsonl, ... whichever doesn't exist yet"""
    if not path.exists():
        return path
    i = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


if ARGS.results is None:
    model_slug = ARGS.model.replace("/", "_").replace(":", "_")
    ARGS.results = str(Path(ARGS.results_dir) / f"{model_slug}_{_config_slug()}.jsonl")

results_path = Path(ARGS.results)
if not ARGS.resume and results_path.exists():
    suffixed = _next_available(results_path)
    print(f"--resume not set and {results_path} already exists; writing to {suffixed} instead")
    ARGS.results = str(suffixed)

Path(ARGS.results).parent.mkdir(parents=True, exist_ok=True)

# load pkgs into the global namespace for use in all packages
with open(ARGS.input, "rb") as f:
    pkgs = pickle.load(f)


run_id = str(int(time.time()))
results = ResultsStore(
    run_id=run_id,
    filepath=ARGS.results,
    model=ARGS.model,
    max_attempts=ARGS.max_attempts,
    raw_buildsys=ARGS.raw_buildsys,
    distilled_cmake=ARGS.distilled_cmake,
    tree=ARGS.tree,
    random_recipe=ARGS.random_recipe,
    random_buildsys_recipe=ARGS.random_buildsys_recipe,
    similar_recipe=ARGS.similar_recipe,
    audit=ARGS.audit,
    rag=ARGS.rag,
)
ui = ProgressUI(total=ARGS.samples, max_attempts=ARGS.max_attempts)
rate_limiter = RateLimiter(max_calls=10, period=60, log=ui.log)
artifacts = ArtifactStore(run_id=run_id)

if ARGS.rag:
    rag_idx = load_index_from_cache(
        cache_root="packrag_cache",
        model_name="nomic-ai/nomic-embed-code",
        max_lines_per_chunk=20,
        # packages_by_name=packages_by_name,
        # source_path=data_path
    )


def generate_recipe(
    pkg_name: str,
    attempt: int,
    fresh_prompt: str = None,
    error: str = None,
    audit_output: str = None,
    past_recipe: str = None,
    references: dict = {},
    rag_chunks=None,
) -> tuple[int, str]:
    if fresh_prompt:
        prompt = fresh_prompt
    else:
        # if it's not a fresh prompt
        # create a new prompt that includes the error and other information we'd like
        # to include for the LLM to see
        prompt = render_template(
            "generate_loop",
            {
                "error": error,
                "audit_output": audit_output,
                "past_recipe": past_recipe,
                "references": references,
                "rag_chunks": rag_chunks,
            },
        )

    artifacts.save(pkg_name, f"generate_{attempt}.txt", prompt)

    rate_limiter.wait()
    num_tokens, recipe = call_llm(prompt, ARGS.model, log=ui.log)
    # sometimes models will indent the first line, we don't want this to count against them if they output otherwise valid recipes
    recipe = recipe.lstrip()
    artifacts.save(pkg_name, f"package_{attempt}.py", recipe)

    return num_tokens, recipe


def generate_handler(pkg: Package):
    pkg_recipe = status = error = audit_output = rag_chunks = None
    references = {}

    ui.start_pkg(pkg.name)
    with BuilderContainer(log=ui.log) as ctr:
        for attempt_num in range(ARGS.max_attempts):
            scores = {
                "dependency_score": None,
                "variants_score": None,
                "variants_extras": None,
            }

            ui.set_attempt(attempt_num, error)

            if not error:
                num_tokens, pkg_recipe, references, rag_chunks = generate_pkg(pkg)
            else:
                num_tokens, pkg_recipe = generate_recipe(
                    pkg.name,
                    attempt=attempt_num,
                    error=error,
                    audit_output=audit_output,
                    past_recipe=pkg_recipe,
                    references=references,
                    rag_chunks=rag_chunks,
                )

            ctr.write_pkg(pkg.name, pkg_recipe)
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
                    ui.set_stage(status)

                    # once the package has been loaded, we can get the scores for the package class
                    # we cannot get scores when git_repos is enabled as there is no ground truth for them...
                    if status == "load" and not ARGS.git_repos:
                        try:
                            scores["dependency_score"] = ctr.deps_score(pkg.name)
                        except Exception as exc:
                            ui.log(f"error getting dep score: {exc}")
                        try:
                            scores["variants_score"], scores["variants_extras"] = (
                                ctr.cmake_args_score(pkg.name)
                            )
                        except Exception as exc:
                            ui.log(f"error getting cmake score: {exc}")

                    results.log(
                        pkg_name=pkg.name,
                        status=status,
                        attempt_num=attempt_num,
                        references=references,
                        num_tokens=num_tokens,
                        **scores,
                    )

                    # we're done generating if we succeeded
                    if status == ARGS.success_status:
                        return
                except SpackError as exc:
                    error = str(exc)
                    status = f"{stage.name}_fail"
                    ui.set_stage(status)
                    results.log(
                        pkg_name=pkg.name,
                        status=status,
                        attempt_num=attempt_num,
                        message=error,
                        references=references,
                        num_tokens=num_tokens,
                        **scores,
                    )

                    # if audit checks are enabled, collect the output so it can be added
                    # to the regenerate job
                    # we don't care about audit checks if the pkg can't even load
                    # bc it would show the same message as before..

                    if ARGS.audit and status != "load_fail":
                        audit_output = ctr.audit_pkg(pkg.name)

                    # do not want to continue inside this attempt if there is a failure
                    break


def generate_pkg(
    target_pkg: Package,
) -> tuple[Package, str, str]:
    """
    Generates a package recipe and adds it to the specified Spack repo.

    args:
        target_pkg: the Package object of the pkg you'd like to experiment with

    returns:
        None if generation is not available
        otherwise, a Package object representing the generated package
        and the text of the prompt to generate the package
        and the text of the recipe
    """
    references = {}
    rag_chunks = None

    ui.log(f"generating package {target_pkg.name}")

    prompt_input = {"pkg_name": target_pkg.name}

    # extraction (fetch + build-system detection + cmake parse) is the same for a
    # given package regardless of ablation config/model, so it's cached across runs.
    # skipped for --git_repos (no stable version/cache key) and --tree (tree walk
    # needs the actual expanded source dir, which doesn't survive past the stage
    # context manager, so it isn't part of what gets cached).
    cache_eligible = not ARGS.git_repos and not ARGS.tree
    cache = load_extraction_cache(target_pkg.name) if cache_eligible else None

    if cache is not None:
        prompt_input["version"] = cache["version"]
        if not ARGS.baseline:
            if _target_buildsys_mismatch(cache["build_sys"]):
                raise GenerateException(
                    f"detected build sys {cache['build_sys']!r} != "
                    f"--target_buildsys {ARGS.target_buildsys!r}"
                )
            prompt_input["build_sys"] = cache["build_sys"]
            prompt_input["features"] = cache["features"]
            prompt_input["cmake_parsed"] = cache["cmake_parsed"]
            prompt_input["raw_files"] = cache.get("raw_files")
            if ARGS.raw_buildsys:
                prompt_input["raw_buildsys"] = prompt_input["raw_files"]
    else:
        if ARGS.git_repos:
            stage = GitCloneStage(target_pkg.url)
        else:
            # fetch the package and put it into a tmp directory
            stage, version = fetch_and_expand(target_pkg)
            if stage is None and version is None:
                raise GenerateException(f"could not fetch {target_pkg.name}")

        if not stage:
            raise GenerateException(f"stage not created for {target_pkg.name}")

        with stage:
            if ARGS.git_repos:
                path = Path(stage.path)
                version = f'git attrib: {target_pkg.url}; version directive: version("{target_pkg.branch}", branch="{target_pkg.branch}")'
            else:
                path = Path(stage.path) / "spack-src"

            prompt_input["version"] = str(version)
            if not ARGS.baseline:
                # name of the primary build system detected, any features that were found
                # this uses file-based detection, which doesn't really work with rocm or oneapi
                # the LLM would need to deduce this based on what it finds the build system info (parsed or raw)
                prompt_input["build_sys"], prompt_input["features"] = (
                    detect_build_systems(path)
                )

                if _target_buildsys_mismatch(prompt_input["build_sys"]):
                    raise GenerateException(
                        f"detected build sys {prompt_input['build_sys']!r} != "
                        f"--target_buildsys {ARGS.target_buildsys!r}"
                    )

                # cmake_parsed: the parsed CMakeContext AST, cmake-only, used only by
                # --distilled_cmake. raw_files: the raw build-file text dump, used by
                # --raw_buildsys -- computed the same way (get_build_files, shallowest
                # depth) for every supported build system, cmake included, so the
                # "raw" comparison is symmetric across build systems rather than
                # giving cmake the richer AST-parsed view.
                prompt_input["cmake_parsed"] = None
                prompt_input["raw_files"] = None

                if prompt_input["build_sys"] == "cmake":
                    prompt_input["cmake_parsed"] = str(parse_cmake(path))
                    raw = get_build_files(path, "cmake")
                    prompt_input["raw_files"] = (
                        format_raw_build_files(raw, repo_root=path) or None
                    )
                elif prompt_input["build_sys"] in ("autotools", "autoreconf", "makefile"):
                    # no distillation template exists for these yet -- only the raw
                    # file dump (get_build_files) is generalized, not an LLM-summarized
                    # equivalent of cmake_distilled
                    if ARGS.distilled_cmake:
                        raise GenerateException(
                            f"build sys {prompt_input['build_sys']} not supported "
                            "for --distilled_cmake (no distillation template implemented)"
                        )
                    raw = get_build_files(
                        path,
                        "autotools"
                        if prompt_input["build_sys"] in ("autotools", "autoreconf")
                        else "makefile",
                    )
                    prompt_input["raw_files"] = (
                        format_raw_build_files(raw, repo_root=path) or None
                    )
                elif ARGS.raw_buildsys or ARGS.distilled_cmake:
                    raise GenerateException(
                        f"build sys {prompt_input['build_sys']} not supported "
                        "for --raw_buildsys/--distilled_cmake (no parser implemented)"
                    )
                    # TODO add ability to parse and extract build system info for
                    # any other build systems (bazel, meson, ...)

            if ARGS.raw_buildsys:
                prompt_input["raw_buildsys"] = prompt_input["raw_files"]

            if ARGS.tree:
                prompt_input["tree"] = build_tree(path, max_depth=ARGS.tree_depth)

        if cache_eligible and not ARGS.baseline:
            save_extraction_cache(
                target_pkg.name,
                {
                    "version": prompt_input["version"],
                    "build_sys": prompt_input["build_sys"],
                    "features": prompt_input["features"],
                    "cmake_parsed": prompt_input["cmake_parsed"],
                    "raw_files": prompt_input["raw_files"],
                },
            )

    if ARGS.distilled_cmake:
        # not cached: distillation is model-dependent and we want it computed
        # fresh every run rather than reused across models/configs
        cmake_distilled_prompt = render_template("cmake_distilled", prompt_input)
        artifacts.save(
            target_pkg.name, "cmake_distilled_prompt.txt", cmake_distilled_prompt
        )

        rate_limiter.wait()
        _, cmake_distilled = call_llm(cmake_distilled_prompt, ARGS.model, log=ui.log)
        prompt_input["cmake_distilled"] = cmake_distilled

    if (
        ARGS.rag
        and ARGS.distilled_cmake
        and not ARGS.similar_recipe
        and not ARGS.random_buildsys_recipe
        and not ARGS.random_recipe
    ):
        deps, vars = extract_distilled_cmake(cmake_distilled)
        # query always needs to start with query:
        rag_query = f"query: depends_on: {' '.join(deps)} variant: {' '.join(vars)} features: {' '.join(prompt_input['features'])} buildsys: {prompt_input['build_sys']}"
        rag_chunks = retrieve_chunks(
            rag_idx,
            rag_query,
            top_k_packages=5,
            chunks_per_pkg=2,
            exclude_package_name=target_pkg.name,
            output_line_cap=12,
        )

    if ARGS.similar_recipe:
        prompt_input["num_similar_refs"] = ARGS.similar_recipe

        detected_dependencies, detected_variants = extract_distilled_cmake(
            cmake_distilled
        )

        similar_pkgs = find_similar_packages(
            pkgs,
            pkg_name=target_pkg.name,
            build_system=prompt_input["build_sys"],
            dependencies=detected_dependencies,
            variants=detected_variants,
            num_similar_refs=prompt_input["num_similar_refs"],
        )

        for n, (name, recipe, _score) in enumerate(similar_pkgs):
            key = f"similar{n}"
            references[key] = {}
            references[key]["pkg"] = name
            references[key]["recipe"] = recipe
    if ARGS.random_buildsys_recipe:
        for n in range(ARGS.random_buildsys_recipe):
            key = f"random_buildsys{n}"

            references[key] = {}

            references[key]["pkg"], references[key]["recipe"] = get_random_recipe(
                pkgs, build_system=prompt_input["build_sys"], avoid=target_pkg.name
            )
    if ARGS.random_recipe:
        for n in range(ARGS.random_recipe):
            key = f"random{n}"
            references[key] = {}

            references[key]["pkg"], references[key]["recipe"] = get_random_recipe(
                pkgs, avoid=target_pkg.name
            )

    # add references to the prompt input
    prompt_input["references"] = references
    prompt_input["rag_chunks"] = rag_chunks
    # GENERATE PROMPT
    prompt = render_template("generate_recipe", prompt_input)
    num_tokens, recipe = generate_recipe(
        pkg_name=target_pkg.name,
        attempt=0,  # first attempt
        fresh_prompt=prompt,
    )

    return (num_tokens, recipe, references, rag_chunks)


def pipeline():
    """
    sets up experimental pipeline given the `pkgs` in the global namespace and puts generated packages in the provided repo
    """

    eligible = []
    runs = 0

    completed: set[str] = set()
    if ARGS.resume:
        completed = load_completed_pkgs(
            ARGS.results, ARGS.success_status, ARGS.max_attempts
        )
        runs = len(completed)
        ui.completed = runs
        ui.log(f"--resume: {runs} package(s) already completed in {ARGS.results}")

    if ARGS.git_repos:
        eligible = load_git_repos(ARGS.git_repos_file)
        eligible = [p for p in eligible if p.name not in completed]
    else:
        include = PKG_LIST if PKG_LIST is not None else []
        exclude = list(completed)
        # non-deterministic by default; pass --seed to fix the shuffle order so the
        # same package subset is drawn across configs/models for consistent ablations
        pkgs_list = list(pkgs.values())
        if ARGS.seed is not None:
            random.Random(ARGS.seed).shuffle(pkgs_list)
        else:
            random.shuffle(pkgs_list)
        for pkg in pkgs_list:
            if include:
                if pkg.name in include:
                    eligible.append(pkg)
                # if the pkg name isn't in include, then don't add anything..
                continue
            # this is used to set a filter for the packages to include in the test set..pkg.build_systems is not used in the
            # generation step as the program should be able to figure it out by itself
            if pkg.name.startswith("py-"):
                continue
            # pkg.build_systems (Spack's declared build_system variant) is only a cheap
            # pre-filter/prediction here -- the live detect_build_systems() call inside
            # generate_pkg() is the real check, enforced via --target_buildsys
            if (ARGS.target_buildsys or "cmake") in pkg.build_systems:
                eligible.append(pkg)

        eligible = [p for p in eligible if p.name not in exclude]

    eligible = [
        p
        for p in eligible
        if not any(
            fnmatch.fnmatch(p.name.lower(), pat.lower()) for pat in IGNORE_PATTERNS
        )
    ]

    for pkg in eligible:
        if runs >= ARGS.samples:
            break

        try:
            generate_handler(pkg)
        except GenerateException as exc:
            # store a generation error so it can be easily identified during analysis and thrown out
            # mark workflow failure and keep going until we hit the target runs
            results.log(pkg_name=pkg.name, status="workflow_fail", message=str(exc))
            continue
        else:
            # valid run
            runs += 1
            ui.finish_pkg()


try:
    pipeline()
finally:
    ui.close()
