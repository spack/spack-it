import argparse
import functools
import os
import pickle
import random
import time
from pathlib import Path

from neomodel import config, db

from extraction.cmake import parse_repo as parse_cmake
from extraction.package_schema import Package
from extraction.repository import (
    build_tree,
    detect_build_systems,
    fetch_and_expand,
)
from generate.container import BuilderContainer, SpackError, Stage
from generate.util import (
    ArtifactStore,
    GenerateException,
    GitCloneStage,
    RateLimiter,
    ResultsStore,
    call_llm,
    extract_distilled_cmake,
    get_random_recipe,
    load_git_repos,
    render_template,
)
from rag.retrieve import load_index_from_cache, retrieve_chunks

####### CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", help="filename to a pickled file that has a list of Packages"
)
parser.add_argument(
    "--samples", type=int, default=5, help="number of experiments to run"
)
parser.add_argument(
    "--model",
    type=str,
    help="the model to use",
)
parser.add_argument(
    "--results",
    type=str,
    default="data/results.jsonl",
    help="where to store results of each run",
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
    help="include similar spack recipe in the LLM prompt (using GraphRAG)",
)
parser.add_argument(
    "--baseline",
    action="store_true",
    help="option for experiment where we let the llm do all the research with no parsed data included",
)

ARGS = parser.parse_args()
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
rate_limiter = RateLimiter(max_calls=10, period=60)
artifacts = ArtifactStore(run_id=run_id)

config.DATABASE_URL = os.getenv("NEO4J_URI")

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
    num_tokens, recipe = call_llm(prompt, ARGS.model)
    # sometimes models will indent the first line, we don't want this to count against them if they output otherwise valid recipes
    recipe = recipe.lstrip()
    artifacts.save(pkg_name, f"package_{attempt}.py", recipe)

    return num_tokens, recipe


def generate_handler(pkg: Package):
    pkg_recipe = status = error = audit_output = rag_chunks = None
    references = {}

    with BuilderContainer() as ctr:
        for attempt_num in range(ARGS.max_attempts):
            scores = {
                "dependency_score": None,
                "variants_score": None,
                "variants_extras": None,
            }

            print(f"attempt={attempt_num}, prev_error={error}")

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

                    # once the package has been loaded, we can get the scores for the package class
                    # we cannot get scores when git_repos is enabled as there is no ground truth for them...
                    if status == "load" and not ARGS.git_repos:
                        try:
                            scores["dependency_score"] = ctr.deps_score(pkg.name)
                        except Exception as exc:
                            print(f"error getting dep score: {exc}")
                        try:
                            scores["variants_score"], scores["variants_extras"] = (
                                ctr.cmake_args_score(pkg.name)
                            )
                        except Exception as exc:
                            print(f"error getting cmake score: {exc}")

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

    print(f"generating package {target_pkg.name}")

    if ARGS.git_repos:
        stage = GitCloneStage(target_pkg.url)
    else:
        # fetch the package and put it into a tmp directory
        stage, version = fetch_and_expand(target_pkg)
        if stage is None and version is None:
            raise GenerateException(f"could not fetch {target_pkg.name}")

    if not stage:
        raise GenerateException(f"stage not created for {target_pkg.name}")

    prompt_input = {"pkg_name": target_pkg.name}
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
            prompt_input["build_sys"], prompt_input["features"] = detect_build_systems(
                path
            )

            if prompt_input["build_sys"] != "cmake":
                raise GenerateException(
                    f"build sys {prompt_input['build_sys']} not supported"
                )
                # TODO add ability to parse and extract build system info
                # need to change/generalize parse_cmake and get_build_files

            # store this for use in distilled_cmake
            # parsed cmake is always required for distilled_cmake..but it needs a different name or else it'll be injected into the prompt for recipe generation
            prompt_input["cmake_parsed"] = str(parse_cmake(path))

        if ARGS.raw_buildsys:
            prompt_input["raw_buildsys"] = prompt_input["cmake_parsed"]

        if ARGS.tree:
            prompt_input["tree"] = build_tree(path, max_depth=ARGS.tree_depth)

    if ARGS.distilled_cmake:
        cmake_distilled_prompt = render_template("cmake_distilled", prompt_input)
        artifacts.save(
            target_pkg.name, "cmake_distilled_prompt.txt", cmake_distilled_prompt
        )

        rate_limiter.wait()
        _, cmake_distilled = call_llm(cmake_distilled_prompt, ARGS.model)
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

        if ARGS.distilled_cmake:
            detected_dependencies, detected_variants = extract_distilled_cmake(
                cmake_distilled
            )

            cypher_query = render_template(
                "cypher_similar_pkg",
                {
                    # could replace this with list comprehension inside the template?
                    "dependencies": f'["{'","'.join(map(str, detected_dependencies))}"]',
                    "variants": f'["{'","'.join(map(str, detected_variants))}"]',
                    "pkg_name": target_pkg.name,
                    "build_sys": prompt_input["build_sys"],
                    "num_similar_refs": prompt_input["num_similar_refs"],
                },
            )
        else:
            # need to use the llm to generate the query for us because it relies on parsed buildsys
            cypher_query_prompt = render_template("find_similar_pkg", prompt_input)
            artifacts.save(
                target_pkg.name, "cypher_query_prompt.txt", cypher_query_prompt
            )
            _, cypher_query = call_llm(cypher_query_prompt, ARGS.model)
        artifacts.save(target_pkg.name, "cypher_query.txt", cypher_query)
        try:
            results, _ = db.cypher_query(cypher_query)
        except Exception as exc:
            raise GenerateException(f"cypher query error: {exc}")

        for n in range(len(results)):
            key = f"similar{n}"
            references[key] = {}
            references[key]["pkg"] = results[n][0]
            references[key]["recipe"] = results[n][1]
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

    if ARGS.git_repos:
        eligible = load_git_repos()
    else:
        include = []
        exclude = []
        # want to make sure the order in which samples are chosen is non-deterministic
        pkgs_list = list(pkgs.values())
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
            if "cmake" in pkg.build_systems:
                eligible.append(pkg)

        eligible = [p for p in eligible if p.name not in exclude]

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


pipeline()
