#!/usr/bin/env python3
"""
Script to extract Spack package metadata to JSON.
Extracts package information including dependencies, variants, and build metadata.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import spack
import spack.repo
import spack.variant
from spack.package_base import PackageBase
from spack.spec import Spec


def extract_dependencies(pkg: PackageBase) -> list[str]:
    """Extract dependency names from a package"""
    dependencies = set()

    for dep_dict in getattr(pkg, "dependencies", {}).values():
        for dep in dep_dict.values():
            dependencies.add(Spec(dep.spec).name)

    return sorted(dependencies)


def extract_variants_and_build_systems(pkg: PackageBase) -> tuple[list[str], list[str]]:
    """Extract variant names and build systems"""
    variant_names = set()
    build_systems = set()

    for variants_dict in getattr(pkg, "variants", {}).values():
        for variant in variants_dict.values():
            if variant.name == "build_system":
                raw_values = variant.values
                if raw_values:
                    for value in raw_values:
                        if isinstance(value, (str, bool)):
                            build_systems.add(str(value))
                        elif isinstance(value, spack.variant.ConditionalValue):
                            build_systems.add(str(value.value))
            else:
                variant_names.add(variant.name)

    return sorted(variant_names), sorted(build_systems)


def extract_package_info_worker(args: tuple[str, str]) -> dict | None:
    """Worker function for parallel extraction - gets its own repo instance"""
    pkg_name, repo_name = args
    repo = spack.repo.PATH.get_repo(repo_name)
    return extract_package_info(pkg_name, repo)


def extract_package_info(pkg_name: str, repo: spack.repo.Repo) -> dict | None:
    """Extract package metadata for a single package"""
    try:
        pkg_class = repo.get_pkg_class(f"{repo.namespace}.{pkg_name}")
    except (spack.repo.UnknownPackageError, spack.repo.RepoError):
        print(f"Warning: Could not load package {pkg_name}")
        return None

    pkg_info = {
        "name": pkg_class.name,
        "virtual": getattr(pkg_class, "virtual", False),
        "has_code": getattr(pkg_class, "has_code", True),
    }

    for attr in ("doc", "homepage", "git"):
        if value := getattr(pkg_class, attr, None):
            pkg_info[attr] = value
    if tags := list(getattr(pkg_class, "tags", [])):
        pkg_info["tags"] = tags

    variants, build_systems = extract_variants_and_build_systems(pkg_class)
    if build_systems:
        pkg_info["build_systems"] = build_systems

    with open(repo.filename_for_package_name(pkg_class.name)) as f:
        pkg_info["recipe"] = f.read()

    if dependencies := extract_dependencies(pkg_class):
        pkg_info["dependencies"] = dependencies
    if variants:
        pkg_info["variants"] = variants

    return pkg_info


def collect_packages_with_filters(repo: spack.repo.Repo, tags: list[str]) -> set[str]:
    # Get tagged packages
    tagged_pkgs = set()
    for tag in tags:
        tagged_pkgs.update(spack.repo.PATH.packages_with_tags(tag))

    other_pkgs = set()

    # Get dependencies of tagged packages
    for pkg_name in tagged_pkgs:
        try:
            pkg_class = repo.get_pkg_class(f"{repo.namespace}.{pkg_name}")
            other_pkgs.update(extract_dependencies(pkg_class))
        except Exception:
            print(f"Warning: Could not process dependencies for {pkg_name}")

    # Get providers for virtual packages (but not the virtuals themselves)
    virtual_pkgs = set(spack.repo.PATH.provider_index.providers.keys())
    provider_pkgs = {
        Spec(provider).name
        for pkg in virtual_pkgs
        for provider in spack.repo.PATH.providers_for(pkg)
    }

    other_pkgs.update(provider_pkgs)
    other_pkgs -= tagged_pkgs

    # Filter out virtual packages from all collections
    all_pkgs = (tagged_pkgs | other_pkgs) - virtual_pkgs
    print(
        f"Collected {len(all_pkgs)} packages ({len(tagged_pkgs)} tagged + {len(other_pkgs)} deps/providers)"
    )
    return all_pkgs


def build_indexes(packages: dict[str, dict]) -> dict:
    """Build reverse indexes for fast lookups"""
    by_build_system = {}
    by_dependency = {}
    by_variant = {}

    for pkg_name, pkg_info in packages.items():
        # Index by build system
        for bs in pkg_info.get("build_systems", []):
            by_build_system.setdefault(bs, []).append(pkg_name)

        # Index by dependency
        for dep in pkg_info.get("dependencies", []):
            by_dependency.setdefault(dep, []).append(pkg_name)

        # Index by variant
        for variant in pkg_info.get("variants", []):
            by_variant.setdefault(variant, []).append(pkg_name)

    return {
        "by_build_system": by_build_system,
        "by_dependency": by_dependency,
        "by_variant": by_variant,
    }


def extract_all_packages(
    output_file: str | None = None,
    repo_name: str = "builtin",
    tags: list[str] | None = None,
) -> dict[str, dict]:
    """Extract all filtered packages and write to JSON file"""
    if tags is None:
        tags = ["e4s"]

    repo = spack.repo.PATH.get_repo(repo_name)
    pkg_names = collect_packages_with_filters(repo, tags)
    packages = {}

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                extract_package_info_worker, (pkg_name, repo_name)
            ): pkg_name
            for pkg_name in sorted(pkg_names)
        }

        for future in as_completed(futures):
            try:
                if pkg_info := future.result():
                    packages[futures[future]] = pkg_info
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    # Build indexes for fast lookups
    indexes = build_indexes(packages)

    if output_file:
        output_data = {"packages": packages, "indexes": indexes}
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved {len(packages)} packages to {output_file}")

    return packages


def main():
    parser = argparse.ArgumentParser(
        description="Extract Spack package metadata to JSON"
    )
    parser.add_argument(
        "--tags", default="e4s", help="Comma-separated package tags (default: e4s)"
    )
    parser.add_argument(
        "--repo", default="builtin", help="Spack repository name (default: builtin)"
    )
    parser.add_argument(
        "--output", help="Output JSON file (default: packages-<timestamp>.json)"
    )
    args = parser.parse_args()

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    output = args.output or f"packages-{int(time.time())}.json"

    extract_all_packages(output, args.repo, tags)


if __name__ == "__main__":
    main()
