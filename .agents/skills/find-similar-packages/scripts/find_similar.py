#!/usr/bin/env python3
"""
Find similar packages based on build system, dependencies, and variants.
"""

import argparse
import json


def load_data(json_file: str) -> tuple[dict, dict]:
    """Load package data and indexes from JSON file"""
    with open(json_file) as f:
        data = json.load(f)
    return data["packages"], data["indexes"]


def find_similar_by_metadata(
    build_system: str,
    dependencies: list[str],
    variants: list[str],
    packages: dict,
    indexes: dict,
    exclude_package: str,
    top_k: int = 10,
    dep_weight: float = 0.6,
    var_weight: float = 0.4,
) -> list[tuple[str, float, str]]:
    """
    Find top K similar packages given metadata.

    Args:
        build_system: Build system to filter by
        dependencies: List of dependencies to match
        variants: List of variants to match
        packages: Dictionary of all packages
        indexes: Reverse indexes for fast lookups
        exclude_package: Package name to exclude from results (and similar names)
        top_k: Number of similar packages to return
        dep_weight: Weight for dependency overlap (default: 0.6)
        var_weight: Weight for variant overlap (default: 0.4)

    Returns:
        List of (package_name, score, recipe) tuples sorted by score descending
    """
    target_deps = set(dependencies)
    target_vars = set(variants)

    # Get candidate packages with matching build system
    candidates = set(indexes["by_build_system"].get(build_system, []))

    # Filter out excluded package and similar names
    candidates.discard(exclude_package)
    candidates = {
        c
        for c in candidates
        if exclude_package.lower() not in c.lower()
        and c.lower() not in exclude_package.lower()
    }

    # Score each candidate
    results = []
    for candidate in candidates:
        pkg = packages[candidate]
        deps = set(pkg.get("dependencies", []))
        variants_set = set(pkg.get("variants", []))

        # Calculate overlap scores
        dep_score = len(target_deps & deps)
        var_score = len(target_vars & variants_set)
        total_score = dep_weight * dep_score + var_weight * var_score

        if total_score > 0:
            results.append((candidate, total_score, pkg.get("recipe", "")))

    # Sort by score descending and return top K
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def main():
    parser = argparse.ArgumentParser(
        description="Find similar Spack packages based on build system, dependencies, and variants"
    )
    parser.add_argument("data_file", help="JSON file with package data")
    parser.add_argument(
        "--build-system", "-b", required=True, help="Build system to filter by"
    )
    parser.add_argument(
        "--dependencies", "-d", help="Comma-separated list of dependencies"
    )
    parser.add_argument("--variants", "-v", help="Comma-separated list of variants")
    parser.add_argument(
        "--exclude", "-e", required=True, help="Package name to exclude from results"
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=10, help="Number of results (default: 10)"
    )
    parser.add_argument(
        "--dep-weight",
        type=float,
        default=0.6,
        help="Dependency overlap weight (default: 0.6)",
    )
    parser.add_argument(
        "--var-weight",
        type=float,
        default=0.4,
        help="Variant overlap weight (default: 0.4)",
    )
    parser.add_argument(
        "--show-recipe", "-r", action="store_true", help="Show package recipes"
    )
    args = parser.parse_args()

    # Load data
    packages, indexes = load_data(args.data_file)

    # Parse inputs
    dependencies = (
        [d.strip() for d in args.dependencies.split(",")] if args.dependencies else []
    )
    variants = [v.strip() for v in args.variants.split(",")] if args.variants else []

    # Find similar packages
    results = find_similar_by_metadata(
        args.build_system,
        dependencies,
        variants,
        packages,
        indexes,
        args.exclude,
        args.top_k,
        args.dep_weight,
        args.var_weight,
    )

    # Display results
    for name, score, recipe in results:
        if args.show_recipe:
            print(f"{name}: {score:.1f}\n{recipe}\n")
        else:
            print(f"{name}: {score:.1f}")


if __name__ == "__main__":
    main()
