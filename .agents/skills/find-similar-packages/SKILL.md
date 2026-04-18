---
name: find-similar-packages
description: Extract Spack package metadata and find similar packages based on build system, dependencies, and variants. Use to discover reference recipes for package generation.
---

# Find Similar Packages

## When to use this skill
Use this skill when you need to find existing Spack packages similar to a target package. This is particularly useful:
- Before generating a new Spack recipe, to find reference implementations
- When looking for examples of packages with similar build configurations
- To discover how similar packages handle dependencies, variants, or build systems

## Prerequisites
- Basic metadata about the target package (build system, dependencies, variants)
- If a recent package data file doesn't exist, run the extraction first (Step 1)

## Step 1 -- extract package metadata (if needed)

If a recent package data file doesn't exist or is outdated, extract fresh metadata:

```bash
python scripts/extract_packages.py --tags e4s --repo builtin
```

This creates `packages-<timestamp>.json` in the current directory containing:
- All packages with the specified tags and their dependencies
- Reverse indexes for fast lookup by build system, dependency, and variant


## Step 2 -- identify target package metadata

Use the `cmake-extract` skill to extract metadata from the target repository. Read the resulting `spack_metadata.txt` file to gather:

1. **Build system** (required): Found in the BUILD_SYSTEM section
2. **Dependencies** (optional but recommended): Found in the SUMMARY section under "dependencies"
3. **Variants** (optional but recommended): Found in the SUMMARY section under "variants"
4. **Package name to exclude** (required): The target package name

## Step 3 -- search for similar packages

Run the similarity search with the gathered metadata:

```bash
python scripts/find_similar.py packages-<timestamp>.json \
  --build-system <build_system> \
  --dependencies "<dep1>,<dep2>,..." \
  --variants "<var1>,<var2>,..." \
  --exclude <target_pkg_name> \
  --top-k <N>
```

### Parameters

**Required:**
- `--build-system, -b`: Build system to filter by (e.g., "cmake", "autotools")
- `--exclude, -e`: Package name to exclude (and packages with similar names)

**Optional:**
- `--dependencies, -d`: Comma-separated list of dependencies
- `--variants, -v`: Comma-separated list of variants
- `--top-k, -k`: Number of results to return (default: 10)
- `--dep-weight`: Weight for dependency overlap (default: 0.6)
- `--var-weight`: Weight for variant overlap (default: 0.4)
- `--show-recipe, -r`: Show full package recipes in output

### How scoring works

The similarity score is calculated as:
```
score = (dep_weight × num_overlapping_deps) + (var_weight × num_overlapping_variants)
```

Results are sorted by score descending. Higher scores mean more similar packages.

### Output format

Without `--show-recipe`:
```
package1: 8.5
package2: 7.2
package3: 6.1
```

With `--show-recipe`:
```
package1: 8.5
<full recipe source code>

package2: 7.2
<full recipe source code>
```

## Step 4 -- retrieve reference recipes

Retrieve the top 2 similar packages with their full recipes:

```bash
python scripts/find_similar.py packages-*.json \
  --build-system <build_system> \
  --dependencies "<dep1>,<dep2>,..." \
  --variants "<var1>,<var2>,..." \
  --exclude <target_pkg_name> \
  --top-k 2 \
  --show-recipe
```

Use these recipes as references when generating new Spack packages. They provide:
- Implementation patterns for similar build systems
- Examples of how to handle similar dependencies
- Variant definitions and defaults
- Common testing and validation approaches

**The top 2 results are usually sufficient.** Only proceed to Step 5 if you encounter issues during recipe generation.

## Step 5 -- iterate if needed (optional)

**Only use this step if:**
- Recipe generation failed and you need different reference examples
- The initial search returned no results (see Debugging below)

**To find more references:**
```bash
# Get more results
python scripts/find_similar.py packages-*.json \
  --build-system cmake --dependencies "mpi,cuda" \
  --variants "shared" --exclude mypackage \
  --top-k 5 --show-recipe
```

## Debugging

**If you get no results from your initial search:**

1. **Broaden the search** - Remove some dependencies or variants:
   ```bash
   python scripts/find_similar.py packages-*.json \
     --build-system cmake --dependencies "mpi" \
     --exclude mypackage --top-k 2 --show-recipe
   ```

2. **Adjust scoring weights** if you suspect good matches exist but aren't ranking high:
   ```bash
   # Prioritize dependency overlap
   python scripts/find_similar.py packages-*.json \
     --build-system cmake --dependencies "mpi,cuda" \
     --exclude mypackage --dep-weight 0.8 --var-weight 0.2 \
     --top-k 2 --show-recipe
   ```

3. **Verify build system name** - Check it matches what's in Spack (e.g., "cmake" not "CMake")

**Do not adjust parameters unless you get zero results or recipe generation fails.**
