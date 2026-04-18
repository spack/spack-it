---
name: generate-recipe
description: Translate extracted repository metadata into a Spack package recipe, then validate and iterate until the package installs successfully.
---

# Recipe Generation

## When to use this skill
Use this skill after the `cmake-extract` skill has completed and produced a `spack_metadata.txt` file. This skill translates the extracted metadata into a working Spack package recipe.

## Prerequisites
- Read `AGENTS.md` before invoking any spack commands — it defines how spack must be called in this environment.
- `spack_metadata.txt` must exist in the repository directory
- User must provide the package download URL (tarball or git) and package name
- **Recommended:** Run the `find-similar-packages` skill first to find 2 reference recipes. These provide implementation patterns for similar packages and improve recipe quality.

## Step 1 -- read metadata
Read `spack_metadata.txt` from the repository directory. The file contains:
- **BUILD_SYSTEM**: determines the Spack base class (e.g. cmake → CMakePackage)
- **FEATURES**: corroborating repository-level signals
- **CMAKE MAPPINGS**: authoritative instructions — emit exactly the Spack abstraction shown on each line, consulting the evidence field for ambiguous situations
- **SUMMARY**

## Step 2 -- generate recipe

Given the metadata, generate a Spack package recipe as a plain text Python source file.

### import rules
Determine required builder classes from the metadata and import them first, in this order:

  from spack_repo.builtin.build_systems.cmake import CMakePackage  # if cmake
  from spack_repo.builtin.build_systems.cuda import CudaPackage    # if cuda, etc.
  from spack.package import *

Respect this import order or the package will fail to build.

### class naming
Name the class using PascalCase derived directly from the Spack spec name:
  Example: "bricks" → class Bricks(CMakePackage):

### Builder classes
Include additional builder classes (e.g. CudaPackage, ROCmPackage) when strongly implied by
the build system or FEATURES. List them as multiple inheritance after the primary base class.

### Variants, dependencies, conflicts
If cmake metadata is available, translate every mapping line from the CMAKE MAPPINGS section into its corresponding Spack directive.

If no cmake mappings exist (e.g. the build system is Autotools, Meson, or something else), derive the equivalent mappings yourself by reading the build files directly — look for optional feature flags, `find_package`-style dependency checks, and conditional compilation blocks. Apply the same translation logic: each meaningful build option becomes a variant or dependency, each version check becomes a constraint. All decisions must be traceable to something in the build files (see Constraint traceability).

### Beyond metadata
If there is other behavior to encode in the recipe that is necessary for installation or feature-completeness (such as build environment setup or other logic), feel free to add as long as they are reflected in Spack best practices.

### Version constraints

Version constraints use **prefix matching**, which differs from most package managers. A bound like `1.2` means "the `1.2` prefix", not the exact version `1.2.0`:

- `@1.2` matches any version with prefix `1.2` — `1.2`, `1.2.3`, `1.2.3.4.5` — but not `1.1.x` or `1.3.x`.
- `@:1.2` (upper bound) includes `1.2.3`, `1.2.3.4.5`, but not `1.3.0`.
- `@1.2:` (lower bound) matches `1.2.*` and anything after.
- `@1.1:1.2` matches the `1.1.*` through `1.2.*` series but not `1.3.0`.

**Avoid gaps between version ranges.** Use minor-version boundaries rather than exact patch versions so ranges are contiguous. For example, write `@:5.1` and `@5.2:` — not `@:5.1.0` and `@5.2.1:`, which leaves a gap.

### Omit these directives entirely
maintainers, tags

### Using reference recipes
If you used the `find-similar-packages` skill, incorporate the reference recipes found:
- Use them to inform implementation patterns for the build system
- Adopt similar approaches for dependency handling and variant definitions
- **Do not copy behavior that is not backed by the extracted metadata**

### General rules
- No prose, footnotes, or explanations in the output
- No markdown, no syntax highlighting backticks
- No unnecessary methods or overrides
- Prioritize accuracy over completeness — do not hallucinate any information

### Constraint traceability
For every non-trivial constraint added to the recipe — version bounds, conflicts, conditional dependencies, variant defaults — retain a clear mental record of why it was added and where the evidence came from. Sources include:

- The extracted cmake metadata
- Upstream commit history or changelogs (via `gh`)
- Upstream issues or PRs that document a known incompatibility
- Build error output observed during iteration

If the user asks why a constraint exists, give a precise answer: what broke, what commit or version introduced the breakage, and what evidence supports the bound. A vague answer is not acceptable. If the justification cannot be reconstructed, say so and offer to re-investigate rather than guessing.

Example of the expected quality:
  `depends_on("zlib@1.2.11:")` — `CMakeLists.txt` calls `find_package(ZLIB 1.2.11 REQUIRED)`; mapped directly from the version argument.

## Step 3 -- create package file
Run the following to create the package scaffold and capture the output path:

  spack create <url> --name <pkg_name> --skip-editor

The URL can be a link to a tarball or a git repository. Ask the user for this information if not already provided. With the URL, Spack will look for versions and checksums automatically.

Parse the output to extract the file path where the package was created. Read the generated file, and merge it with the recipe generated in Step 2.

### Handling versions for git repositories

When using a git URL, you must verify what versions actually exist before adding them to the recipe:

1. **Check for release tags first** using `gh api repos/<owner>/<repo>/tags` to see what tags exist in the repository.

2. **If no release tags exist:**
   - Use `version("main", branch="main")` (or the appropriate default branch name)
   - Ensure the recipe has `git = "<url>"` set
   - Add a comment noting that the user can supply tarball URLs later for proper versioned entries

## Step 4 -- evaluate and iterate
Run each command below in order. Use any errors or warnings to revise the recipe and re-run from the failing phase. Continue until installation succeeds or an issue is deemed intractable.

Any information needed to fix the recipe should be available in the error messages, the target repository's build configuration files, any reference packages, or Spack documentation. It should not be necessary to search for external details unless there is a critical issue you cannot resolve with the available information.

### Environment setup
Create a temporary directory and a Spack environment rooted there. Use this directory path for all subsequent `-e` invocations:

  export tmp_env=$(mktemp -d)
  spack env create --dir $tmp_env

### Phase 1 — syntax check
  spack -e $tmp_env list <pkg_name>

Catches syntax errors or load failures. Fix any issues before proceeding.

### Phase 2 — static analysis
  spack -e $tmp_env audit packages <pkg_name>

Checks for missing dependencies, malformed directives, and other static issues.
Fix any issues before proceeding.

### Phase 3 — concretization
  spack -e $tmp_env spec <pkg_name>

Resolves all dependencies and constraints. Fix any issues before proceeding.

### Phase 4 — installation
Add the package as a root spec in the environment, then install:

  spack -e $tmp_env add <pkg_name>
  spack -e $tmp_env install

Attempts a full build and install. Fix any issues and re-run from the failing phase.

### General tips
**Make small, incremental changes.** When iterating on a failing recipe, change one thing at a time and re-run the failing phase. Changing multiple things at once makes it harder to identify what fixed or broke something.

**Always concretize before any other env operation** (`stage`, `install`, `location`, etc.). After concretizing, use just the package name — no `@version` — since the version is already pinned. Without prior concretization, Spack resolves the spec ad-hoc and may not pick the version you expect.

### Dependency failures
If any phase fails due to a broken or missing dependency (rather than an issue in the target package itself), stop and ask the user:
  "The failure appears to be in dependency <dep>. Would you like to fix that first?"
Do not attempt to fix dependency packages autonomously — stay focused on the target package.

### Debugging with gh
If online access is available, `gh` is a powerful debugging tool. Use it freely to gather context that isn't in the local error output:

- **Tags and releases**: `gh api repos/<owner>/<repo>/tags` — find what versions actually exist upstream
- **Raw file content**: `gh api repos/<owner>/<repo>/contents/<path> --jq '.content' | base64 -d` — read `CMakeLists.txt`, `setup.py`, etc. at a specific tag without cloning
- **Upstream issues and PRs**: `gh issue list -R <owner>/<repo> --search "<keyword>"` — check if a build failure is a known upstream bug
- **Diffs between versions**: `gh api repos/<owner>/<repo>/compare/<base>...<head>` — understand what changed between two releases that may affect the recipe
- **CI logs**: if a failure URL is provided (e.g. a GitLab or GitHub Actions link), fetch the raw log directly by appending `/raw` or using `gh run view`

When a user provides a failure URL, try fetching its raw content before asking for more information.

## Step 5 -- smoke tests (optional)
Ask the user before proceeding: "Installation succeeded. Would you like to run smoke tests across variant combinations?"

If yes, select a representative set of specs to test — toggle boolean variants, pin optional dependencies to versions within their declared bounds, and combine constraints that are likely to interact. Do not attempt to cover every combination; be judicious. Good candidates include:
- The default spec (already installed — skip if env from Step 4 is still intact)
- Key boolean variants flipped from their default (e.g. `+cuda`, `~mpi`)
- An optional dependency pinned to a specific version within its allowed range
- A combination of two or more non-conflicting variants

For each spec to test, create a fresh environment, add the spec, and install:

  export tmp_env_N=$(mktemp -d)
  spack env create --dir $tmp_env_N
  spack -e $tmp_env_N add <pkg_name> <constraints>
  spack -e $tmp_env_N install

Use a separate `tmp_env_N` variable for each spec (e.g. `tmp_env_1`, `tmp_env_2`, ...) so environments remain independent.

Report a summary of which specs passed and which failed. Failures in smoke tests should be investigated and fixed the same way as Phase 4 failures, unless they trace back to a dependency.
