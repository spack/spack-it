---
name: cmake-extract
description: Extract and analyze CMake build metadata from a repository, mapping it to Spack-relevant abstractions for use in recipe generation
---

# Repository Extraction
Use this skill when a user initiates a request to create or update a Spack package recipe. The details extracted by this skill will be used in downstream tasks like package generation and repair.

## Step 1 -- obtain repository
The user will provide a URL (tarball/Git) or local path to a codebase. Use standard tools (curl/wget/tar/zip/git) to download and open the repo. 

Place downloaded files into a newly created temporary directory (`mktemp -d "$TMPDIR/XXXXX"`).

Downloads should be efficient (i.e., use `depth=1` with Git).

Once the repository is accessible, run the two extraction steps below. The scripts accept directory paths as their argument and expand shell variables (e.g., `$TMPDIR`).

## Step 2 -- detect build system
Run: `python scripts/detect_build_systems.py <repo_dir>`

This outputs two values:
- **build_system**: the detected build system (e.g. cmake, autotools, meson)
- **features**: additional repository-level features (e.g. cuda, python)

If the detected build system is not `cmake`, stop immediately and inform the user:
"Only CMake-based repositories are supported."

## Step 3 -- extract CMake metadata
Run: `python scripts/cmake_extract.py <repo_dir>`

This script scans CMake files in the repository and outputs a structured summary of:
- cache_sets: cache variables (type, default, docstring)
- normal_sets: plain set() variables
- commands: ordered list of CMake commands and arguments
- options: option(NAME "help" DEFAULT) toggles
- if blocks: conditional logic with true/false bodies

Because the script will read the CMake files, do not `cat` or `grep` the files.

## Step 4 -- map CMake metadata to Spack abstractions
Your task is to translate the CMake metadata from step 3 into a concise manifest of Spack-relevant insights used for recipe generation.

Brevity and accuracy are critical; do not invent anything not explicitly present in the metadata.

### Output format

Plain text, one mapping per line:
    <cmake-symbol> → <spack-abstraction> ← <evidence>
Example lines:
  WITH_ADIAK → variant('+adiak') ← cache_sets:WITH_ADIAK (BOOL, default ON)
  find_package(Python) → variant('python') + depends_on('python', when='+python') ← commands:find_package
  cmake_minimum_required(3.14) → depends_on('cmake@3.14:', type='build') ← commands:cmake_minimum_required
  project(LANGUAGES CXX) → depends_on('cxx', type='build') ← commands:project LANGUAGES

After all mapping lines, append a SUMMARY block:

  SUMMARY
  variants: var1 var2 var3
  dependencies: dep1 dep2 dep3
  conflicts: '+tight_error when=round=never'
  compilers: cxx fortran

All identifiers must use Spack-style naming (lowercase, no cmake prefixes): cmake, cuda, mpi, python, fortran, c, cxx, openmpi, etc.

### Mapping guidelines

#### Variants
- `WITH_*`, `ENABLE_*`, `USE_*`, `BUILD_*` BOOL variables → `variant(...)`
- STRING variables with enumerated STRINGS property → multi-valued variant
  Example: ZFP_ROUNDING_MODE (NEVER, FIRST, LAST) → variant('round', values=('never','first','last'))
- Emit variant('shared') if and only if BUILD_SHARED_LIBS or a direct equivalent is present
- Normalize enum values: strip CMake prefixes, lowercase
  Example: ZFP_ROUND_NEVER → 'never'
- Use canonical names for language bindings: variant('python'), variant('fortran'), variant('c')
  Do not use: zfpy, zforp, cfp, or other non-standard names
- Do not skip options that default to OFF, are marked ADVANCED, or appear internal,
  if they control behavior, optional outputs, or language bindings

#### Conflicts
- If a CMake option is only valid when another has a specific value, emit a Spack conflict:
  Example: TIGHT_ERROR only valid when ROUNDING != NEVER →
    conflicts('+tight_error', when='round=never')

#### Compilers
Emit depends_on('<lang>', type='build') for any language that is actually used to compile sources.
Use the following signals, in rough order of strength:

  Strong signals (language is definitely used):
  - project(... LANGUAGES C/CXX/Fortran ...)
  - enable_language(C/CXX/Fortran)
  - source files with .c, .cpp, .f90, .f, .f95 extensions referenced in CMakeLists

  Weak signals (language is likely used, emit with note):
  - check_cxx_compiler_flag, check_language, check_c_compiler_flag
  - target_link_libraries referencing C/C++/Fortran objects
  - CMAKE_CXX_FLAGS, CMAKE_C_FLAGS, CMAKE_Fortran_COMPILER variables set or tested

  You must emit the dependency even if there is only a single weak signal.

  Emit exactly one mapping line per language detected, citing the strongest evidence found.

#### Dependencies
- find_package(X REQUIRED) or find_library(X) → depends_on('x')
- find_package(X COMPONENTS foo bar) → depends_on('x') and note components
- Wrapped in a condition tied to a variant → add when='+<variant>'
- pkg_check_modules(X) → depends_on('x')
- target_link_libraries referencing an external library → depends_on (if not already covered)
- MPI is a virtual package: find_package(MPI) → depends_on('mpi')
- OpenMP is never a dependency; if find_package(OpenMP) is present → variant('openmp') only
- cmake_minimum_required(VERSION X.Y) → depends_on('cmake@X.Y:', type='build')
- CUDA: `find_package(CUDA)` or `enable_language(CUDA)`
   - GPU-specific files, target_link_libraries(... cuda) → implies depends_on('cuda'), gated on +cuda  
- Python  : any `find_package(Python* …)`            ⟶ `variant('python')`, associated deps

#### What not to emit
- Do not guess version constraints for any dependency
- Do not emit dependencies or variants that are not evidenced in the metadata
- Do not impart knowledge that is not present in the CMake metadata. This will cause downstream tasks to fail.

## Step 5 -- write output to disk
Write all results to: <repo_dir>/spack_metadata.txt

The file must contain the following sections in order. Include the `Note:` lines, they are for the user.

  BUILD_SYSTEM:
    Note: Use this to select the correct Spack base class (e.g. cmake → CMakePackage).
    <value>
  FEATURES: 
    <comma-separated list from Step 2>

  --- CMAKE MAPPINGS ---
    Note: Each line is a direct instruction: emit exactly the spack abstraction shown, and consult the evidence field to resolve ambiguity or determine constraints.
    <all mapping lines from Step 4>

  --- SUMMARY ---
  variants: ...
  dependencies: ...
  conflicts: ...
  compilers: ...

This file is the bridge between this skill and other Spack-adjacent tasks.