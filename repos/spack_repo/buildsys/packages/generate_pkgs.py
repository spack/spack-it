import os

# Mapping from filenames (no .py) to class names (without "Package" suffix)
filename_to_class = {
    "autotools": "Autotools",
    "bundle": "Bundle",
    "cached_cmake": "CachedCMake",
    "cmake": "CMake",
    "compiler": "Compiler",
    "cuda": "Cuda",
    "gnu": "GNUMirror",
    "lua": "Lua",
    "makefile": "Makefile",
    "maven": "Maven",
    "meson": "Meson",
    "msbuild": "MSBuild",
    "nmake": "NMake",
    "oneapi": "IntelOneApi",
    "perl": "Perl",
    "python": "Python",
    "qmake": "QMake",
    "rocm": "ROCm",
    "scons": "SCons",
    "sourceforge": "Sourceforge",
    "sourceware": "Sourceware",
    "xorg": "Xorg",
}

for filename, cls in filename_to_class.items():
    dir_name = f"generic_{filename}"
    os.makedirs(dir_name, exist_ok=True)

    with open(os.path.join(dir_name, "package.py"), "w") as f:
        f.write(
            f"from spack_repo.builtin.build_systems.{filename} import {cls}Package\n"
        )
        f.write("from spack.package import *\n\n\n")
        f.write(f"class Generic{cls}({cls}Package):\n")
        f.write("    pass\n")
