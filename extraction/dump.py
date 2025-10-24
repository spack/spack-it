import pickle
import time

import spack
import spack.tag

from extraction.package import get_pkg_objs


def dump_packages():
    repo = spack.repo.PATH.get_repo("builtin")
    tag_pkgs = spack.tag.packages_with_tags(
        ["e4s", "proxy-app"], installed=False, skip_empty=True
    )

    e4s_objs = get_pkg_objs(tag_pkgs["e4s"], repo)
    e4s_pkgs = set(e4s_objs.keys())
    other_pkgs = set()

    # NOTE: this assumes that the only packages in `pkgs` are e4s
    # since we are looking for top-level dependencies of e4s packages
    for pkg in e4s_objs.values():
        deps = pkg.dependencies
        for dep in deps:
            # we only want the string of the spec package name, not the spec itself because that can contain other metadata
            other_pkgs.add(spack.spec.Spec(dep.spec).name)

    # get lists of all virtuals and provider specs
    # compilers, libraries, other important packages
    virtual_pkgs = set(spack.repo.PATH.provider_index.providers.keys())

    provider_pkgs = {
        spack.spec.Spec(provider).name
        for pkg in virtual_pkgs
        # providers_for returns a list of specs so we're just flattening the list
        for provider in spack.repo.PATH.providers_for(pkg)
    }

    # Union all potential extra specs
    other_pkgs.update(provider_pkgs)

    # remove everything we've already collected
    other_pkgs -= e4s_pkgs

    other_objs = get_pkg_objs(other_pkgs, repo)
    all_objs = e4s_objs | other_objs

    filename = f"data/packages-{int(time.time())}.pkl"

    with open(filename, "wb") as f:
        pickle.dump(all_objs, f)

    print(f"saved data to {filename}")


def dump_generic_buildsys_packages():
    # we use generic packages that inherit from spack's build system classes in order to subtract inherited dependencies and variants
    repo = spack.repo.PATH.get_repo("buildsys")
    pkgs = [
        "generic-autotools",
        "generic-bundle",
        "generic-cached-cmake",
        "generic-cmake",
        "generic-compiler",
        "generic-cuda",
        "generic-gnu",
        "generic-lua",
        "generic-makefile",
        "generic-maven",
        "generic-meson",
        "generic-msbuild",
        "generic-oneapi",
        "generic-perl",
        "generic-python",
        "generic-qmake",
        "generic-rocm",
        "generic-scons",
        "generic-sourceforge",
        "generic-sourceware",
        "generic-xorg",
    ]
    buildsys_objs = get_pkg_objs(pkgs, repo)

    filename = f"data/buildsys-{int(time.time())}.pkl"

    with open(filename, "wb") as f:
        pickle.dump(buildsys_objs, f)

    print(f"saved data to {filename}")


def dump_new_pkgs():
    """packages to experiment with after the cutoff date of a model: these are all cmake packages added ≥ oct-1-24, after the gpt-5 cutoff date"""

    repo = spack.repo.PATH.get_repo("builtin")
    pkgs = [
        "libftdi",
        "libmetatensor",
        "libmetatomic-torch",
        "vir-simd",
        "dplasma",
        "fairroot",
        "fairsoft-bundle",
        "ddc",
        "gribjump",
        "jonquil",
        "mstore",
        "dolfinx-mpc",
        "mscclpp",
        "tempestextremes",
        "mamba",
        "openfpgaloader",
        "alpscore",
        "quest",
        "elastix",
        "regenie",
        "ambertools",
        "cgsi-gsoap",
        "cbqn",
        "sopt",
        "cubature",
        "purify",
        "aocl-da",
        "otf-cpt",
        "soqt",
        "flux",
        "prometheus-cpp",
        "mpidiff",
        "verdict",
        "hip-tests",
        "fmi4cpp",
        "byte-lite",
        "neofoam",
        "costo",
        "fusion-io",
        "rocjpeg",
        "tomlplusplus",
        "yyjson",
        "pace",
        "indicators",
        "alps",
        "sphexa",
        "pumgen",
        "hard",
        "fluidnumerics-self",
        "kynema",
    ]

    objs = get_pkg_objs(pkgs, repo)

    filename = f"data/packages-new-{int(time.time())}.pkl"

    with open(filename, "wb") as f:
        pickle.dump(objs, f)

    print(f"saved data to {filename}")


if __name__ == "__main__":
    dump_packages()
    dump_generic_buildsys_packages()
    # dump_new_pkgs()
