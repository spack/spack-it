import ast
import importlib
import sys

import spack
import spack.fetch_strategy as fs
from spack.package_base import PackageBase
from spack.spec import Spec

from extraction.package_schema import (
    Conflict,
    ConflictCondition,
    Dependency,
    DisableRedistribute,
    Extends,
    Package,
    Patch,
    Provided,
    ProvidedTogether,
    Requirement,
    RequirementCondition,
    Resource,
    Variant,
    VariantValue,
    Version,
)

# TODO other extractions
# represent relationships in the graph
# splice_specs -> 4/23/25, this isn't present in any packages, search for `can_splice`
# _has_make_target
# cuda:compute_capabilities
# cmake things: define/define_from_variant, cmake_args
# for getting versions/hashes: find_valid_url_for_version
# install targets/executables: if not defined then don't make it up
# list_depth, sanity_check in packagebase
# store the file itself and the makefile here? THE MAKEFILE ATTRIBUTES
# anything extra in the builder class? run_after_callbacks etc
# preferred_version
# the Version instance has other metadata that might be helpful in the future (bounds)


class ExtractionError(Exception):
    pass


def sgetattr(obj, name):
    """safe getattr"""
    return getattr(obj, name, None)


def extract_base(pkg, attrs):
    data = {}

    for attr in attrs:
        data[attr] = sgetattr(pkg, attr)

    return data


def spec_condition(spec: spack.spec.Spec) -> str | None:
    # often Specs are used to specify the conditions for an aspect of metadata
    # this returns the string representation of the spec
    # or None if there is no condition

    if str_spec := str(spec):
        return str_spec

    return None


def extract_patches_shared(
    patch_dict: dict[spack.spec.Spec, list[spack.patch.Patch]],
) -> list[Patch]:
    """
    extracts metadata from a dict of spack.patch.Patches and conditions
    made to be agnostic so it can be used with other extraction helpers
    """

    def fmt_path_or_url(path_or_url: str) -> str:
        # converts from an absolute url
        # eg: /Users/bob/src/spack/spack/var/spack/repos/builtin/packages/slepc/install_name_371.patch
        # to a patch url hosted on github
        # THIS WILL NEED TO BE UPDATED/MIGRATED WHEN THE PACKAGES REPO IS CREATED

        if path_or_url.startswith("http"):
            return path_or_url

        split_point = "repos/spack_repo/builtin/packages"

        trimmed_path = split_point + path_or_url.split(split_point, 1)[1]

        url = f"https://raw.githubusercontent.com/spack/spack-packages/refs/heads/develop/{trimmed_path}"

        return url

    patches = []

    # there is a list of patches for every spec/condition
    for spec, patch_list in patch_dict.items():
        condition = spec_condition(spec)

        for patch in patch_list:
            patches.append(
                Patch(
                    condition=condition,
                    url=fmt_path_or_url(patch.path_or_url),
                    level=patch.level,
                    reverse=patch.reverse,
                    ordering_key=patch.ordering_key,
                )
            )

    return patches


def extract_licenses(pkg: PackageBase) -> list[str]:
    return list(getattr(pkg, "licenses", {}).values())


def extract_versions(pkg: PackageBase, spec: str) -> list[Version]:
    versions = []

    for obj, metadata in getattr(pkg, "versions", {}).items():
        # some packages represent `submodules` with different types
        # which we can't always handle
        if not isinstance(
            metadata.get("submodules"),
            (
                bool,
                str,
                type(None),
            ),
        ):
            metadata["submodules"] = str(metadata.get("submodules"))

        # not sure why the spec needs to be called this way
        fetcher = fs.for_package_version(pkg(Spec(spec)), obj)
        versions.append(
            Version(
                version=obj.string,
                fetcher=fetcher,
                isdevelop=obj.isdevelop(),
                is_prerelease=obj.is_prerelease(),
                **metadata,
            )
        )

    return versions


def extract_variants(pkg: PackageBase) -> list[Variant]:
    variants = []

    # spec defines the condition, variants_dict is in {"cuda": Variant(cuda)} format
    for spec, variants_dict in getattr(pkg, "variants", {}).items():
        condition = spec_condition(spec)

        # iterate over Variants
        for variant in variants_dict.values():
            raw_values = variant.values
            variant_values = []

            # variant values should always be a list
            if raw_values is None:
                variant_values = []
            # sometimes variants are expressed in complex formats
            elif isinstance(raw_values, spack.variant.DisjointSetsOfValues):
                # example: {'sets': [{'none'}, {'35', '90a', '89', '120', '72', '12', '100', '32', '61', '10', '13', '101a', '30', '120a', '11', '87', '52', '80', '101', '60', '53', '20', '50', '90', '86', '100a', '75', '21', '62', '70', '37'}], 'feature_values': ('35', '90a', '89', '120', '72', '12', '100', '32', '61', '10', '13', '101a', '30', '120a', '11', '87', '52', '80', '101', '60', '53', '20', '50', '90', '86', '100a', '75', '21', '62', '70', '37'), 'default': 'none', 'multi': True, 'error_fmt': "the value 'none' is mutually exclusive with any of the other values"}
                variant_values = raw_values.__dict__
            else:
                # values can be plain (str, bool) or conditional (with a 'when' clause)
                for value in raw_values:
                    if isinstance(value, (str, bool)):
                        variant_values.append(VariantValue(value=value))
                    elif isinstance(value, spack.variant.ConditionalValue):
                        val_obj = VariantValue(value=value.value)
                        if value.when:
                            val_obj.condition = spec_condition(value.when)
                        variant_values.append(val_obj)

            variants.append(
                Variant(
                    condition=condition,
                    name=variant.name,
                    default=variant.default,
                    description=variant.description,
                    values=variant_values,
                )
            )

    return variants


def extract_dependencies(pkg: PackageBase) -> list[Dependency]:
    dependencies = []

    for spec, dep_dict in getattr(pkg, "dependencies", {}).items():
        condition = spec_condition(spec)

        for dep in dep_dict.values():
            dependencies.append(
                Dependency(
                    condition=condition,
                    pkg_name=Spec(dep.spec).name,
                    spec=str(dep.spec),
                    patches=extract_patches_shared(dep.patches),
                    types=spack.deptypes.flag_to_tuple(dep.depflag),
                )
            )

    return dependencies


def extract_conflicts(pkg: PackageBase) -> list[Conflict]:
    conflicts = []

    for spec, conflict in getattr(pkg, "conflicts", {}).items():
        conflict_obj = Conflict(spec=str(spec))

        for condition, msg in conflict:
            conflict_obj.conditions.append(
                ConflictCondition(condition=str(condition), msg=msg)
            )
        conflicts.append(conflict_obj)

    return conflicts


def extract_requirements(pkg: PackageBase) -> list[Requirement]:
    requirements = []

    for spec, reqs in getattr(pkg, "requirements", {}).items():
        # reqs is a list of tuples
        for req in reqs:
            req_obj = Requirement(spec=str(spec))
            # req looks like ((%gcc@4.9:,), 'one_of', None)
            req_obj.conditions.append(
                RequirementCondition(
                    conditions=[str(condition) for condition in req[0]],
                    type=req[1],
                    msg=req[2],
                )
            )

            requirements.append(req_obj)

    return requirements


def extract_resources(pkg: PackageBase) -> list[Resource]:
    resources = []

    for spec, res_list in getattr(pkg, "resources", {}).items():
        # resources is a list of Resources
        condition = spec_condition(spec)

        for res in res_list:
            resources.append(
                Resource(
                    condition=condition,
                    name=res.name,
                    fetcher=str(res.fetcher),
                    destination=res.destination,
                    placement=res.placement,
                )
            )

    return resources


def extract_provided(pkg: PackageBase) -> list[Provided]:
    provided = []

    for spec, provided_specs in getattr(pkg, "provided", {}).items():
        # TODO helper function for this
        # for some reason, in provided, the conditions include the pkg name by default
        # so we remove this to be consistent with other classes of data in the dataset
        str_spec = str(spec).replace(pkg.name, "")
        condition = spec_condition(str_spec)

        provided.append(
            Provided(condition=condition, specs=[str(spec) for spec in provided_specs])
        )

    return provided


def extract_provided_together(
    pkg: PackageBase,
) -> list[ProvidedTogether]:
    provided_together = []

    for spec, provided_spec_sets in getattr(pkg, "provided_together", {}).items():
        # this comes in the format of a condition: list of spec sets for this condition

        # for some reason, in provided, the conditions include the pkg name by default
        # so we remove this to be consistent with other classes of data in the dataset
        str_spec = str(spec).replace(pkg.name, "")
        condition = spec_condition(str_spec)

        for spec_set in provided_spec_sets:
            # iterate over the spec sets that go together for this condition
            provided_together.append(
                ProvidedTogether(
                    condition=condition,
                    specs={str(provided_spec) for provided_spec in spec_set},
                )
            )

    return provided_together


def extract_extendees(pkg: PackageBase) -> list[Extends]:
    extendees = []

    # https://github.com/spack/spack/blob/b42ef1e7b8e46f1632b5d21dee0d5138fa659db4/lib/spack/spack/directives.py#L462
    # pkg.extendees[dep_spec.name] = (dep_spec, when_spec)
    for dep_spec, when_spec in getattr(pkg, "extendees", {}).values():
        extendees.append(
            Extends(spec=str(dep_spec), condition=spec_condition(when_spec))
        )

    return extendees


def extract_disable_redistribute(pkg: PackageBase) -> list[DisableRedistribute]:
    disable_redistribute = []

    for spec, policy in getattr(pkg, "disable_redistribute", {}).items():
        condition = spec_condition(spec)

        disable_redistribute.append(
            DisableRedistribute(
                condition=condition, binary=policy.binary, source=policy.source
            )
        )

    return disable_redistribute


def extract_patches(pkg: PackageBase) -> list[Patch]:
    """
    extracts patches from the package recipe
    """

    return extract_patches_shared(getattr(pkg, "patches", {}))


def extract_build_systems(variants: list[Variant]) -> set[str]:
    return {val.value for v in variants if v.name == "build_system" for val in v.values}


def extract_recipe(pkg: PackageBase, repo: spack.repo.Repo) -> str:
    pkg_path = repo.filename_for_package_name(pkg.name)
    with open(pkg_path, "r") as recipe:
        return recipe.read()


def extract_base_classes(recipe: str) -> set:
    tree = ast.parse(recipe)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            return {base.id for base in node.bases if isinstance(base, ast.Name)}
    return set()


# tuple of top-level fields to extract directly
BASE_FIELDS = (
    "__doc__",
    "homepage",
    "url",
    "git",
    "tags",
    "maintainers",
    "name",
    "virtual",
    "has_code",
    "extendable",
)

EXTRACTORS = {
    "licenses": extract_licenses,
    "versions": lambda pkg: set(),
    "variants": extract_variants,
    "dependencies": extract_dependencies,
    "conflicts": extract_conflicts,
    "requirements": extract_requirements,
    "resources": extract_resources,
    "provided": extract_provided,
    "provided_together": extract_provided_together,
    "extendees": extract_extendees,
    "disable_redistribute": extract_disable_redistribute,
    "patches": extract_patches,
    # no-op so the Package can get initialized; build_systems will be provided later
    "build_systems": lambda pkg: set(),
    "recipe": lambda pkg: set(),
    "base_classes": lambda pkg: set(),
}


def get_pkg_obj(pkg_name: str, repo: spack.repo.Repo) -> Package:
    importlib.invalidate_caches()

    for modname in list(sys.modules):
        if (
            modname == f"spack_repo.{repo.namespace}.packages.{pkg_name}"
            or modname.startswith(f"spack_repo.{repo.namespace}.packages.{pkg_name}")
        ):
            sys.modules.pop(modname, None)

    sys.modules.pop(f"spack_repo.{repo.namespace}.packages", None)

    try:
        # if the package is in another repo, we want to query the package class specifically from that repo
        # in spack.yaml, we set the built in repository as having priority because we don't want the dependencies
        # of the spec we're experimenting with getting pulled into the DAG if they're in the experimental repo,
        # we want to use the packages that are written and "accurate"
        pkg_class = repo.get_pkg_class(f"{repo.namespace}.{pkg_name}")
    except spack.repo.UnknownPackageError as exc:
        raise ExtractionError(f"error when trying to load package from repo: {exc}")
    except spack.repo.RepoError as exc:
        raise ExtractionError(f"RepoError: {exc}")

    pkg = Package(
        **extract_base(pkg_class, BASE_FIELDS),
        **{key: extractor(pkg_class) for key, extractor in EXTRACTORS.items()},
    )

    # we depend on the pkg.variants to extract the build systems, so we need to separate this assignment
    pkg.build_systems = extract_build_systems(pkg.variants)
    pkg.versions = extract_versions(pkg_class, spec=f"{repo.namespace}.{pkg_name}")
    pkg.recipe = extract_recipe(pkg, repo)
    pkg.base_classes = extract_base_classes(pkg.recipe)
    return pkg


def get_pkg_objs(pkg_names: list[str], repo: spack.repo.Repo) -> dict[str, Package]:
    pkg_dict = {}

    for pkg_name in pkg_names:
        obj = get_pkg_obj(pkg_name, repo)
        if obj:
            pkg_dict[pkg_name] = obj

    return pkg_dict
