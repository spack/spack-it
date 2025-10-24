# first delete from inside the neo4j console
# MATCH (n)
# DETACH DELETE n
# spack python extraction/graph.py --input data/packages.pkl

import argparse
import os
import pickle

from neomodel import (
    ArrayProperty,
    BooleanProperty,
    RelationshipTo,
    StringProperty,
    StructuredNode,
    StructuredRel,
    config,
)

config.DATABASE_URL = os.getenv("NEO4J_URI")


class DependsOnRel(StructuredRel):
    condition = StringProperty()
    types = ArrayProperty(StringProperty())


class Package(StructuredNode):
    name = StringProperty(unique_index=True)
    doc = StringProperty()
    homepage = StringProperty()
    url = StringProperty()
    git = StringProperty()
    tags = ArrayProperty(StringProperty())
    maintainers = ArrayProperty(StringProperty())
    virtual = BooleanProperty()
    has_code = BooleanProperty()
    extendable = BooleanProperty()
    licenses = ArrayProperty(StringProperty())
    build_systems = ArrayProperty(StringProperty())
    recipe = StringProperty()

    dependencies = RelationshipTo("Package", "DEPENDS_ON", model=DependsOnRel)

    variants = RelationshipTo("Variant", "HAS_VARIANT")


class Dependency(StructuredNode):
    spec = StringProperty()
    types = ArrayProperty(StringProperty())
    condition = StringProperty()


class VariantValue(StructuredNode):
    value = StringProperty()
    condition = StringProperty()


class Variant(StructuredNode):
    name = StringProperty()
    default = StringProperty()
    description = StringProperty()
    condition = StringProperty()

    values = RelationshipTo("VariantValue", "HAS_VALUE")


def insert_pkg(pkg_obj):
    # inserts package node
    existing_pkg = Package.nodes.get_or_none(name=pkg_obj.name)
    if not existing_pkg:
        pkg = Package(
            name=pkg_obj.name,
            doc=pkg_obj.__doc__,
            homepage=pkg_obj.homepage,
            url=pkg_obj.url,
            git=pkg_obj.git,
            tags=pkg_obj.tags,
            maintainers=pkg_obj.maintainers,
            virtual=pkg_obj.virtual,
            has_code=pkg_obj.has_code,
            extendable=pkg_obj.extendable,
            licenses=pkg_obj.licenses,
            build_systems=pkg_obj.build_systems,
            recipe=pkg_obj.recipe,
        ).save()


def insert_deps(pkg_obj):
    pkg = Package.nodes.get(name=pkg_obj.name)

    # --- Link Dependencies ---
    for dep in pkg_obj.dependencies:
        dep_pkg_name = dep.pkg_name
        types = dep.types
        condition = dep.condition

        dep_pkg = Package.nodes.get_or_none(name=dep_pkg_name)
        if dep_pkg:
            pkg.dependencies.connect(dep_pkg, {"types": types, "condition": condition})


def insert_variants(pkg_obj):
    pkg = Package.nodes.get(name=pkg_obj.name)

    for var in pkg_obj.variants:
        # Create Variant node
        variant_node = Variant(
            name=var.name,
            default=var.default,
            description=var.description,
            condition=var.condition,
        ).save()

        # Handle values (assumes list of VariantValue)
        values = var.values if isinstance(var.values, list) else []
        for val in values:
            val_node = VariantValue(
                value=str(val.value), condition=val.condition or ""
            ).save()
            variant_node.values.connect(val_node)

        # Connect Variant to Package
        pkg.variants.connect(variant_node)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", help="filename to a pickled file that has a list of Packages"
)


ARGS = parser.parse_args()
# load pkgs into the global namespace for use in all packages
with open(ARGS.input, "rb") as f:
    pkgs = pickle.load(f)

for pkg in pkgs.values():
    insert_pkg(pkg)

for pkg in pkgs.values():
    insert_deps(pkg)

for pkg in pkgs.values():
    insert_variants(pkg)
