from dataclasses import dataclass, field
from typing import Optional

from spack.fetch_strategy import FetchStrategy

# TODO add some sort of annotation to each field in order to map values?
## document each field and make sure schema makes sense...this will be helpful context to the prompt


@dataclass(kw_only=True)
class Dependency:
    condition: Optional[str] = None
    pkg_name: str
    spec: str  # this can have info other than a package name so if you need the spec name need to parrse it as spec...(is there a cheaper way vs Spec()?)
    patches: Optional[None] = None  # TODO there should be a patch type
    types: tuple


@dataclass(kw_only=True)
class Conflict:
    spec: str
    conditions: list = field(default_factory=list)


@dataclass(kw_only=True)
class ConflictCondition:
    condition: str
    msg: Optional[str] = None


@dataclass(kw_only=True)
class Requirement:
    spec: str
    conditions: list = field(default_factory=list)


@dataclass(kw_only=True)
class RequirementCondition:
    conditions: list[str]
    type: str
    msg: Optional[str] = None


@dataclass(kw_only=True)
class Resource:
    condition: Optional[str] = None
    name: str
    fetcher: str
    destination: str
    placement: str


@dataclass(kw_only=True)
class Version:
    version: str
    branch: Optional[str] = None
    commit: Optional[str] = None
    deprecated: Optional[bool] = False
    extension: Optional[str] = None
    get_full_repo: Optional[bool] = False
    git: Optional[str] = None
    preferred: Optional[bool] = False
    sha256: Optional[str] = None
    submodules: Optional[bool | str] = None
    tag: Optional[str] = None
    url: Optional[str] = None
    expand: Optional[str] = None
    fetcher: FetchStrategy
    isdevelop: bool
    is_prerelease: bool
    no_cache: Optional[bool] = False


@dataclass(kw_only=True)
class VariantValue:
    value: str | bool
    condition: Optional[str] = None


@dataclass(kw_only=True)
class Extends:
    spec: str
    condition: str | None


@dataclass(kw_only=True)
class DisableRedistribute:
    condition: str | None
    binary: bool
    source: bool


@dataclass(kw_only=True)
class Variant:
    condition: str
    name: str
    default: str
    description: str
    # can also be DisjointSetsOfValues
    values: list[VariantValue] | dict


@dataclass(kw_only=True)
class Provided:
    condition: Optional[str] = None
    specs: list[str]


@dataclass(kw_only=True)
class ProvidedTogether:
    condition: Optional[str] = None
    specs: set[str]


@dataclass(kw_only=True)
class Patch:
    condition: Optional[str] = None
    url: str
    level: int
    reverse: bool
    ordering_key: tuple


@dataclass(kw_only=True)
class Package:
    # these are trivial to extract and have simple formats
    __doc__: str
    homepage: str
    url: str
    git: str
    tags: list[str]
    maintainers: list[str]
    name: str
    virtual: bool
    has_code: bool
    extendable: bool
    # these require manual extraction
    licenses: list[str]
    versions: list[Version]
    variants: list[Version]
    dependencies: list[dict]
    conflicts: list[dict]
    requirements: list[dict]
    resources: list[dict]
    provided: list[dict]
    provided_together: list[dict]
    extendees: list[dict]
    disable_redistribute: list[dict]
    patches: list[dict]
    build_systems: set[str]
    recipe: str
    base_classes: set
