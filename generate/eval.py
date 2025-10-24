import ast
import copy
import pickle

from extraction.package_schema import Package

BUILDSYS_FILES_PATH = "data/buildsys.pkl"


def get_base_class_deps() -> dict:
    """Load build system dependency mapping from base class to dependencies."""
    with open(BUILDSYS_FILES_PATH, "rb") as f:
        buildsys = pickle.load(f)
        # ex: {"cmake": list of dependencies}
        return {list(p.base_classes)[0]: p.dependencies for p in buildsys.values()}


def remove_base_class_attr(pkg: Package, base_class_deps: dict, attr: str):
    """Remove build-system-provided elements from a given attribute of the package."""
    items_to_remove = [
        item
        for base_class in pkg.base_classes
        for item in base_class_deps.get(base_class, [])
    ]
    return [item for item in getattr(pkg, attr) if item not in items_to_remove]


def deps_score(orig_pkg: Package, new_pkg: Package) -> float:
    base_class_deps = get_base_class_deps()

    orig_pkg_deps = remove_base_class_attr(
        orig_pkg, base_class_deps, attr="dependencies"
    )
    new_pkg_deps = remove_base_class_attr(new_pkg, base_class_deps, attr="dependencies")

    matches = {}
    for orig_dep in orig_pkg_deps:
        # strict match on the pkg_name
        matches_pkg_name = [d for d in new_pkg_deps if d.pkg_name == orig_dep.pkg_name]

        # find the most similar match based on the spec, condition, and types
        candidates = {}

        for match in matches_pkg_name:
            spec_score = 0
            condition_score = 0
            types_score = 0
            if orig_dep.spec == match.spec:
                spec_score = 1
            if orig_dep.condition == match.condition:
                condition_score = 1

            types_score = len(set(orig_dep.types) & set(match.types)) / max(
                len(orig_dep.types), 1
            )

            # get 0.5 automatically for matching the package name
            total_score = (
                (1 * 0.6)
                + (types_score * 0.2)
                + (spec_score * 0.1)
                + (condition_score * 0.1)
            )

            candidates[str(match)] = total_score
        # get the best candidate
        if candidates:
            matched_dep = max(candidates, key=candidates.get)
            matched_score = candidates[matched_dep]
        else:
            matched_dep = None
            matched_score = 0

        matches[str(orig_dep)] = {"match": matched_dep, "score": matched_score}

    scores = [m["score"] for m in matches.values()]

    # protect against divide by zero error
    return sum(scores) / max(len(scores), 1)


def cmake_args_score(orig_pkg: Package, new_pkg: Package) -> tuple[float, int]:
    # # returns score and how many extra variants are created by the new package
    class DefineArgExtractor(ast.NodeVisitor):
        def __init__(self):
            self.keys: list[str] = []
            self._seen: set[str] = set()  # dedup safeguard
            self.dicts: dict[str, dict[str, str]] = {}
            self.aliases: dict[str, str] = {}
            self.list_vars: set[str] = set()
            # NEW: hold lists of 2-tuples like [("SOL2_TESTS","tests"), ...]
            self.tuple_lists: dict[str, list[tuple[str, str]]] = {}

        # -----------------------
        # Helpers
        # -----------------------
        def _append_key_once(self, k: str):
            if k not in self._seen:
                self._seen.add(k)
                self.keys.append(k)

        def _add_from_dashD(self, s: str):
            if not isinstance(s, str):
                return
            if not s.startswith("-D"):
                return
            body = s[2:]
            key = body.split("=", 1)[0].strip()
            if key:
                self._append_key_once(key)

        def _harvest_str_like(self, node: ast.AST):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self._add_from_dashD(node.value)

            elif (
                isinstance(node, ast.BinOp)
                and isinstance(node.left, ast.Constant)
                and isinstance(node.left.value, str)
            ):
                self._add_from_dashD(node.left.value)

            elif isinstance(node, ast.JoinedStr):
                for v in node.values:
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        self._add_from_dashD(v.value)
                        break

            elif isinstance(node, ast.List):
                for elt in node.elts:
                    self._harvest_str_like(elt)

            # "-D...".format(...)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"format"}
            ):
                self._harvest_str_like(node.func.value)

            # else: ignore

        def _harvest_from_list_assignment(self, list_node: ast.List):
            for elt in list_node.elts:
                self._harvest_str_like(elt)

        def _resolve_define_func_name(self, call: ast.Call) -> str | None:
            """Return 'define'/'define_from_variant'/'from_variant' if matches (considering aliases), else None."""
            if isinstance(call.func, ast.Attribute):
                if call.func.attr in {"define", "define_from_variant", "from_variant"}:
                    return call.func.attr
            elif isinstance(call.func, ast.Name):
                return self.aliases.get(call.func.id)
            return None

        def _collect_from_mapping_call(
            self,
            call: ast.Call,
            left_name: str,
            right_name: str,
            mapping: dict[str, str],
        ):
            """
            If first arg is the tuple's left name -> collect mapping.keys()
            If first arg is the tuple's right name -> collect mapping.values()
            """
            if not call.args:
                return
            first = call.args[0]
            if isinstance(first, ast.Name):
                if first.id == left_name:
                    for k in mapping.keys():
                        self._append_key_once(k)
                elif first.id == right_name:
                    for v in mapping.values():
                        self._append_key_once(v)

        # NEW: like _collect_from_mapping_call, but for list-of-2-tuples
        def _collect_from_tuple_list_call(
            self,
            call: ast.Call,
            left_name: str,
            right_name: str,
            pairs: list[tuple[str, str]],
        ):
            if not call.args:
                return
            first = call.args[0]
            if isinstance(first, ast.Name):
                if first.id == left_name:
                    for k, _ in pairs:
                        self._append_key_once(k)
                elif first.id == right_name:
                    for _, v in pairs:
                        self._append_key_once(v)

        # -----------------------
        # Visitors
        # -----------------------
        def visit_ClassDef(self, node):
            for stmt in node.body:
                self.visit(stmt)

        def visit_FunctionDef(self, node):
            for stmt in node.body:
                self.visit(stmt)

        def visit_Assign(self, node):
            # Track mapping = {...}
            if isinstance(node.targets[0], ast.Name) and isinstance(
                node.value, ast.Dict
            ):
                varname = node.targets[0].id
                if all(
                    isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
                    for k, v in zip(node.value.keys, node.value.values)
                ):
                    self.dicts[varname] = {
                        k.value: v.value
                        for k, v in zip(node.value.keys, node.value.values)
                    }

            # Track from_variant = self.define_from_variant
            elif (
                isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
                and node.value.attr in {"define", "define_from_variant", "from_variant"}
            ):
                self.aliases[node.targets[0].id] = node.value.attr

            # Track args = [ ... ] (list of string-likes)
            if isinstance(node.targets[0], ast.Name) and isinstance(
                node.value, ast.List
            ):
                varname = node.targets[0].id
                self.list_vars.add(varname)
                self._harvest_from_list_assignment(node.value)

                # NEW: capture lists of 2-tuples of constants: [("FOO","bar"), ...]
                if all(
                    isinstance(elt, ast.Tuple)
                    and len(elt.elts) == 2
                    and all(
                        isinstance(e, ast.Constant) and isinstance(e.value, str)
                        for e in elt.elts
                    )
                    for elt in node.value.elts
                ):
                    self.tuple_lists[varname] = [
                        (elt.elts[0].value, elt.elts[1].value)  # type: ignore[attr-defined]
                        for elt in node.value.elts
                    ]

            # List comprehension cases:
            #   args = [ self.define_from_variant(... ) for left,right in mapping.items() ]
            #   args = [ self.define_from_variant(... ) for left,right in variant_map ]
            if isinstance(node.targets[0], ast.Name) and isinstance(
                node.value, ast.ListComp
            ):
                lc = node.value
                if len(lc.generators) == 1 and isinstance(lc.elt, ast.Call):
                    gen = lc.generators[0]

                    # Case A: mapping dict's items()
                    if (
                        isinstance(gen.iter, ast.Call)
                        and isinstance(gen.iter.func, ast.Attribute)
                        and gen.iter.func.attr == "items"
                        and isinstance(gen.iter.func.value, ast.Name)
                        and isinstance(gen.target, ast.Tuple)
                        and len(gen.target.elts) == 2
                        and all(isinstance(e, ast.Name) for e in gen.target.elts)
                    ):
                        dict_name = gen.iter.func.value.id
                        mapping = self.dicts.get(dict_name)
                        left_name = gen.target.elts[0].id
                        right_name = gen.target.elts[1].id
                        func_name = self._resolve_define_func_name(lc.elt)
                        if mapping and func_name in {
                            "define",
                            "define_from_variant",
                            "from_variant",
                        }:
                            self._collect_from_mapping_call(
                                lc.elt, left_name, right_name, mapping
                            )

                    # Case B (NEW): list of 2-tuples like variant_map
                    elif (
                        isinstance(gen.iter, ast.Name)
                        and isinstance(gen.target, ast.Tuple)
                        and len(gen.target.elts) == 2
                        and all(isinstance(e, ast.Name) for e in gen.target.elts)
                    ):
                        list_name = gen.iter.id
                        pairs = self.tuple_lists.get(list_name)
                        if pairs:
                            left_name = gen.target.elts[0].id
                            right_name = gen.target.elts[1].id
                            func_name = self._resolve_define_func_name(lc.elt)
                            if func_name in {
                                "define",
                                "define_from_variant",
                                "from_variant",
                            }:
                                self._collect_from_tuple_list_call(
                                    lc.elt, left_name, right_name, pairs
                                )

            self.generic_visit(node)

        def visit_AugAssign(self, node):
            # args += ["-D..."]
            if isinstance(node.target, ast.Name) and node.target.id in self.list_vars:
                self._harvest_str_like(node.value)
            self.generic_visit(node)

        def visit_For(self, node):
            # for left, right in mapping.items():
            if (
                isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr == "items"
                and isinstance(node.iter.func.value, ast.Name)
            ):
                dict_name = node.iter.func.value.id
                mapping = self.dicts.get(dict_name)
                if not mapping:
                    return

                if (
                    isinstance(node.target, ast.Tuple)
                    and len(node.target.elts) == 2
                    and all(isinstance(e, ast.Name) for e in node.target.elts)
                ):
                    left_name = node.target.elts[0].id
                    right_name = node.target.elts[1].id

                    # Walk the body to find ANY nested calls (e.g., args.append(self.define_from_variant(...)))
                    for stmt in node.body:
                        for sub in ast.walk(stmt):
                            if isinstance(sub, ast.Call):
                                func_name = self._resolve_define_func_name(sub)
                                if func_name in {
                                    "define",
                                    "define_from_variant",
                                    "from_variant",
                                }:
                                    self._collect_from_mapping_call(
                                        sub, left_name, right_name, mapping
                                    )

                    # Keep existing replacement/recursive logic for other patterns
                    cmake_var = right_name
                    for val in mapping.values():
                        replacer = LoopVarReplacer(cmake_var, val)
                        for stmt in node.body:
                            stmt_copy = copy.deepcopy(stmt)
                            stmt_copy = ast.fix_missing_locations(stmt_copy)
                            replacer.visit(stmt_copy)
                            self.visit(stmt_copy)

            # NEW: for left, right in variant_map:
            elif (
                isinstance(node.iter, ast.Name)
                and isinstance(node.target, ast.Tuple)
                and len(node.target.elts) == 2
                and all(isinstance(e, ast.Name) for e in node.target.elts)
            ):
                list_name = node.iter.id
                pairs = self.tuple_lists.get(list_name)
                if not pairs:
                    return

                left_name = node.target.elts[0].id
                right_name = node.target.elts[1].id

                # Walk the body to find define* calls and expand over pairs
                for stmt in node.body:
                    for sub in ast.walk(stmt):
                        if isinstance(sub, ast.Call):
                            func_name = self._resolve_define_func_name(sub)
                            if func_name in {
                                "define",
                                "define_from_variant",
                                "from_variant",
                            }:
                                self._collect_from_tuple_list_call(
                                    sub, left_name, right_name, pairs
                                )

            else:
                self.generic_visit(node)

        def visit_Call(self, node):
            # direct self.define_from_variant("FOO", ...)
            func_name = self._resolve_define_func_name(node)
            if func_name in {"define", "define_from_variant", "from_variant"}:
                if node.args and isinstance(node.args[0], ast.Constant):
                    self._append_key_once(node.args[0].value)

            # args.append/extend into tracked lists
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"append", "extend"}
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in self.list_vars
            ):
                if node.func.attr == "append" and node.args:
                    self._harvest_str_like(node.args[0])
                elif node.func.attr == "extend" and node.args:
                    self._harvest_str_like(node.args[0])

            self.generic_visit(node)

    class LoopVarReplacer(ast.NodeTransformer):
        def __init__(self, varname, value):
            self.varname = varname
            self.value = value

        def visit_Name(self, node):
            if node.id == self.varname and isinstance(node.ctx, ast.Load):
                return ast.Constant(value=self.value)
            return node

    def extract_define_keys(source_code: str) -> list[str]:
        tree = ast.parse(source_code)
        extractor = DefineArgExtractor()
        extractor.visit(tree)
        return extractor.keys

    # rather than ussing an invocation of pkg_class(pkg_class).cmake_args(), we statically extract the cmake args
    # cmake_args() conditionally checks which variants are part of the spec. for example, if +hip is not part of the spec we're evaluating
    # that wouldn't get picked up in cmake_args.
    # because we analyze the recipe statically, this ensures we don't pick up on any variants that would be added to the package class
    # if this was dynamically evaluated (due to the inheritance of build system classes)

    orig_cmake_args = extract_define_keys(orig_pkg.recipe)
    new_cmake_args = extract_define_keys(new_pkg.recipe)
    # strip and lower for better comparison
    orig_cmake_args = {i.strip().lower() for i in orig_cmake_args}
    new_cmake_args = {i.strip().lower() for i in new_cmake_args}
    overlap = len(orig_cmake_args & new_cmake_args)
    # how many extra variants are included by the new recipe
    extras = len(new_cmake_args) - overlap

    if len(orig_cmake_args) == 0:
        return None, extras
    else:
        score = overlap / len(orig_cmake_args)

    return score, extras
