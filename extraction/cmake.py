# extract information from cmake files inside a directory
# usage: parse_repo(directory_name) -> string of cmake metadata found

import logging
from pathlib import Path

import yaml
from cmake_parser.ast import (
    Break,
    Command,
    ForEach,
    Function,
    If,
    Include,
    Macro,
    Math,
    Option,
    Return,
    Set,
    Unset,
    While,
)
from cmake_parser.error import CMakeParseError
from cmake_parser.parser import parse_tree

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

EXCLUDE_DIRS = {
    "test",
    "tests",
    "ctest",
    "examples",
    "example",
    "doc",
    "docs",
    "blt",
    ".git",
    "__pycache__",
}


class CMakeContext:
    def __init__(self):
        self.cache_sets = []
        self.normal_sets = []
        self.commands = []
        self.options = []
        self.ifs = []

    def __str__(self):
        return (
            f"Cache sets: \n {yaml.dump(self.cache_sets)}, "
            f"Normal sets: \n {yaml.dump(self.normal_sets)}, "
            f"Commands: \n {yaml.dump(self.commands)}, "
            f"Options: \n {yaml.dump(self.options)}, "
            f"If statements: \n {self.format_ifs(self.ifs)}"
        )

    def format_ifs(self, if_blocks):
        def flatten_if(if_block):
            out = {
                "condition": if_block["condition"],
                "if_true": {
                    k: if_block["if_true"].get(k, [])
                    for k in ["commands", "cache_sets", "normal_sets", "options"]
                },
                "if_false": (
                    {
                        k: if_block["if_false"].get(k, [])
                        for k in ["commands", "cache_sets", "normal_sets", "options"]
                    }
                    if if_block["if_false"]
                    else None
                ),
            }
            # Recursively embed nested ifs
            nested_true = if_block["if_true"].get("ifs", [])
            nested_false = (
                if_block["if_false"].get("ifs", []) if if_block["if_false"] else []
            )
            if nested_true:
                out["if_true"]["ifs"] = [flatten_if(b) for b in nested_true]
            if nested_false:
                out["if_false"]["ifs"] = [flatten_if(b) for b in nested_false]
            return out

        return yaml.dump([flatten_if(b) for b in if_blocks])


def safe_get(lst, index, default=None):
    return lst[index] if 0 <= index < len(lst) else default


def extract_from_ast(nodes, context):
    def parse_set(items):
        # https://cmake.org/cmake/help/latest/command/set.html
        if "CACHE" in items:
            set_cache = {
                "variable": items[0],
                "value": items[1],
                "type": items[3],
                "docstring": safe_get(items, 4, ""),
            }
            if len(items) == 6:
                set_cache["force"] = items[5]
            context.cache_sets.append(set_cache)

        else:
            set_normal = {"variable": items[0], "values": items[1:]}
            context.normal_sets.append(set_normal)

    def parse_command(node, items):
        command = {"cmd": node.identifier, "args": items}
        context.commands.append(command)

    def parse_option(items):
        option = {
            "name": items[0],
            "help_text": items[1],
            "default": safe_get(items, 2),
        }
        context.options.append(option)

    def parse_if(node):
        condition = [token.value for token in node.args]

        # Recursively extract into new temporary CMakeContext instances
        true_context = CMakeContext()
        extract_from_ast(node.if_true, context=true_context)

        if_context = {
            "condition": condition,
            "if_true": {
                "cache_sets": true_context.cache_sets,
                "normal_sets": true_context.normal_sets,
                "commands": true_context.commands,
                "options": true_context.options,
                "ifs": true_context.ifs,
                "whiles": getattr(true_context, "whiles", []),
                "breaks": getattr(true_context, "breaks", []),
                "foreaches": getattr(true_context, "foreaches", []),
            },
            "if_false": None,
        }

        if node.if_false:
            false_context = CMakeContext()
            extract_from_ast(node.if_false, context=false_context)

            if_context["if_false"] = {
                "cache_sets": false_context.cache_sets,
                "normal_sets": false_context.normal_sets,
                "commands": false_context.commands,
                "options": false_context.options,
                "ifs": false_context.ifs,
                "whiles": getattr(false_context, "whiles", []),
                "breaks": getattr(false_context, "breaks", []),
                "foreaches": getattr(false_context, "foreaches", []),
            }

        context.ifs.append(if_context)

    def parse_while(node):
        condition = [token.value for token in node.args]

        # Recursively extract the body into a new temporary CMakeContext
        body_context = CMakeContext()
        extract_from_ast(node.body, context=body_context)

        while_context = {
            "condition": condition,
            "body": {
                "cache_sets": body_context.cache_sets,
                "normal_sets": body_context.normal_sets,
                "commands": body_context.commands,
                "options": body_context.options,
                "ifs": body_context.ifs,
                "whiles": getattr(body_context, "whiles", []),
                "breaks": getattr(body_context, "breaks", []),
                "foreaches": getattr(body_context, "foreaches", []),
            },
        }

        # Add whiles list to context if it doesn't exist
        if not hasattr(context, "whiles"):
            context.whiles = []
        context.whiles.append(while_context)

    def parse_break(node):
        # Break statements are simple - they don't have args or body
        break_context = {
            "line": getattr(node, "line", None),
            "column": getattr(node, "column", None),
        }

        # Add breaks list to context if it doesn't exist
        if not hasattr(context, "breaks"):
            context.breaks = []
        context.breaks.append(break_context)

    def parse_foreach(node):
        # Parse the foreach arguments
        items = [token.value for token in node.args]

        # Recursively extract the body into a new temporary CMakeContext
        body_context = CMakeContext()
        extract_from_ast(node.body, context=body_context)

        foreach_context = {
            "variable": items[0] if items else None,
            "items": items[1:] if len(items) > 1 else [],
            "body": {
                "cache_sets": body_context.cache_sets,
                "normal_sets": body_context.normal_sets,
                "commands": body_context.commands,
                "options": body_context.options,
                "ifs": body_context.ifs,
                "whiles": getattr(body_context, "whiles", []),
                "breaks": getattr(body_context, "breaks", []),
                "foreaches": getattr(body_context, "foreaches", []),
            },
        }

        # Add foreaches list to context if it doesn't exist
        if not hasattr(context, "foreaches"):
            context.foreaches = []
        context.foreaches.append(foreach_context)

    for node in nodes:
        items = (
            [token.value for token in node.args]
            if hasattr(node, "args") and node.args
            else []
        )

        if isinstance(node, Set):
            parse_set(items)
        elif isinstance(node, Command):
            parse_command(node, items)
        elif isinstance(node, Option):
            parse_option(items)
        elif isinstance(node, If):
            parse_if(node)
        elif isinstance(node, While):
            parse_while(node)
        elif isinstance(node, Break):
            parse_break(node)
        elif isinstance(node, ForEach):
            parse_foreach(node)
        elif isinstance(node, Math | Unset | Include | Function | Macro | Return):
            # function could have some interesting stuff in the future
            # could do something like try to find the dependencies and variants first, then move to conflicts in If
            # DO NOT TRY TO PARSE
            pass
        else:
            log.warning(f"unknown node type: {node}")


def find_cmake_files(dir: str):
    root = Path(dir)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if not any(part in EXCLUDE_DIRS for part in path.parts):
            if path.name == "CMakeLists.txt" or path.suffix == ".cmake":
                yield path.resolve()


def safe_read_text(file_path):
    """
    Safely read text from a file, handling various encodings and binary files.
    """
    # Skip known binary file extensions
    BINARY_EXTENSIONS = {
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".a",
        ".lib",
        ".obj",
        ".o",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".ico",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".7z",
        ".rar",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".wav",
    }

    if file_path.suffix.lower() in BINARY_EXTENSIONS:
        return ""

    # Try different encodings in order of likelihood
    encodings = ["utf-8", "latin-1", "cp1252", "ascii"]

    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception:
            # Handle other potential file reading errors
            break

    # If all encodings fail, try with error handling
    try:
        return file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def strip_at_lines(text):
    lines = text.split("\n")
    filtered_lines = [
        line
        for line in lines
        if not (line.strip().startswith("@") and line.strip().endswith("@"))
    ]
    return "\n".join(filtered_lines)


def parse_repo(dir: str) -> str:
    context = CMakeContext()
    for file_path in find_cmake_files(dir):
        try:
            content = safe_read_text(file_path)
            content = strip_at_lines(content)
            if not content:  # Skip empty or unreadable files
                continue
            # TODO do we want to capture comments seperately?
            nodes = parse_tree(content, skip_comments=True)
            extract_from_ast(nodes, context)
        except CMakeParseError as e:
            log.error(f"parse error in {file_path}: {e}")
    return context
