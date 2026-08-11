import argparse
import json
from collections import Counter

parser = argparse.ArgumentParser(
    description="pass/fail/inconclusive breakdown for a --smoke_test results jsonl "
    "(see generate/container.py:BuilderContainer.smoke_test_pkg). A row counts as "
    "pass if smoke_kind='binary' with zero crashes, or 'headers_or_libs_only' "
    "(existence check only, nothing executed); fail if 'empty_prefix'/'no_prefix', "
    "or 'binary' with any crash; inconclusive if 'smoke_test_error' (the smoke "
    "test's own tooling failed -- not a verdict on the package)."
)
parser.add_argument("results", help="path to a results jsonl generated with --smoke_test")
ARGS = parser.parse_args()

rows = [json.loads(line) for line in open(ARGS.results) if line.strip()]
installs = [r for r in rows if r["status"] == "install"]

print("smoke_kind breakdown:", dict(Counter(r.get("smoke_kind") for r in installs)))

passed, failed, inconclusive = [], [], []
for r in installs:
    kind = r.get("smoke_kind")
    if kind == "binary":
        (passed if r.get("smoke_crashed", 0) == 0 else failed).append(r["pkg_name"])
    elif kind == "headers_or_libs_only":
        passed.append(r["pkg_name"])
    elif kind in ("empty_prefix", "no_prefix"):
        failed.append(r["pkg_name"])
    elif kind == "smoke_test_error":
        inconclusive.append(r["pkg_name"])

n = len(installs)
print(f"pass: {len(passed)}/{n} = {len(passed) / n:.1%}" if n else "pass: 0/0")
print(f"fail: {len(failed)}/{n}", failed)
print(f"inconclusive: {len(inconclusive)}/{n}", inconclusive)
