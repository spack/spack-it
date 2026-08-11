import argparse

import pandas as pd

from analysis.core import pivot_tbl, pre

parser = argparse.ArgumentParser(
    description="collapse the per-(model, reference, buildsys) pivot table into a "
    "single n-weighted row per model -- the summary used for tables/all.tex -- from "
    "one or more results jsonl files, without editing a hardcoded path in a script "
    "first"
)
parser.add_argument(
    "results", nargs="+", help="one or more results jsonl paths to summarize together"
)
parser.add_argument(
    "--attempt-max", type=int, default=None, help="passthrough to pivot_tbl"
)
parser.add_argument(
    "--to-latex", action="store_true", help="print a LaTeX tabular instead of text"
)
ARGS = parser.parse_args()

df = pd.concat([pre(path) for path in ARGS.results], ignore_index=True)
per_cell = pivot_tbl(df, attempt_max=ARGS.attempt_max)


def _weighted(g: pd.DataFrame) -> pd.Series:
    n = g["n"].sum()
    return pd.Series(
        {
            "install": (g["install"] * g["n"]).sum() / n,
            "dependency_score": (g["dependency_score"] * g["n"]).sum() / n,
            "variants_score": (g["variants_score"] * g["n"]).sum() / n,
            "n": n,
        }
    )


out = (
    per_cell.reset_index()
    .groupby("model")
    .apply(_weighted, include_groups=False)
    .sort_values("install", ascending=False)
    .round(3)
)

if ARGS.to_latex:
    print(out.to_latex(float_format="%.3f"))
else:
    print(out.to_string())
