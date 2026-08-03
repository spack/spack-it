import argparse

import pandas as pd

from analysis.core import pivot_tbl, pre

parser = argparse.ArgumentParser(
    description="summarize one or more results jsonl files into the per-(model, "
    "reference, buildsys) table used for tables/ablation.tex, without editing a "
    "hardcoded path in a script first"
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
out = pivot_tbl(df, attempt_max=ARGS.attempt_max)

if ARGS.to_latex:
    print(out.to_latex(float_format="%.3f"))
else:
    print(out.to_string())
