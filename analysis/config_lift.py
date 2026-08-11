import argparse

import pandas as pd

from analysis.core import pivot_tbl, pre

parser = argparse.ArgumentParser(
    description="model-normalized config lift: for each (reference, buildsys) "
    "config, computes install/variants_score/dependency_score minus that SAME "
    "model's own none/distilled baseline, then averages equally across models -- "
    "isolates 'what does this config add' from 'this model is just strong "
    "everywhere'. Feeds the 'Effect of reference type' paragraph and the "
    "Observation 5 config-comparison claim."
)
parser.add_argument("results", nargs="+", help="one or more results jsonl paths")
parser.add_argument(
    "--sort-by",
    choices=["install", "variants_score", "dependency_score"],
    default="install",
)
ARGS = parser.parse_args()

df = pd.concat([pre(path) for path in ARGS.results], ignore_index=True)
out = pivot_tbl(df).reset_index()

baseline = out[(out["reference"] == "none") & (out["buildsys"] == "distilled")].set_index(
    "model"
)

metrics = ["install", "variants_score", "dependency_score"]
for metric in metrics:
    out[f"{metric}_lift"] = out.apply(
        lambda r, m=metric: r[m] - baseline.loc[r["model"], m], axis=1
    )

lift_cols = [f"{m}_lift" for m in metrics]
summary = (
    out.groupby(["reference", "buildsys"])[lift_cols]
    .mean()
    .round(3)
    .sort_values(f"{ARGS.sort_by}_lift", ascending=False)
)
print(summary.to_string())
