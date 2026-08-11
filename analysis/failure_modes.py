import argparse
from collections import Counter, defaultdict

import pandas as pd

parser = argparse.ArgumentParser(
    description="per-model failure-mode breakdown: among all *_fail attempt rows "
    "(not just terminal outcomes), which stage (load_fail/concretize_fail/"
    "install_fail) dominates for each model. Feeds the 'Cross-model summary' "
    "paragraph's failure-mode claim (results.tex)."
)
parser.add_argument("results", nargs="+", help="one or more results jsonl paths")
ARGS = parser.parse_args()

df = pd.concat([pd.read_json(path, lines=True) for path in ARGS.results], ignore_index=True)
fails = df[df["status"].str.endswith("_fail", na=False)]

by_model: dict[str, Counter] = defaultdict(Counter)
for _, row in fails.iterrows():
    by_model[row["model"]][row["status"]] += 1

for model, counts in by_model.items():
    total = sum(counts.values())
    ranked = counts.most_common()
    breakdown = ", ".join(f"{status}={n} ({n / total:.0%})" for status, n in ranked)
    print(f"{model:30s} n_fail_rows={total:5d}  {breakdown}")
