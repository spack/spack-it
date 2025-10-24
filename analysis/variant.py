# ruff: noqa: F401


from analysis.core import (
    pre,
    summarize_status_pct,
)

# TO NOTE: the they all previously installed but the variants were implemented incorrectly..

df_baseline = pre("results/variant_sampling/baseline.jsonl", spec=True)
df_gen = pre("results/variant_sampling/llm_generated_variant_sampling.jsonl", spec=True)

summarize_status_pct(df_baseline, title="===variant sampling, baseline===")
summarize_status_pct(df_gen, title="===variant sampling, llm generated===")
