# ruff: noqa: F401


from analysis.core import (
    pre,
    summarize_status_pct,
)

# NOTE: our baseline is our own generation with our approach, not any human or ground truth..
df_baseline = pre("results/swe_agent/baseline.jsonl", spec=True)
df_swe_agent = pre("results/swe_agent/with_swe_agent.jsonl", spec=True)

summarize_status_pct(
    df_baseline, title="===max_attempts=1, buildsys=raw, reference=none==="
)
summarize_status_pct(df_swe_agent, title="===swe-agent, only 1 attempt===")
