# ruff: noqa: F401

import matplotlib.pyplot as plt

from analysis.core import (
    build_funnel_from_raw,
    plot_funnel_horizontal_flipped,
    plot_stage_survival_incremental_bars,
    pre,
)

df_all = pre("results/main_results/all.jsonl")

# TODO check out
# plot_first_success_distribution_bars
# build_stage_survival_table_sets
# plot_stage_survival_incremental_bars


# these current plots aren't great but they're a start at showing survival...
fig, ax = plot_stage_survival_incremental_bars(
    df_all,
    model="gpt-4.1",
    title="Incremental Attempt Contributions — GPT-4.1",
)
plt.show()


df_funnel = build_funnel_from_raw(
    df_all,
    group_cols=("model", "reference", "buildsys"),
    status_col="status",
)

# 2) Plot
plot_funnel_horizontal_flipped(
    df_funnel,
    panel_rows="reference",
    panel_cols="buildsys",
    y="model",
)
