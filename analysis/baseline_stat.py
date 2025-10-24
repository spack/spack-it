# ruff: noqa: F401

import pandas as pd

from analysis.core import (
    rm_bad_rows,
)

df_raw = pd.read_json("results/main_results/all.jsonl", lines=True)
df_raw = df_raw.loc[lambda d: d["status"].ne("workflow_fail")].assign(
    attempt_num=lambda d: d["attempt_num"] + 1,
)
# remove rows that were cut off in the middle of a run
df_raw = rm_bad_rows(df_raw)
df_raw["installed"] = (df_raw["status"] == "install").astype(int)
df_last_attempt1 = (
    df_raw[
        (df_raw["attempt_num"] == 1)  # & (df_raw["model"] == "gpt-5")
        & (df_raw["no_recipe"] == True)
        & (df_raw["raw_buildsys"] == True)
    ]
    .sort_values(["run_id", "pkg_name", "timestamp"])  # sort chronologically
    .groupby(["run_id", "pkg_name"], dropna=False, group_keys=False)
    .tail(1)  # last row in each group
    .reset_index(drop=True)
)
print(df_last_attempt1["installed"].mean())
