# ruff: noqa: F401

import numpy as np
import pandas as pd

from analysis.core import rm_bad_rows

df_raw = pd.read_json("results/main_results/all.jsonl", lines=True)
df_raw = df_raw.loc[lambda d: d["status"].ne("workflow_fail")].assign(
    attempt_num=lambda d: d["attempt_num"] + 1,
)
# remove rows that were cut off in the middle of a run
df_raw = rm_bad_rows(df_raw)


# # plots the error recovery diffs between concretize and install...do we have enough data to have significance??
# print(plot_attempt_deltas(df_raw))


def print_attempt_deltas(df):
    """
    Compute and print summary statistics for concretize and install recovery deltas.
    Delta = first <success>.attempt_num - first <fail>.attempt_num preceding it.
    """
    d = df.copy()
    d["attempt_num"] = d["attempt_num"].astype(int)
    d = d.sort_values(["run_id", "pkg_name", "attempt_num"])

    def delta_first_fail(group, fail_label, success_label):
        succ = group.loc[group["status"] == success_label, "attempt_num"]
        if succ.empty:
            return None
        succ_attempt = int(succ.iloc[0])

        prior_fails = group.loc[
            (group["status"] == fail_label) & (group["attempt_num"] < succ_attempt),
            "attempt_num",
        ]
        if prior_fails.empty:
            return None

        first_fail = int(prior_fails.min())
        return succ_attempt - first_fail

    conc_deltas, inst_deltas = [], []
    for (_, _), grp in d.groupby(["run_id", "pkg_name"], sort=False):
        conc = delta_first_fail(grp, "concretize_fail", "concretize")
        inst = delta_first_fail(grp, "install_fail", "install")
        if conc is not None:
            conc_deltas.append(conc)
        if inst is not None:
            inst_deltas.append(inst)

    # Print results
    print("=== Recovery Attempt Deltas ===")
    if conc_deltas:
        print(
            f"Concretize: n={len(conc_deltas)}, mean={np.mean(conc_deltas):.2f}, "
            f"median={np.median(conc_deltas):.2f}, min={min(conc_deltas)}, max={max(conc_deltas)}"
        )
    else:
        print("Concretize: no fail→success transitions found.")

    if inst_deltas:
        print(
            f"Install:    n={len(inst_deltas)}, mean={np.mean(inst_deltas):.2f}, "
            f"median={np.median(inst_deltas):.2f}, min={min(inst_deltas)}, max={max(inst_deltas)}"
        )
    else:
        print("Install: no fail→success transitions found.")

    return conc_deltas, inst_deltas


print_attempt_deltas(df_raw)
