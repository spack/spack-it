# ruff: noqa: F401

import pandas as pd

from analysis.core import (
    categorize_buildsys,
    categorize_model,
    categorize_reference,
    rm_bad_rows,
)

df_raw = pd.read_json("results/main_results/all.jsonl", lines=True)
df_raw = df_raw.loc[lambda d: d["status"].ne("workflow_fail")]
# remoove rows that were cut off in the middle of a run
df_raw = rm_bad_rows(df_raw)

df_raw = df_raw.assign(
    reference=lambda d: d.apply(categorize_reference, axis=1),
    buildsys=lambda d: d.apply(categorize_buildsys, axis=1),
    model=lambda d: d.apply(categorize_model, axis=1),
)

import pandas as pd


def pivot_error_dist(
    df: pd.DataFrame,
    attempt_max: int | None = None,
    only_fail: bool = True,
    top_k: int | None = None,
    other_label: str = "Other",
    include_n: bool = True,
    round_to: int = 3,
):
    """
    Build a per-group distribution of `classified_error`.

    Groups by (model, reference, buildsys).
    - attempt_max: keep rows with attempt_num <= attempt_max (if provided)
    - only_fail: if True, consider only rows where status contains 'fail'
    - top_k: if set, keep top-K error columns overall and collapse the rest into `Other`
    - include_n: include a count column 'n'
    - round_to: number of decimals to round fractions to
    """
    if attempt_max is not None and "attempt_num" in df.columns:
        df = df[df["attempt_num"] <= attempt_max]

    # Optionally restrict to failed rows
    if only_fail and "status" in df.columns:
        fail_mask = df["status"].astype(str).str.contains("fail", case=False, na=False)
        df = df[fail_mask]

    # Grouping columns (no audit)
    group_cols = ["model", "reference", "buildsys"]

    # Ensure the column exists; replace missing with a bucket
    err_col = "classified_error"
    if err_col not in df.columns:
        raise KeyError(f"Expected column `{err_col}` not found.")
    df = df.copy()
    df[err_col] = df[err_col].fillna("(none)")

    # Counts per error per group -> fractions (row-normalized)
    counts = df.groupby(group_cols)[err_col].value_counts().unstack(fill_value=0)

    # Optionally reduce to top-K errors overall
    if top_k is not None and top_k > 0 and counts.shape[1] > top_k:
        # Find top-K by total count across all groups
        totals = counts.sum(axis=0).sort_values(ascending=False)
        keep = list(totals.head(top_k).index)
        drop = [c for c in counts.columns if c not in keep]
        if drop:
            counts[other_label] = counts[drop].sum(axis=1)
            counts = counts[keep + [other_label]]

    # Convert to fractions per group
    row_sums = counts.sum(axis=1).replace(0, 1)  # avoid div-by-zero
    fracs = counts.div(row_sums, axis=0)

    # Optional count column
    if include_n:
        fracs.insert(0, "n", counts.sum(axis=1))

    # Round fractions
    if round_to is not None:
        for c in fracs.columns:
            if c != "n":
                fracs[c] = fracs[c].round(round_to)

    # Sort columns: n first, then errors by overall frequency
    err_cols = [c for c in fracs.columns if c != "n"]
    order = err_cols
    if len(err_cols) > 1:
        overall = (
            counts[err_cols].sum(axis=0).sort_values(ascending=False).index.tolist()
        )
        order = overall
    if include_n:
        fracs = fracs[["n"] + order]
    else:
        fracs = fracs[order]

    return fracs


def error_percentages_by(
    df: pd.DataFrame,
    by: str = "model",  # "model" or "reference"
    attempt_max: int | None = None,
    only_fail: bool = True,
    top_k: int | None = None,
    other_label: str = "Other",
    round_to: int = 1,  # decimals in the percentage values
    include_n: bool = True,  # include a total 'n' column
    order_by_overall_freq: bool = True,  # order error columns by total frequency desc
) -> pd.DataFrame:
    """
    Aggregate error distributions from pivot_error_dist to percentages by a single dimension.

    Uses counts reconstructed from per-group fractions * n, then sums across groups that
    share the requested 'by' value, and finally normalizes to percentages per row.
    """
    tbl = pivot_error_dist(
        df,
        attempt_max=attempt_max,
        only_fail=only_fail,
        top_k=top_k,
        other_label=other_label,
        include_n=True,  # required for correct aggregation
        round_to=3,
    )

    if not isinstance(tbl.index, pd.MultiIndex):
        raise ValueError(
            "Expected pivot_error_dist to return a MultiIndex on (model, reference, buildsys)."
        )

    # map which index level to aggregate by
    index_names = list(tbl.index.names)
    expected = ["model", "reference", "buildsys"]
    if index_names != expected:
        # be lenient if names differ but positions are the same
        pass

    level_map = {"model": 0, "reference": 1}
    if by not in level_map:
        raise ValueError("`by` must be 'model' or 'reference'.")

    lvl = level_map[by]

    # Reconstruct counts from fractions * n for each error column
    error_cols = [c for c in tbl.columns if c != "n"]
    counts = tbl[error_cols].multiply(tbl["n"], axis=0)
    counts["n"] = tbl["n"]

    # Aggregate counts over the requested level (model or reference)
    agg = counts.groupby(level=lvl).sum()

    # Convert to percentages
    err_cols_only = [c for c in agg.columns if c != "n"]
    pct = (agg[err_cols_only].div(agg["n"], axis=0) * 100).round(round_to)

    # Optional: order error columns by overall frequency (descending)
    if order_by_overall_freq and err_cols_only:
        overall = (
            agg[err_cols_only].sum(axis=0).sort_values(ascending=False).index.tolist()
        )
        pct = pct[overall]

    # Insert total n if desired
    if include_n:
        pct.insert(0, "n", agg["n"])

    return pct


print(pivot_error_dist(df_raw))


# 1) Percentages by model (default), include total n, keep top 8 error types overall
pct_by_model = error_percentages_by(
    df_raw,
    by="reference",
    attempt_max=None,
    only_fail=True,
    # top_k=8,
    round_to=1,
    include_n=True,
    order_by_overall_freq=True,
)

print(pct_by_model)
