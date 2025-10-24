import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.stats import linregress


def _apply_style():
    plt.rcParams.update(
        {
            # "figure.dpi": 300,
            # "figure.autolayout": True,
            "axes.grid": True,
            "grid.alpha": 0.25,
            # "axes.spines.top": False,
            # "axes.spines.right": False,
            # "axes.titlesize": 12,
            # "axes.labelsize": 10,
            # "xtick.labelsize": 9,
            # "ytick.labelsize": 9,
            # "legend.fontsize": 9,
        }
    )


SUCCESS_STAGE = "install"

_FINAL_STAGE_MAP = {
    # successes
    "load": "Load",
    "concretize": "Concretize",
    "install": "Install",
    "test": "Test",
    # fails → previous stage
    "load_fail": "no_load",
    "concretize_fail": "Load",
    "install_fail": "Concretize",
    "test_fail": "Install",
}

_PLOT_COLS = ["Test", "Install", "Concretize", "Load", "no_load"]

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: f"{x:.6f}")


# PREPROCESSING


def pre(filename, spec=False):
    """spec=True means to group on spec in addition to run_id and pkg_name. only relevant to the variant sampling"""
    df = pd.read_json(filename, lines=True)
    df = df.loc[lambda d: d["status"].ne("workflow_fail")].assign(
        attempt_num=lambda d: d["attempt_num"] + 1,
    )
    df = rm_bad_rows(df)
    df = process_raw(df, spec=True)

    mapping = {
        "similar pkg (1)": "Similar",
        "random pkg (1) ": "Random",
        "similar pkg (2)": "2 Similar",
        "random cmake pkg (1)": "Random CMake",
        "random cmake pkg (2)": "2 Random CMake",
        "none": "None",
    }

    order = [
        "Similar",
        "2 Similar",
        "Random",
        "Random CMake",
        "2 Random CMake",
        "None",
    ]

    df["ref"] = df["reference"].map(mapping)

    # make it an ordered categorical (Seaborn respects this order)
    df["ref"] = pd.Categorical(df["ref"], categories=order, ordered=True)

    return df


def categorize_reference(row):
    similar = row.get("similar_recipe")
    random_buildsys = row.get("random_buildsys_recipe")
    random_pkg = row.get("random_recipe")

    ref = " "

    if similar:
        ref += f"similar pkg ({similar}) "
    if random_buildsys:
        ref += f"random cmake pkg ({random_buildsys}) "
    if random_pkg:
        return f"random pkg ({random_pkg}) "

    ref_strip = ref.strip()

    if ref_strip:
        return ref_strip

    return "none"


def categorize_buildsys(row):
    if row.get("raw_buildsys", False):
        return "raw"
    if row.get("distilled_cmake", False):
        return "distilled"
    return "none"


def categorize_model(row):
    if row.get("model") == "mistralai/Mistral-Large-Instruct-2411":
        return "mistral-large-2411"
    return row.get("model")


def pick_run(group):
    """given a group, find the successful row or the last row"""
    success_stage = group[group["status"] == SUCCESS_STAGE]
    # if we've had success, the length of success_stage is 1 so we just grab the top
    if not success_stage.empty:
        return success_stage.head(1)
    else:
        # otherwise we get get the last attempt..
        return group.tail(1)


def preprocess(df: pd.DataFrame, spec=False) -> pd.DataFrame:
    def _apply_with_keys(g: pd.DataFrame) -> pd.DataFrame:
        out = pick_run(g)
        if spec:
            run_id, pkg_name, s = g.name
            return out.assign(run_id=run_id, pkg_name=pkg_name, spec=s)
        else:
            run_id, pkg_name = g.name
            return out.assign(run_id=run_id, pkg_name=pkg_name)

    groupby = ["run_id", "pkg_name"]
    if spec:
        groupby.append("spec")

    return (
        df.groupby(groupby, dropna=False, group_keys=False)
        .apply(_apply_with_keys, include_groups=False)
        .reset_index(drop=True)
        .copy()
    )


def rm_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """remove rows from the raw df that haven't reached the sucess stage under max_attempts...
    this usually means that the run was cut off in the middle of execution"""
    processed = preprocess(df)
    keys = processed.loc[
        (processed["status"] != SUCCESS_STAGE)
        & (processed["attempt_num"] != processed["max_attempts"]),
        ["run_id", "pkg_name"],
    ]

    removed = df.merge(keys, on=["run_id", "pkg_name"], how="inner")
    kept = (
        df.merge(keys, on=["run_id", "pkg_name"], how="left", indicator=True)
        .loc[lambda d: d["_merge"].eq("left_only")]
        .drop(columns="_merge")
    )

    # print(f"Will remove {len(removed)} rows")
    # print(removed)
    return kept


def process_raw(df: pd.DataFrame, spec=False) -> pd.DataFrame:
    """normalize some rows"""
    load_set = {
        "load",
        "concretize",
        "concretize_fail",
        "install",
        "install_fail",
        "test",
        "test_fail",
    }
    concretize_set = {"concretize", "install", "install_fail", "test", "test_fail"}
    install_set = {"install", "test", "test_fail"}

    return (
        preprocess(df, spec)
        .assign(
            reference=lambda d: d.apply(categorize_reference, axis=1),
            buildsys=lambda d: d.apply(categorize_buildsys, axis=1),
            model=lambda d: d.apply(categorize_model, axis=1),
            load=lambda d: d["status"].isin(load_set),
            concretize=lambda d: d["status"].isin(concretize_set),
            install=lambda d: d["status"].isin(install_set),
        )
        .reset_index(drop=True)
        .copy()
    )


def ground_success(df: pd.DataFrame) -> float:
    return df["status"].eq(SUCCESS_STAGE).mean()


df_ground = pd.read_json("results/ground.jsonl", lines=True)
df_ground = preprocess(df_ground)
ground_baseline = ground_success(df_ground)


# TABLES AND OTHER SUMMARY STATS


def coverage_matrix_vs_anchor(df, anchor_model="gpt-4.1"):
    group_cols = ["reference", "buildsys"]

    # anchor configs: all (reference, buildsys) combos seen for the anchor model
    anchor_configs = (
        df[df["model"] == anchor_model][group_cols]
        .drop_duplicates()
        .sort_values(group_cols)
    )

    # counts per (model, reference, buildsys)
    counts = df.groupby(["model"] + group_cols).size().rename("n").reset_index()

    # cartesian product of all models × anchor configs
    all_models = pd.DataFrame({"model": sorted(df["model"].unique())})
    grid = all_models.merge(anchor_configs, how="cross")

    # fill missing (model, reference, buildsys) with 0
    coverage = grid.merge(counts, on=["model"] + group_cols, how="left").assign(
        n=lambda x: x["n"].fillna(0).astype(int)
    )

    # pivot to get wide matrix (int values, not float)
    matrix = (
        coverage.pivot(index=["reference", "buildsys"], columns="model", values="n")
        .fillna(0)
        .astype(int)
        .sort_index()
    )

    return matrix


def summarize_status_pct(df, title="===status summary==="):
    load_set = {
        "load",
        "concretize",
        "concretize_fail",
        "install",
        "install_fail",
        "test",
        "test_fail",
    }
    concretize_set = {"concretize", "install", "install_fail", "test", "test_fail"}
    install_set = {"install", "test", "test_fail"}

    # Base percentages
    summary = df.assign(
        load=df["status"].isin(load_set),
        concretize=df["status"].isin(concretize_set),
        install=df["status"].isin(install_set),
    )[["load", "concretize", "install"]].mean()

    # Add dependency and variant scores if present (in the middle)
    if "dependency_score" in df.columns:
        summary["dependency_score"] = df["dependency_score"].dropna().mean()
    if "variants_score" in df.columns:
        summary["variants_score"] = df["variants_score"].dropna().mean()

    # Add num_tokens at the bottom
    if "num_tokens" in df.columns:
        summary["num_tokens"] = df["num_tokens"].mean()

    print(title)

    # Print model first
    if "model" in df.columns:
        models = df["model"].dropna().unique()
        if len(models) == 1:
            print(f"model: {models[0]}")
        else:
            print(f"models: {models}")

    # Print summary neatly
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))


def pivot_tbl(df, attempt_max: int = None):
    """
    Build a summary table. If `audit` column exists, include it as another
    grouping dimension. If `split_audit=True`, present audit as columns.
    """
    if attempt_max is not None:
        df = df[df["attempt_num"] <= attempt_max]

    base_group = ["model", "reference", "buildsys"]
    has_audit = "audit" in df.columns
    group_cols = base_group + (["audit"] if has_audit else [])

    # Base pivot: means of your metrics (keeps MultiIndex rows)
    pivot = pd.pivot_table(
        df,
        values=[
            "variants_score",
            "dependency_score",
            # "variants_extras",
            "num_tokens",
        ],
        index=group_cols,
        aggfunc="mean",
    )

    counts = df.groupby(group_cols).size().rename("n")

    # Status sets (inclusive progression)
    load_set = {
        "load",
        "concretize",
        "concretize_fail",
        "install",
        "install_fail",
        "test",
        "test_fail",
    }
    concretize_set = {"concretize", "install", "install_fail", "test", "test_fail"}
    install_set = {"install", "test", "test_fail"}

    # Per-group 0/1 flags averaged to get fractions
    status_pct = (
        df.assign(
            load=df["status"].isin(load_set),
            concretize=df["status"].isin(concretize_set),
            install=df["status"].isin(install_set),
        )
        .groupby(group_cols)[["load", "concretize", "install"]]
        .mean()
    )

    attempts_to_success = (
        df[df["status"].isin(install_set)]
        .groupby(group_cols)["attempt_num"]
        .mean()
        .rename("attempts_to_success")
    )

    # Combine on the same MultiIndex
    out = pivot.join([counts, status_pct, attempts_to_success])

    # Round certain columns
    for c in [
        "load",
        "concretize",
        "install",
        "variants_score",
        "dependency_score",
        # "variants_extras",
        "num_tokens",
        "attempts_to_success",
    ]:
        if c in out.columns:
            out[c] = out[c].round(3)

    # Order columns
    col_order = [
        "n",
        "load",
        "concretize",
        "install",
        "variants_score",
        "dependency_score",
        # "variants_extras",
        "num_tokens",
        "attempts_to_success",
    ]
    out = out[[c for c in col_order if c in out.columns]]

    return out


# PLOTTING CODE


def plot_cumulative_success(
    df_runs: pd.DataFrame,
    by: str = "model",
    title: str = "Cumulative Success Rate by Number of Attempts",
    add_delta: bool = True,
):
    _apply_style()

    # find the unique values of rach group
    groups = pd.Index(df_runs[by].unique())

    fig, axes = plt.subplots(
        1, max(1, len(groups)), figsize=(6 * max(1, len(groups)), 5), sharey=True
    )
    if len(groups) == 1:
        axes = [axes]

    for ax, g in zip(axes, groups):
        # find the runs that are part of the group
        grp = df_runs.loc[df_runs[by] == g].copy()

        att_all = grp["attempt_num"].dropna()
        if att_all.empty:
            raise ValueError(f"{by} group {g} is empty")

        # final attempt number for each run
        att_all = att_all.round().astype(int)
        # define the range of the plot
        min_k, max_k = att_all.min(), att_all.max()
        bins = np.arange(min_k, max_k + 1)

        denom = len(att_all)

        # for each row, get the attempt number where it reached the success stage
        # if it didn't reach the stage, it won't be present in this np array
        success_att = grp.loc[grp["status"] == SUCCESS_STAGE, "attempt_num"].dropna()
        success_att = success_att.round().astype(int)
        # count how many successes are in each bin
        success_counts = (
            success_att.value_counts().reindex(bins, fill_value=0).sort_index()
        )
        # calculate cumulative successes within the attempts
        cum_success = success_counts.cumsum()

        # success rate over the runs in the group
        rate = (cum_success / denom).astype(float)

        # plot success rates
        ax.bar(bins, rate.values)
        ax.set_xlabel("Generation Attempts")
        ax.set_ylabel("Cumulative Installation Success Rate")
        ax.set_xticks(bins.tolist())
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

        # annotate deltas
        if add_delta:
            deltas = rate.diff().fillna(rate.iloc[0])
            for x, y, d in zip(bins, rate.values, deltas.values):
                ax.text(
                    x,
                    y + 0.02,
                    f"+{100 * d:.3f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # overall success rate within the group
        overall_rate = float(rate.iloc[-1]) if len(rate) else 0.0
        n_success = int((grp["status"] == SUCCESS_STAGE).sum())
        ax.set_title(
            f"{by}: {g}\n{n_success}/{denom} {SUCCESS_STAGE}ed ({overall_rate:.0%})",
            fontsize=12,
        )

    fig.suptitle(title, fontsize=14)
    return fig, axes


def plot_status_hist(df):
    # Keep statuses that end with 'fail' but are not exactly 'test_fail'
    filtered = df[df["status"].str.endswith("fail") & (df["status"] != "test_fail")]

    # Plot histogram
    filtered["status"].value_counts().plot(kind="bar")

    # Add labels and title
    plt.xlabel("Status")
    plt.ylabel("Frequency")
    plt.title("Final Status Distribution")
    plt.xticks(rotation=0)
    plt.show()


def plot_cumulative_success_lines(
    df_runs: pd.DataFrame,
    by: str = "model",
    title: str = "Cumulative Success Rate by Number of Attempts",
    baseline: float | None = None,
    facet_by: str | None = None,
    ncols: int = 2,
    width: float = 7,
    height: float = 5,
    sharey: bool = True,
    df_runs_b: pd.DataFrame | None = None,  # optional second dataset
    labels: tuple[str, str] = ("A", "B"),  # legend suffixes for A/B
    alt_line_kw: dict | None = None,  # style overrides for B
    filter_expr: str | None = None,  # optional filter for df_runs
    legend_map: dict[str, str] | None = None,  # map group values (by) -> legend labels
    legend_title: str | None = None,  # legend title describing what legend encodes
    subtitle: str | None = None,  # figure-level subtitle
    # NEW
    label_overrides: tuple[str, str] | None = None,  # completely replaces legend labels
):
    """
    Plot cumulative success rate vs. attempt count. Optionally overlay a second
    dataset (df_runs_b) with tinted colors and alternate line style.

    Parameters
    ----------
    ...
    legend_map : dict, optional
        Mapping from raw group values in `by` to custom legend labels.
    legend_title : str, optional
        Title for the legend (e.g., "Model", "Variant", "Decoder").
    subtitle : str, optional
        Figure-level subtitle placed under the main title.
    label_overrides : tuple of str, optional
        If provided and `df_runs_b` is not None, this completely replaces the
        legend text for dataset A and B. Each dataset uses its respective entry
        (`label_overrides[0]` for A, `label_overrides[1]` for B), and no suffixes
        or mappings are applied.
    """
    _apply_style()

    # ---------- helpers ----------
    from matplotlib.colors import to_hex, to_rgb

    def _tint(color, amount=0.35):
        """Lighten a color by mixing with white (amount in [0,1])."""
        r, g, b = to_rgb(color)
        r = 1 - (1 - r) * (1 - amount)
        g = 1 - (1 - g) * (1 - amount)
        b = 1 - (1 - b) * (1 - amount)
        return to_hex((r, g, b))

    # ---------- filtering ----------
    if filter_expr is not None:
        if isinstance(filter_expr, str):
            df_runs = df_runs.query(filter_expr)
        elif callable(filter_expr):
            df_runs = filter_expr(df_runs)
        else:
            raise ValueError("filter_expr must be a string or callable")

    # ---------- facet setup ----------
    if facet_by is None:
        facets = [None]
    else:
        facets = (
            df_runs[facet_by].dropna().unique().tolist()
            if facet_by in df_runs.columns
            else [None]
        )

    n = len(facets)
    ncols_local = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols_local)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols_local,
        figsize=(width * ncols_local, height * nrows),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )
    axes = np.ravel(axes)

    # ---------- global bins ----------
    frames = [df_runs]
    if df_runs_b is not None:
        frames.append(df_runs_b)

    att_all = pd.concat([f["attempt_num"].dropna() for f in frames], ignore_index=True)
    if att_all.empty:
        return fig, axes[:n]
    att_all = att_all.round().astype(int)
    global_min_k, global_max_k = att_all.min(), att_all.max()
    global_bins = np.arange(global_min_k, global_max_k + 1)

    # ---------- color maps ----------
    all_groups = pd.unique(
        pd.concat(
            [df_runs[by].dropna()]
            + ([df_runs_b[by].dropna()] if df_runs_b is not None else [])
        )
    ).tolist()

    cmap = plt.get_cmap("tab10")
    color_map = {g: cmap(i % cmap.N) for i, g in enumerate(sorted(all_groups))}
    color_map_b = {g: _tint(color_map[g], 0.35) for g in color_map}

    if alt_line_kw is None:
        alt_line_kw = {"linestyle": "--", "alpha": 0.9}

    # ---------- plotting ----------
    first_handles, first_labels = [], []

    for i, facet_val in enumerate(facets):
        ax = axes[i]
        if facet_val is None:
            dfA = df_runs
            dfB = df_runs_b
            facet_title = None
        else:
            dfA = df_runs.loc[df_runs[facet_by] == facet_val]
            dfB = (
                None
                if df_runs_b is None
                else df_runs_b.loc[df_runs_b[facet_by] == facet_val]
            )
            facet_title = f"{facet_by} = {facet_val}"

        if (dfA is None or dfA.empty) and (dfB is None or dfB.empty):
            ax.set_visible(False)
            continue

        series_A = (
            dfA[by].dropna()
            if dfA is not None and not dfA.empty
            else pd.Series([], dtype=object)
        )
        series_B = (
            dfB[by].dropna()
            if dfB is not None and not dfB.empty
            else pd.Series([], dtype=object)
        )
        groups = pd.Index(pd.unique(pd.concat([series_A, series_B])))

        if groups.empty:
            ax.set_visible(False)
            continue

        # ---------- legend label helper ----------
        def _legend_label(base_value: str, dataset_idx: int) -> str:
            # completely replace labels if requested
            if label_overrides is not None:
                return label_overrides[dataset_idx]

            # otherwise use mapping + suffixes
            mapped = (
                legend_map.get(base_value, base_value) if legend_map else base_value
            )
            suffix = f" ({labels[dataset_idx]})" if df_runs_b is not None else ""
            return f"{mapped}{suffix}"

        def _plot_one(df, dataset_idx: int):
            if df is None or df.empty:
                return
            for g in groups:
                grp = df.loc[df[by] == g]
                att = grp["attempt_num"].dropna()
                if att.empty:
                    continue
                att = att.round().astype(int)
                denom = len(att)

                success_att = grp.loc[
                    grp["status"] == SUCCESS_STAGE, "attempt_num"
                ].dropna()
                success_att = success_att.round().astype(int)
                success_counts = (
                    success_att.value_counts()
                    .reindex(global_bins, fill_value=0)
                    .sort_index()
                )
                rate = (success_counts.cumsum() / denom).astype(float)

                label_text = _legend_label(g, dataset_idx)

                if dataset_idx == 0:  # A
                    ax.plot(
                        global_bins,
                        rate.values,
                        marker="o",
                        color=color_map.get(g, "gray"),
                        label=label_text,
                    )
                else:  # B
                    ax.plot(
                        global_bins,
                        rate.values,
                        marker="o",
                        color=color_map_b.get(g, "gray"),
                        label=label_text,
                        **alt_line_kw,
                    )

        _plot_one(dfA, 0)
        _plot_one(dfB, 1)
        # breakpoint()

        # axis formatting
        ax.set_xticks(np.arange(global_min_k, global_max_k + 1))
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        if facet_title:
            ax.set_title(facet_title)

        if baseline is not None:
            ax.hlines(
                y=baseline,
                xmin=global_min_k,
                xmax=global_max_k,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label="Human-Written Baseline",
            )

        if not first_handles:
            first_handles, first_labels = ax.get_legend_handles_labels()

    # ---------- outer labels ----------
    for r in range(nrows):
        for c in range(ncols_local):
            idx = r * ncols_local + c
            if idx >= len(axes) or not axes[idx].get_visible():
                continue
            ax = axes[idx]
            if r == nrows - 1:
                ax.set_xlabel("Generation Attempts")
            if c == 0:
                ax.set_ylabel("Cumulative Installation Success Rate")

    # ---------- titles ----------
    if title:
        fig.suptitle(title, y=0.99)
    if subtitle:
        fig.text(0.5, 0.965, subtitle, ha="center", va="top", fontsize="medium")

    # ---------- shared legend ----------
    if first_handles:
        seen, handles, labels_out = set(), [], []
        for h, lab in zip(first_handles, first_labels):
            if lab not in seen and lab != "":
                seen.add(lab)
                handles.append(h)
                labels_out.append(lab)

        fig.legend(
            handles,
            labels_out,
            loc="lower center",
            ncol=min(len(labels_out), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 0.06),
            title=legend_title,
            title_fontsize="medium",
        )
        fig.set_size_inches(
            fig.get_size_inches()[0],
            fig.get_size_inches()[1] + 0.5,
            forward=True,
        )

    plt.tight_layout(rect=[0, 0.12, 1, 0.965 if subtitle else 0.97])
    fig.subplots_adjust(bottom=0.25)

    return fig, axes[:n]


def plot_first_success_distribution_bars(
    df_runs: pd.DataFrame,
    by: str = "model",
    title: str = "Distribution of First Successful Attempt",
):
    """
    Plot % of runs that first succeed on each attempt (1–5) as bars.
    Each `by` group (e.g. model) gets its own color series.

    X-axis: attempt number (1–5)
    Y-axis: percentage of runs
    """
    _apply_style()

    groups = pd.Index(df_runs[by].unique())
    fig, ax = plt.subplots(figsize=(7, 5))

    bins = np.arange(1, 6)  # attempt numbers 1–5
    width = 0.8 / len(groups)  # spacing for grouped bars

    for i, g in enumerate(groups):
        grp = df_runs.loc[df_runs[by] == g].copy()
        if grp.empty:
            continue

        # Denominator = number of runs in this group
        if "run_id" in grp.columns:
            denom = int(grp["run_id"].nunique())
        else:
            att_all = grp["attempt_num"].dropna()
            if att_all.empty:
                continue
            denom = len(att_all)
        if denom == 0:
            continue

        # Determine the first-success attempt per run
        if "run_id" in grp.columns:
            success_first = (
                grp.loc[grp["status"] == SUCCESS_STAGE, ["run_id", "attempt_num"]]
                .dropna()
                .assign(attempt_num=lambda d: d["attempt_num"].round().astype(int))
                .groupby("run_id", as_index=False)["attempt_num"]
                .min()["attempt_num"]
            )
        else:
            success_first = (
                grp.loc[grp["status"] == SUCCESS_STAGE, "attempt_num"]
                .dropna()
                .round()
                .astype(int)
            )

        # Count first successes per attempt (1–5)
        new_success_counts = (
            pd.Series(success_first)
            .value_counts()
            .reindex(bins, fill_value=0)
            .sort_index()
        )
        new_success_rate = (new_success_counts / denom).astype(float)

        # Bar position offset per group
        x_positions = bins + (i - len(groups) / 2) * width + width / 2
        ax.bar(
            x_positions, new_success_rate.values, width=width, label=str(g), alpha=0.8
        )

    # Axes & style
    ax.set_xlabel("Attempts")
    ax.set_ylabel("Percentage of Runs")
    ax.set_xticks(bins)
    ax.set_xlim(bins.min() - 0.5, bins.max() + 0.5)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(title=None)
    ax.set_title(title)

    return fig, ax


def build_funnel_from_raw(
    df: pd.DataFrame,
    group_cols=("model", "reference", "buildsys"),
    status_col="status",
) -> pd.DataFrame:
    """
    Build final-stage distribution per (model, context, buildsys).

    Rules:
      - If status has no '_fail' → final stage is that status (Load/Concretize/Install/Test).
      - If status ends with '_fail' → final stage is the previous success stage.
      - 'load_fail' → 'no_load' (never reached Load).

    Returns a wide table with columns:
      ['Test', 'Install', 'Concretize', 'Load', 'no_load']
    where each row's values sum to 1.
    """
    df = df.copy()

    # Normalize to final stage bucket; unknown statuses → 'no_load' (conservative)
    df["_final_stage"] = df[status_col].map(_FINAL_STAGE_MAP).fillna("no_load")

    # Count by final stage within each feature group
    grp = df.groupby(list(group_cols), dropna=False, sort=True)["_final_stage"]
    counts = grp.value_counts().unstack(fill_value=0)

    # Ensure all plot columns exist
    for col in _PLOT_COLS:
        if col not in counts.columns:
            counts[col] = 0

    # Convert to fractions per group
    denom = counts.sum(axis=1).replace(0, 1)  # guard against divide-by-zero
    out = (counts[_PLOT_COLS].div(denom, axis=0)).reset_index()

    return out


def plot_funnel_horizontal_flipped(
    df_funnel: pd.DataFrame,
    panel_rows="reference",
    panel_cols="buildsys",
    y="model",
    # flipped order: smallest (Test) at left → then Install → Concretize → Load → no_load tail
    stage_order=("Test", "Install", "Concretize", "Load"),
    bar_height=0.3,
    fig_width_per_col=4.0,
    fig_height_per_row=0.5,
):
    """
    Horizontal stacked bars (compact) with flipped order:
      left → right: Test, Install, Concretize, Load, (then a muted 'no_load' tail).
    Values in df_funnel should be *non-overlapping bands* that sum to 1 with 'no_load'.
    """
    # vibrant colors for stages
    colors = {
        "Test": "#7000c7",  # purple
        "Install": "#54a24b",  # green
        "Concretize": "#f58518",  # orange
        "Load": "#4c78a8",  # blue
        "no_load": "#d9d9d9",  # light gray (de-emphasized, not in legend)
    }

    # panels
    r_vals = (
        sorted(df_funnel[panel_rows].dropna().unique())
        if panel_rows in df_funnel
        else [None]
    )
    c_vals = (
        sorted(df_funnel[panel_cols].dropna().unique())
        if panel_cols in df_funnel
        else [None]
    )
    if not r_vals:
        r_vals = [None]
    if not c_vals:
        c_vals = [None]

    # compact figure sizing
    fig_w = max(6.0, fig_width_per_col * len(c_vals))
    fig_h = max(2.0, fig_height_per_row * len(r_vals))
    fig, axes = plt.subplots(
        len(r_vals),
        len(c_vals),
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )

    for i, r in enumerate(r_vals):
        for j, c in enumerate(c_vals):
            ax = axes[i, j]
            sub = df_funnel.copy()
            if r is not None:
                sub = sub[sub[panel_rows] == r]
            if c is not None:
                sub = sub[sub[panel_cols] == c]
            if sub.empty:
                ax.axis("off")
                continue

            # consistent order on y
            y_vals = list(sorted(sub[y].astype(str).unique()))
            sub = sub.set_index(y).reindex(y_vals)

            left = np.zeros(len(y_vals))

            # stack core stages in flipped order
            handles = []
            for stage in stage_order:
                if stage not in sub.columns:
                    vals = np.zeros(len(y_vals))
                else:
                    vals = sub[stage].fillna(0.0).values
                h = ax.barh(
                    y_vals,
                    vals,
                    left=left,
                    height=bar_height,
                    color=colors.get(stage, "#999999"),
                    edgecolor="white",
                    label=stage,
                )
                handles.append(h)

                # annotate if wide enough
                for yi, v, l in zip(y_vals, vals, left):
                    if v >= 0.06:
                        ax.text(
                            l + v / 2,
                            yi,
                            f"{v:.2f}",
                            va="center",
                            ha="center",
                            color="white",
                            fontsize=8,
                        )
                left += vals

            # append muted no_load to the far right (not in legend)
            tail = (
                sub["no_load"].fillna(0.0).values
                if "no_load" in sub.columns
                else np.zeros(len(y_vals))
            )
            ax.barh(
                y_vals,
                tail,
                left=left,
                height=bar_height,
                color=colors["no_load"],
                edgecolor="white",
            )

            # axes cosmetics
            ax.set_xlim(0, 1)
            ax.set_xlabel("fraction of cases")
            ax.set_ylabel(y)
            ax.set_yticks(y_vals)
            ax.set_yticklabels(y_vals)
            title_bits = []
            if r is not None:
                title_bits.append(f"recipe={r}")
            if c is not None:
                title_bits.append(f"cmake={c}")
            ax.set_title(" | ".join(title_bits))

    # lean legend: stages only, in flipped order
    handles, labels = axes[-1][-1].get_legend_handles_labels()
    keep = [i for i, lab in enumerate(labels) if lab in stage_order]
    handles = [handles[i] for i in keep]
    labels = [labels[i] for i in keep]
    # order legend to match stage_order
    ordered = [labels.index(s) for s in stage_order if s in labels]
    handles = [handles[i] for i in ordered]
    labels = [labels[i] for i in ordered]
    fig.legend(
        handles, labels, loc="upper center", ncols=len(stage_order), frameon=False
    )

    plt.show()


def plot_scores_over_attempts(
    df_runs: pd.DataFrame,
    title: str = "Scores across attempts",
):
    _apply_style()
    agg = df_runs.groupby("attempt_num").agg(
        {
            # "num_tokens": ["mean", "std"],
            "dependency_score": ["mean", "std"],
            "variants_score": ["mean", "std"],
        }
    )

    # Properly flatten MultiIndex columns like ('num_tokens','mean') -> 'num_tokens_mean'
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]

    for col in ["dependency_score", "variants_score"]:
        plt.errorbar(
            agg.index,
            agg[f"{col}_mean"],
            yerr=agg[f"{col}_std"],
            label=col,
            marker="o",
            capsize=3,
        )

    plt.xlabel("Attempt")
    plt.ylabel("mean ± std_dev")
    plt.title(title)
    plt.legend()
    plt.show()


def hist_attempts(df_runs: pd.DataFrame):
    plt.figure(figsize=(7, 5))
    df_runs["attempt_num"].hist(
        bins=range(1, int(df_runs["attempt_num"].max()) + 2), align="left", rwidth=0.8
    )
    plt.xlabel("Attempt number")
    plt.ylabel("Count")
    plt.title("Histogram of Attempts Needed to Succeed")
    plt.xticks(range(1, int(df_runs["attempt_num"].max()) + 1))
    plt.show()

    attempts = df_runs["attempt_num"].sort_values()
    cdf = attempts.rank(method="average", pct=True)

    plt.figure(figsize=(7, 5))
    plt.step(attempts, cdf, where="post")
    plt.xlabel("Attempt number")
    plt.ylabel("CDF (success probability by attempt)")
    plt.title("CDF of Attempts Needed to Succeed")
    plt.grid(True)
    plt.show()


def build_stage_survival_table_sets(
    df: pd.DataFrame,
    model: str,
    attempt_min: int = 1,
    attempt_max: int = 5,
    stage_order=("load", "concretize", "install"),
):
    """
    Returns tidy survival table using inclusive status sets (the 'right way'):
        attempt_num | stage | fraction

    Semantics:
      - A run is counted as having reached a stage at attempt a if ANY row for that run
        in attempts <= a has a status in the inclusive set for that stage.
      - Denominator = all unique runs for the model (even if some have no rows at some attempts).
    """

    # ---- filter to model; keep all runs for denominator
    df_all = df[df["model"] == model].copy()
    if df_all.empty:
        raise ValueError(f"No rows for model={model}")

    runs = df_all["run_id"].dropna().unique()
    if len(runs) == 0:
        raise ValueError(f"No run_id for model={model}")

    # ---- keep only attempts in window (compute on this slice but denom is all runs)
    dfm = df_all[df_all["attempt_num"].between(attempt_min, attempt_max)].copy()

    # ---- inclusive sets (exactly as you specified)
    load_set = {
        "load",
        "concretize",
        "concretize_fail",
        "install",
        "install_fail",
        "test",
        "test_fail",
    }
    concretize_set = {"concretize", "install", "install_fail", "test", "test_fail"}
    install_set = {"install", "test", "test_fail"}

    # Map stage -> set, preserving your stage_order
    stage_sets = {
        "load": load_set,
        "concretize": concretize_set,
        "install": install_set,
    }
    # If you include "test" as a stage, add:
    if "test" in stage_order and "test" not in stage_sets:
        stage_sets["test"] = {"test"}  # only 'test' counts as ≥ test

    # ---- per-row boolean flags for each stage (0/1)
    for s in stage_order:
        st = stage_sets[s]
        dfm[s] = dfm["status"].isin(st).astype(int)

    # ---- collapse to one row per (run_id, attempt_num):
    # within same attempt for a run, if ANY row reached the stage → 1
    gb = dfm.groupby(["run_id", "attempt_num"], as_index=False)[list(stage_order)].max()

    # ---- build full run × attempt grid and left join, fill missing with 0
    attempts = np.arange(attempt_min, attempt_max + 1)
    grid = pd.MultiIndex.from_product(
        [runs, attempts], names=["run_id", "attempt_num"]
    ).to_frame(index=False)
    full = grid.merge(gb, on=["run_id", "attempt_num"], how="left").fillna(0)
    full[list(stage_order)] = full[list(stage_order)].astype(int)

    # ---- by-run cumulative OR up to each attempt (monotone survival)
    full = full.sort_values(["run_id", "attempt_num"])
    for s in stage_order:
        full[s] = full.groupby("run_id")[s].cummax()

    # ---- aggregate: fraction across runs at each attempt
    total_runs = len(runs)
    records = []
    for a in attempts:
        sub = full[full["attempt_num"] == a]
        for s in stage_order:
            frac = sub[s].sum() / total_runs
            records.append({"attempt_num": a, "stage": s, "fraction": float(frac)})

    out = pd.DataFrame(records).sort_values(["attempt_num", "stage"])
    breakpoint()
    return out


def plot_stage_survival_incremental_bars(
    df_runs: pd.DataFrame,
    model: str,
    title: str | None = None,
    stage_order=("load", "concretize", "install"),
    attempt_min: int = 1,
    attempt_max: int = 5,
    annotate_threshold: float = 0.03,
):
    """
    Horizontal stacked bars showing how much each attempt contributes to the
    % of runs reaching ≥ each stage. Bold yellow→purple palette.
    """
    _apply_style()

    df_surv = build_stage_survival_table_sets(
        df_runs,
        model=model,
        attempt_min=attempt_min,
        attempt_max=attempt_max,
        stage_order=stage_order,  # e.g., ("load","concretize","install") or include "test"
    )
    piv = df_surv.pivot(
        index="stage", columns="attempt_num", values="fraction"
    ).reindex(stage_order)
    attempts = np.arange(attempt_min, attempt_max + 1)

    # Incremental deltas
    deltas = piv.copy()
    for a in attempts:
        deltas[a] = (
            piv[a] if a == attempt_min else (piv[a] - piv[a - 1]).clip(lower=0.0)
        )

    # 🔥 Hardcoded bold color palette (yellow → purple)
    bold_colors = [
        "#F5D547",  # bright yellow
        "#F28E2B",  # orange
        "#E15759",  # red
        "#76B7B2",  # teal
        "#4E79A7",  # blue
        "#7B3F95",  # purple
    ]
    colors = {a: bold_colors[a - attempt_min] for a in attempts}

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    y_positions = np.arange(len(stage_order))
    left = np.zeros(len(stage_order), dtype=float)

    for a in attempts:
        width = deltas[a].values
        color = colors[a % len(bold_colors)]
        ax.barh(
            y_positions,
            width,
            left=left,
            color=color,
            edgecolor="none",
            label=f"Attempt {a}",
        )

        # annotate inside segment if big enough
        for i, w in enumerate(width):
            if w >= annotate_threshold:
                x_center = left[i] + w / 2
                ax.text(
                    x_center,
                    y_positions[i],
                    f"{100 * w:.0f}%",
                    va="center",
                    ha="center",
                    fontsize=8,
                    color="white",
                )
        left += width

    # Axis and layout
    ax.set_yticks(y_positions)
    ax.set_yticklabels(stage_order)
    ax.set_xlabel("Percentage of Runs Reaching ≥ Stage")
    ax.set_xlim(0, 1.06)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(title or f"Incremental Attempt Contributions — {model}")
    ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(title=None, bbox_to_anchor=(1.02, 1), loc="upper left")

    return fig, ax


def plot_score_success_scatter(
    df: pd.DataFrame,
    score: str,
):
    """
    Scatter plot of variants_score vs binary success (status == SUCCESS_STAGE)
    with a linear regression fit.
    """
    _apply_style()

    x_name = f"{score}_score"

    # Prepare data
    df_valid = df.dropna(subset=[x_name, "status"]).copy()
    df_valid["success"] = (df_valid["status"] == SUCCESS_STAGE).astype(int)

    x = df_valid[x_name].astype(float)
    y = df_valid["success"].astype(float)

    # Fit linear regression
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    y_pred = intercept + slope * x

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.4, s=20, label="Data")
    ax.plot(
        np.sort(x),
        intercept + slope * np.sort(x),
        color="red",
        linewidth=2,
        label="fit",
    )

    ax.set_xlabel(x_name)
    ax.set_ylabel(f"{SUCCESS_STAGE}ed")
    ax.set_title(
        f"{x_name} vs success\n$r={r_value:.3f}$, p={p_value:.3g}, slope={slope:.3f}"
    )
    ax.set_ylim(-0.1, 1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()

    fig.tight_layout()
    return (
        fig,
        ax,
        {
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
        },
    )


def hm(
    df: pd.DataFrame,
    mode: str = "scores",  # "scores", "passk", or "avg_attempts"
    k: int = 1,  # attempt k for pass@k
    cmap_scores="Greens",
    cmap_pass="Reds",
    cmap_attempts="Purples",
):
    """
    Grid of heatmaps (rows = metrics, cols = buildsys).

    - mode="scores": dependency_score, variants_score in [0,1]
    - mode="passk" : pass@k (1 if success at attempt==k)
    - mode="avg_attempts": mean attempt_num where status==SUCCESS_STAGE
        * lower = better (darker color)
        * colorbar direction is standard (low at bottom, high at top)
    """

    required = {"buildsys", "model", "reference"}
    if not required.issubset(df.columns):
        raise ValueError("Expected columns: 'buildsys', 'model', 'reference'.")

    # --- Choose metrics and preprocess ---
    if mode == "scores":
        scores = ["dependency_score", "variants_score"]
        if any(s not in df.columns for s in scores):
            raise ValueError(
                "Expected columns: 'dependency_score' and 'variants_score'."
            )
        row_titles = [s.replace("_", " ").title() for s in scores]
        cmap = cmap_scores
        fmt = ".2g"

    elif mode == "passk":
        if not {"status", "attempt_num"}.issubset(df.columns):
            raise ValueError("mode='passk' requires columns: 'status', 'attempt_num'.")
        df = df.copy()
        pass_col = f"pass@{k}"
        df[pass_col] = (
            (df["status"] == SUCCESS_STAGE) & (df["attempt_num"] == k)
        ).astype(int)
        scores = [pass_col]
        row_titles = [pass_col]
        cmap = cmap_pass
        fmt = ".2g"

    elif mode == "avg_attempts":
        if not {"status", "attempt_num"}.issubset(df.columns):
            raise ValueError(
                "mode='avg_attempts' requires columns: 'status', 'attempt_num'."
            )
        df = df.copy()
        metric = "avg_attempts_to_success"
        df[metric] = df["attempt_num"].where(df["status"] == SUCCESS_STAGE)
        scores = [metric]
        row_titles = ["mean attempts to success"]
        # use reversed cmap so lower (better) = darker
        cmap = cmap_attempts + "_r"
        fmt = ".2f"

    else:
        raise ValueError("mode must be 'scores', 'passk', or 'avg_attempts'.")

    builds = sorted(df["buildsys"].unique())
    all_models = sorted(df["model"].unique())
    all_refs = sorted(df["reference"].unique())

    # --- Compute global ranges per metric ---
    ranges = {}
    for s in scores:
        if mode in ("scores", "passk"):
            vmin, vmax = 0.0, 1.0
        else:
            vals = pd.to_numeric(df[s], errors="coerce").to_numpy()
            if np.all(np.isnan(vals)):
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.nanmin(vals))
                vmax = float(np.nanmax(vals))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = 0.0, 1.0
        ranges[s] = (vmin, vmax)

    # --- Create figure ---
    fig, axes = plt.subplots(
        nrows=len(scores),
        ncols=len(builds),
        figsize=(5.5 * len(builds), 4.8 * len(scores)),
        constrained_layout=False,
    )

    # normalize axes to 2D list
    if len(scores) == 1 and len(builds) == 1:
        axes = [[axes]]
    elif len(scores) == 1:
        axes = [axes]
    elif len(builds) == 1:
        axes = [[ax] for ax in axes]

    for i, score in enumerate(scores):
        vmin, vmax = ranges[score]
        for j, bsys in enumerate(builds):
            subset = df[df["buildsys"] == bsys]
            pivot = subset.pivot_table(
                values=score,
                index="reference",
                columns="model",
                aggfunc="mean",
            ).reindex(index=all_refs, columns=all_models)

            show_cbar = j == len(builds) - 1
            heat = sns.heatmap(
                pivot,
                ax=axes[i][j],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt=fmt,
                cbar=show_cbar,
                linewidths=0.6,
                linecolor="lightgray",
                mask=pivot.isna(),
                square=True,
            )

            if show_cbar:
                cbar = heat.collections[0].colorbar
                cbar.set_label(row_titles[i], rotation=270, labelpad=20, fontsize=11)
                # keep standard direction (low at bottom, high at top)

            axes[i][j].set_title(f"{bsys}", fontsize=12, pad=10)
            axes[i][j].set_xlabel("")
            axes[i][j].set_ylabel("" if j > 0 else "Reference")

    plt.subplots_adjust(
        left=0.08, right=0.92, top=0.90, bottom=0.08, wspace=0.25, hspace=0.35
    )
    plt.show()


def plot_attempt_deltas(df):
    """
    Compute and plot side-by-side histograms of concretize and install deltas.
    Delta = first <success>.attempt_num - first <fail>.attempt_num preceding it.
    Includes mean (red dashed) and median (green dotted) lines.
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
        return succ_attempt - first_fail  # e.g., fail@2 ... success@4 → 2

    conc_deltas, inst_deltas = [], []
    for (_, _), grp in d.groupby(["run_id", "pkg_name"], sort=False):
        conc = delta_first_fail(grp, "concretize_fail", "concretize")
        inst = delta_first_fail(grp, "install_fail", "install")
        if conc is not None:
            conc_deltas.append(conc)
        if inst is not None:
            inst_deltas.append(inst)

    # Nothing to plot?
    if not conc_deltas and not inst_deltas:
        print("No fail→success transitions to plot.")
        return

    # Determine bins across both datasets for consistency
    all_max = 1
    if conc_deltas:
        all_max = max(all_max, max(conc_deltas))
    if inst_deltas:
        all_max = max(all_max, max(inst_deltas))
    bins = range(1, int(all_max) + 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    def plot_hist(ax, data, title):
        if not data:
            ax.set_title(f"{title}\n(no data)")
            return
        mean_val = pd.Series(data).mean()
        median_val = pd.Series(data).median()
        ax.hist(data, bins=bins, color="orange", edgecolor="black", alpha=0.8)
        ax.axvline(
            mean_val,
            color="red",
            linestyle="dashed",
            linewidth=1.5,
            label=f"Mean = {mean_val:.2f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="dotted",
            linewidth=1.5,
            label=f"Median = {median_val:.2f}",
        )
        ax.set_title(f"# Recovery Attempts ({title})")
        ax.set_xlabel("Attempts from first fail → first success")
        ax.set_ylabel("Count")
        ax.legend()

    breakpoint()

    # Concretize histogram
    plot_hist(axes[0], conc_deltas, "concretize")
    # Install histogram
    plot_hist(axes[1], inst_deltas, "install")

    fig.suptitle("Recovery From Failure", fontsize=14)
    fig.tight_layout()
    plt.show()


def plot_install_scores(df, agg="mean"):
    """
    Two horizontal bar charts (variants_score and dependency_score)
    with one bar per (model, raw_buildsys, reference) combo,
    using only rows where status=='install'.
    """
    required = {
        "raw_buildsys",
        "reference",
        "model",
        "status",
        "dependency_score",
        "variants_score",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    dfi = df[df["status"] == "install"].copy()
    if dfi.empty:
        print("No rows with status=='install'.")
        return

    group_cols = ["model", "raw_buildsys", "reference"]
    scores = ["variants_score", "dependency_score"]

    if agg == "median":
        grouped = dfi.groupby(group_cols, as_index=False)[scores].median(
            numeric_only=True
        )
    else:
        grouped = dfi.groupby(group_cols, as_index=False)[scores].mean(
            numeric_only=True
        )

    grouped["combo"] = grouped.apply(
        lambda r: f"{r['model']} | {r['raw_buildsys']} | {r['reference']}", axis=1
    )

    def barh(score_col, title):
        g = grouped[["combo", score_col]].sort_values(score_col, ascending=True)
        fig_h = max(4, 0.4 * len(g))
        fig, ax = plt.subplots(figsize=(12, fig_h))
        ax.barh(g["combo"], g[score_col])
        ax.set_xlabel(title)
        ax.set_ylabel("model | raw_buildsys | reference")
        ax.set_title(f"{title} — status=install")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        plt.tight_layout()
        plt.show()

    barh("variants_score", "Variants score")
    barh("dependency_score", "Dependency score")


def hme(
    df: pd.DataFrame,
    mode: str = "scores",  # "scores", "passk", or "avg_attempts"
    k: int = 1,
    cmap_scores="Greens",
    cmap_pass="Reds",
    cmap_attempts="Purples",
    *,
    cell_size: float = 0.25,  # inches per matrix cell (smaller => tighter)
    cbar_width_in: float = 0.18,  # colorbar column width in *inches* (fixed)
    # explicit outer padding in inches (fixed, not dynamic)
    pad_left_in: float = 0.6,
    pad_right_in: float = 0.35,
    pad_top_in: float = 0.35,
    pad_bottom_in: float = 0.65,
    font_scale: float = 0.85,
    annot_fontsize: int = 8,
    tick_fontsize: int = 8,
    title_fontsize: int = 11,
    xtick_rotation: int = 45,
):
    required = {"buildsys", "model", "reference"}
    if not required.issubset(df.columns):
        raise ValueError("Expected columns: 'buildsys', 'model', 'reference'.")

    # --- Choose metrics and preprocess ---
    if mode == "scores":
        scores = ["dependency_score", "variants_score"]
        if any(s not in df.columns for s in scores):
            raise ValueError(
                "Expected columns: 'dependency_score' and 'variants_score'."
            )
        row_titles = [s.replace("_", " ").title() for s in scores]
        cmap = cmap_scores
        fmt = ".2g"

    elif mode == "passk":
        if not {"status", "attempt_num"}.issubset(df.columns):
            raise ValueError("mode='passk' requires columns: 'status', 'attempt_num'.")
        df = df.copy()
        pass_col = f"pass@{k}"
        df[pass_col] = (
            (df["status"] == SUCCESS_STAGE) & (df["attempt_num"] == k)
        ).astype(int)
        scores = [pass_col]
        row_titles = [pass_col]
        cmap = cmap_pass
        fmt = ".2g"

    elif mode == "avg_attempts":
        if not {"status", "attempt_num"}.issubset(df.columns):
            raise ValueError(
                "mode='avg_attempts' requires columns: 'status', 'attempt_num'."
            )
        df = df.copy()
        metric = "avg_attempts_to_success"
        df[metric] = df["attempt_num"].where(df["status"] == SUCCESS_STAGE)
        scores = [metric]
        row_titles = ["mean attempts to success"]
        cmap = cmap_attempts + "_r"
        fmt = ".2f"
    else:
        raise ValueError("mode must be 'scores', 'passk', or 'avg_attempts'.")

    builds = sorted(df["buildsys"].unique())
    all_models = sorted(df["model"].unique())
    all_refs = sorted(df["reference"].unique())

    # --- Global ranges ---
    ranges = {}
    for s in scores:
        if mode in ("scores", "passk"):
            vmin, vmax = 0.0, 1.0
        else:
            vals = pd.to_numeric(df[s], errors="coerce").to_numpy()
            if np.all(np.isnan(vals)):
                vmin, vmax = 0.0, 1.0
            else:
                vmin = float(np.nanmin(vals))
                vmax = float(np.nanmax(vals))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = 0.0, 1.0
        ranges[s] = (vmin, vmax)

    # --- Size from data shape ---
    n_rows = len(scores)
    n_cols = len(builds)
    n_models = max(1, len(all_models))
    n_refs = max(1, len(all_refs))

    panel_w_in = n_models * cell_size
    panel_h_in = n_refs * cell_size

    # physical spacing between panels (inches)
    wspace_in = 0.25
    hspace_in = 0.35

    # total figure size in inches (panels + spaces + fixed pads + fixed cbar)
    fig_w_in = (
        n_cols * panel_w_in
        + (n_cols - 1) * wspace_in
        + cbar_width_in
        + pad_left_in
        + pad_right_in
    )
    fig_h_in = (
        n_rows * panel_h_in + (n_rows - 1) * hspace_in + pad_top_in + pad_bottom_in
    )

    # Convert inch paddings to figure fractions for GridSpec "rect"
    left = pad_left_in / fig_w_in
    right = 1.0 - pad_right_in / fig_w_in
    bottom = pad_bottom_in / fig_h_in
    top = 1.0 - pad_top_in / fig_h_in

    # width ratios in *inches* so cbar stays fixed-physical width
    width_ratios_in = [panel_w_in] * n_cols + [cbar_width_in]
    # Matplotlib normalizes ratios, but since fig width includes the same inches,
    # the cbar column ends up with ~cbar_width_in physical width.

    # turn on constrained_layout, but with a defined rect for fixed margins
    plt.rcParams.update({"font.size": 10 * font_scale})
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols + 1,
        width_ratios=width_ratios_in,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        # convert inch gaps to fractions of a single panel size (keeps feel constant)
        wspace=wspace_in / max(panel_w_in, 1e-6),
        hspace=hspace_in / max(panel_h_in, 1e-6),
    )

    # Build axes; share y per row
    axes = []
    for i in range(n_rows):
        row_axes = []
        for j in range(n_cols):
            sharey_ax = row_axes[0] if j > 0 else None
            ax = fig.add_subplot(gs[i, j], sharey=sharey_ax)
            row_axes.append(ax)
        axes.append(row_axes)

    # --- Draw heatmaps (no inline cbar) + one fixed-width cbar per row ---
    for i, score in enumerate(scores):
        vmin, vmax = ranges[score]
        sm = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap
        )

        for j, bsys in enumerate(builds):
            subset = df[df["buildsys"] == bsys]
            pivot = subset.pivot_table(
                values=score, index="reference", columns="model", aggfunc="mean"
            ).reindex(index=all_refs, columns=all_models)

            ax = axes[i][j]
            sns.heatmap(
                pivot,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                annot=True,
                fmt=fmt,
                annot_kws={"size": annot_fontsize},
                cbar=False,
                linewidths=0.4,
                linecolor="lightgray",
                mask=pivot.isna(),
            )
            ax.set_aspect("auto")

            # Titles & labels
            ax.set_title(f"{bsys}", fontsize=title_fontsize, pad=6)
            ax.set_xlabel("")
            # y labels only on the leftmost column
            if j == 0:
                ax.set_ylabel("Reference", fontsize=tick_fontsize)
                ax.tick_params(axis="y", labelleft=True, labelsize=tick_fontsize)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

            # ensure x tick labels are visible on *every* heatmap
            ax.tick_params(
                axis="x",
                labelrotation=xtick_rotation,
                labelsize=tick_fontsize,
                labelbottom=True,
            )

        # Dedicated, fixed-width colorbar column
        cax = fig.add_subplot(gs[i, -1])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.ax.tick_params(labelsize=tick_fontsize)
        cbar.set_label(
            row_titles[i], rotation=270, labelpad=10, fontsize=title_fontsize
        )

    plt.show()


import pandas as pd


def cumulative_success_df(
    df_runs: pd.DataFrame,
    by: str = "model",
    facet_by: str | None = None,
) -> pd.DataFrame:
    """
    Build a tidy DataFrame of cumulative success by attempt.

    Output columns:
      - group: value from `by`
      - facet: value from `facet_by` (or None)
      - attempt: int
      - successes_cum: int
      - denom: int
      - rate: float in [0,1]
    """
    empty_out = pd.DataFrame(
        columns=["group", "facet", "attempt", "successes_cum", "denom", "rate"]
    )

    if df_runs.empty or "attempt_num" not in df_runs.columns:
        return empty_out

    # global attempt bins
    attempts_all = df_runs["attempt_num"].dropna()
    if attempts_all.empty:
        return empty_out
    bins = np.arange(
        attempts_all.round().astype(int).min(),
        attempts_all.round().astype(int).max() + 1,
    )

    # capture ordered category definitions (if any) to reapply later
    group_cats = None
    if by in df_runs.columns and pd.api.types.is_categorical_dtype(df_runs[by]):
        if df_runs[by].cat.ordered:
            group_cats = df_runs[by].cat.categories

    facet_cats = None
    if (
        facet_by is not None
        and facet_by in df_runs.columns
        and pd.api.types.is_categorical_dtype(df_runs[facet_by])
    ):
        if df_runs[facet_by].cat.ordered:
            facet_cats = df_runs[facet_by].cat.categories

    # facets list (preserve ordered categorical order if present)
    if facet_by is None or facet_by not in df_runs.columns:
        facets = [None]
    else:
        if facet_cats is not None:
            facets = list(facet_cats)
        else:
            # unique() preserves first-occurrence order
            vals = df_runs[facet_by].dropna().unique().tolist()
            facets = vals if vals else [None]

    rows = []

    for facet in facets:
        sub = df_runs if facet is None else df_runs.loc[df_runs[facet_by] == facet]

        if sub.empty or by not in sub.columns:
            continue

        # (Optional) trim unused cats in this slice so groupby won't emit empty groups
        if pd.api.types.is_categorical_dtype(sub[by]):
            sub = sub.copy()
            sub[by] = sub[by].cat.remove_unused_categories()
        if (
            facet_by is not None
            and facet_by in sub.columns
            and pd.api.types.is_categorical_dtype(sub[facet_by])
        ):
            sub = sub.copy()
            sub[facet_by] = sub[facet_by].cat.remove_unused_categories()

        # Prevent implicit resorting and exclude unused categories
        for group_val, grp in sub.groupby(by, dropna=True, sort=False, observed=True):
            att = grp["attempt_num"].dropna().round().astype(int)
            if att.empty:
                continue
            denom = int(len(att))

            succ_att = (
                grp.loc[grp["status"] == SUCCESS_STAGE, "attempt_num"]
                .dropna()
                .round()
                .astype(int)
            )

            succ_counts = (
                succ_att.value_counts().reindex(bins, fill_value=0).sort_index()
            )
            successes_cum = succ_counts.cumsum().astype(int)
            rate = (successes_cum / denom).astype(float)

            rows.extend(
                {
                    "group": group_val,
                    "facet": facet,
                    "attempt": int(k),
                    "successes_cum": int(s_cum),
                    "denom": denom,
                    "rate": float(r),
                }
                for k, s_cum, r in zip(bins, successes_cum.tolist(), rate.tolist())
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return empty_out

    # Reapply ordered categoricals using only observed levels to avoid empty groups
    if group_cats is not None:
        present_groups = pd.unique(out["group"])
        group_cats_observed = [c for c in group_cats if c in set(present_groups)]
        out["group"] = pd.Categorical(
            out["group"], categories=group_cats_observed, ordered=True
        )

    if facet_by is not None and facet_cats is not None and "facet" in out.columns:
        present_facets = pd.unique(out["facet"])
        facet_cats_observed = [c for c in facet_cats if c in set(present_facets)]
        out["facet"] = pd.Categorical(
            out["facet"], categories=facet_cats_observed, ordered=True
        )

    # Stable sort respects categorical order where present
    out = out.sort_values(["facet", "group", "attempt"], kind="mergesort").reset_index(
        drop=True
    )
    return out
