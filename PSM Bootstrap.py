from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    PerfectSeparationWarning,
    PerfectSeparationError,
)
from tqdm import tqdm

# ---------------------------------------------------------------------
# Quiet the spammy warnings from separation/convergence
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=PerfectSeparationWarning)


# ---------------------------------------------------------------------
# Core simulation + IPTW helpers
# ---------------------------------------------------------------------


def _simulate_once(sample_size: int) -> pd.DataFrame:
    """
    One draw of the DGP using the EXACT attribution function requested:

        T = np.clip(
                np.where(Z1, np.random.choice([0, 1]), 0)
                + np.where(Z2, np.random.choice([0, 1]), 0),
                0,
                1,
            )

    Important: np.random.choice is called twice and returns SCALARS, which act as
    global switches shared by every row. That yields perfect separation by (Z1, Z2).
    """
    # Use np.random.* everywhere for reproducibility with np.random.seed(...)
    W1 = np.random.normal(0, 1, sample_size)
    W2 = np.random.normal(0, 1, sample_size)

    Z1 = (W1 > 1.96).astype(int)
    Z2 = (W2 > 1.96).astype(int)

    # ---- EXACT requested attribution (do not change) ----
    T = np.clip(
        np.where(Z1, np.random.choice([0, 1]), 0)
        + np.where(Z2, np.random.choice([0, 1]), 0),
        0,
        1,
    ).astype(int)
    # -----------------------------------------------------

    Y = 0.7 * T + 1.5 * W1 + 0.3 * np.random.normal(0, 1, sample_size)

    return pd.DataFrame({"W1": W1, "W2": W2, "Z1": Z1, "Z2": Z2, "T": T, "Y": Y})


def _infer_switches_from_data(df: pd.DataFrame) -> tuple[float, float]:
    """
    Infer the realized scalar switches (c1, c2) from the data implied by the exact
    attribution. Uses groups where only one indicator is on:
      - c1 from group Z1=1, Z2=0 (T must equal c1 there)
      - c2 from group Z1=0, Z2=1 (T must equal c2 there)
    Returns (c1_est, c2_est) as 0/1 floats or NaN if the group is empty.
    """
    c1_group = df[(df.Z1 == 1) & (df.Z2 == 0)]["T"]
    c2_group = df[(df.Z1 == 0) & (df.Z2 == 1)]["T"]

    def _const_or_nan(s: pd.Series) -> float:
        if len(s) == 0:
            return float("nan")
        u = s.unique()
        return float(u[0]) if len(u) == 1 else float("nan")

    return _const_or_nan(c1_group), _const_or_nan(c2_group)


def _fit_iptw_weights(
    df: pd.DataFrame,
    stabilized: bool = False,
    truncate: Optional[tuple] = None,
    clip: float = 1e-6,
) -> tuple[Optional[pd.DataFrame], dict]:
    """
    Fit (stabilized) IPTW with a logit model P(T|W1,W2).
    Returns (df_with_weights_or_None, diagnostics_dict).
    """
    diags: dict = {}

    if df["T"].nunique() < 2:
        diags["fit_status"] = "degenerate_outcome"
        return None, diags

    denom_formula = "T ~ W1 + W2"
    numer_formula = "T ~ 1"

    base = df.copy()
    try:
        treat_model = smf.logit(denom_formula, data=base).fit(disp=False)
        diags["fit_status"] = "ok"
    except PerfectSeparationError:
        treat_model = smf.logit(denom_formula, data=base).fit_regularized(
            alpha=1e-4, L1_wt=0.0, disp=False
        )
        diags["fit_status"] = "regularized_due_to_separation"
    except Exception as e:
        diags["fit_status"] = f"fit_failed:{type(e).__name__}"
        return None, diags

    p_denom = np.clip(treat_model.predict(base), clip, 1 - clip)

    # Stabilized numerator
    try:
        numer = smf.logit(numer_formula, data=base).fit(disp=False)
        p_numer = np.clip(numer.predict(base), clip, 1 - clip)
    except Exception:
        p_numer = np.repeat(np.clip(base["T"].mean(), clip, 1 - clip), len(base))

    T_vals = base["T"].astype(int).values
    if stabilized:
        w = np.where(T_vals == 1, p_numer / p_denom, (1 - p_numer) / (1 - p_denom))
    else:
        w = np.where(T_vals == 1, 1.0 / p_denom, 1.0 / (1.0 - p_denom))

    if truncate is not None:
        w = np.clip(w, truncate[0], truncate[1])

    out = base.copy()
    out["p_t_denom"] = p_denom
    out["p_t_numer"] = p_numer
    out["w_treat"] = w.astype(float)

    min_overlap = float(np.minimum(p_denom, 1 - p_denom).min())
    diags.update(
        {
            "p_min": float(np.min(p_denom)),
            "p_max": float(np.max(p_denom)),
            "p_lt_0.01": float(np.mean(p_denom < 0.01)),
            "p_gt_0.99": float(np.mean(p_denom > 0.99)),
            "overlap_margin_min": min_overlap,
            "w_mean": float(np.mean(w)),
            "w_std": float(np.std(w, ddof=1)),
            "w_min": float(np.min(w)),
            "w_max": float(np.max(w)),
            "w_q95": float(np.quantile(w, 0.95)),
            "w_q99": float(np.quantile(w, 0.99)),
            "ess_all": float((w.sum() ** 2) / (np.sum(w**2))),
            "ess_treated": float(
                (w[df["T"] == 1].sum() ** 2) / np.sum(w[df["T"] == 1] ** 2)
                if (df["T"] == 1).any()
                else np.nan
            ),
            "ess_control": float(
                (w[df["T"] == 0].sum() ** 2) / np.sum(w[df["T"] == 0] ** 2)
                if (df["T"] == 0).any()
                else np.nan
            ),
        }
    )
    return out, diags


def _check_perfect_separation_by_groups(df: pd.DataFrame) -> bool:
    """
    True if within each (Z1,Z2) stratum, T is constant.
    """
    g = df.groupby(["Z1", "Z2"])["T"].nunique()
    return bool((g <= 1).all())


# ---------------------------------------------------------------------
# Bootstrap with richer plots (uses exact attribution above)
# ---------------------------------------------------------------------


def bootstrap_instability_of_attribution(
    n_boot: int = 200,
    sample_size: int = 100_000,
    *,
    stabilized: bool = False,
    truncate: Optional[tuple] = None,
    seed: Optional[int] = 0,
    make_plots: bool = True,
) -> dict:
    """
    Re-simulate n_boot times and collect diagnostics using the EXACT T attribution.
    """
    # Seed legacy RNG used by np.random.normal/choice to keep everything reproducible
    if seed is not None:
        np.random.seed(seed)

    rows = []
    for b in tqdm(range(n_boot), desc="Bootstrap replicates", unit="rep"):
        df = _simulate_once(sample_size)
        treated_rate = float(df["T"].mean())
        sep_groups = _check_perfect_separation_by_groups(df)
        c1_est, c2_est = _infer_switches_from_data(df)

        _, diags = _fit_iptw_weights(
            df, stabilized=stabilized, truncate=truncate, clip=1e-6
        )

        rows.append(
            {
                "boot": b,
                "c1_est": c1_est,
                "c2_est": c2_est,
                "treated_rate": treated_rate,
                "perfect_sep_groups": sep_groups,
                **{
                    k: diags.get(k, np.nan)
                    for k in [
                        "fit_status",
                        "p_min",
                        "p_max",
                        "p_lt_0.01",
                        "p_gt_0.99",
                        "overlap_margin_min",
                        "w_mean",
                        "w_std",
                        "w_min",
                        "w_max",
                        "w_q95",
                        "w_q99",
                        "ess_all",
                        "ess_treated",
                        "ess_control",
                    ]
                },
            }
        )

    results = pd.DataFrame(rows)

    # Switch frequency table (inferred)
    counts_by_switch = (
        results.dropna(subset=["c1_est", "c2_est"])
        .groupby(["c1_est", "c2_est"])
        .size()
        .rename("n")
        .reset_index()
    )

    figures = {}
    if make_plots:
        # 1) Treated prevalence across replicates
        fig1 = plt.figure(figsize=(7, 4), dpi=150)
        ax1 = fig1.gca()
        ax1.hist(results["treated_rate"], bins=30, edgecolor="black")
        qs = results["treated_rate"].quantile([0.25, 0.5, 0.75]).values
        for q, lab in zip(qs, ["Q1", "Median", "Q3"]):
            ax1.axvline(q, linestyle="--")
            ax1.text(q, ax1.get_ylim()[1] * 0.9, lab, rotation=90, va="top", ha="right")
        ax1.set_xlabel("Treated rate per replicate")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Instability of treated prevalence across re-simulations")
        figures["treated_rate_hist"] = fig1

        # 2) Distribution of log10(max weights)
        wmax = results["w_max"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(wmax):
            fig2 = plt.figure(figsize=(7, 4), dpi=150)
            ax2 = fig2.gca()
            ax2.hist(np.log10(wmax), bins=30, edgecolor="black")
            ax2.set_xlabel("log10(max IPTW) per replicate")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Heavy-tailed maximum weights (positivity problems)")
            figures["wmax_log_hist"] = fig2

        # 3) ESS vs treated prevalence (log y)
        ess_mask = results["ess_all"].notna()
        if ess_mask.any():
            fig3 = plt.figure(figsize=(7, 4), dpi=150)
            ax3 = fig3.gca()
            ax3.scatter(
                results.loc[ess_mask, "treated_rate"],
                results.loc[ess_mask, "ess_all"],
                s=12,
            )
            ax3.set_xlabel("Treated rate")
            ax3.set_ylabel("ESS (overall)")
            ax3.set_yscale("log")
            ax3.set_title("ESS collapses when treated prevalence is extreme")
            figures["ess_vs_treated"] = fig3

        # 4) Boxplot of ESS by inferred (c1, c2)
        if ess_mask.any() and not counts_by_switch.empty:
            fig4 = plt.figure(figsize=(7, 4), dpi=150)
            ax4 = fig4.gca()
            groups, labels = [], []
            for c1 in [0.0, 1.0]:
                for c2 in [0.0, 1.0]:
                    g = results[(results.c1_est == c1) & (results.c2_est == c2)][
                        "ess_all"
                    ].dropna()
                    if len(g):
                        groups.append(g.values)
                        labels.append(f"c1={int(c1)}, c2={int(c2)}")
            if groups:
                ax4.boxplot(groups, labels=labels, showfliers=False)
                ax4.set_ylabel("ESS (overall)")
                ax4.set_title("ESS by inferred (c1, c2)")
                figures["ess_box_by_switch"] = fig4

        # 5) Heatmap of inferred (c1,c2) counts
        if not counts_by_switch.empty:
            fig5 = plt.figure(figsize=(4, 4), dpi=150)
            ax5 = fig5.gca()
            mat = np.zeros((2, 2), dtype=int)
            for _, r in counts_by_switch.iterrows():
                mat[int(r["c1_est"]), int(r["c2_est"])] = int(r["n"])
            im = ax5.imshow(mat, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax5.text(j, i, mat[i, j], ha="center", va="center")
            ax5.set_xticks([0, 1])
            ax5.set_yticks([0, 1])
            ax5.set_xticklabels(["c2=0", "c2=1"])
            ax5.set_yticklabels(["c1=0", "c1=1"])
            ax5.set_title("Counts of inferred (c1, c2)")
            fig5.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
            figures["switch_heatmap"] = fig5

        # 6) Distribution of overlap margin (min_i min{p,1-p})
        ov = results["overlap_margin_min"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ov):
            fig6 = plt.figure(figsize=(7, 4), dpi=150)
            ax6 = fig6.gca()
            ax6.hist(ov, bins=30, edgecolor="black")
            ax6.set_xlabel("Worst-case overlap margin per replicate")
            ax6.set_ylabel("Frequency")
            ax6.set_title("Small margins indicate severe positivity violations")
            figures["overlap_margin_hist"] = fig6

        # 7) Propensity fit status counts
        fig7 = plt.figure(figsize=(7, 4), dpi=150)
        ax7 = fig7.gca()
        status_counts = results["fit_status"].fillna("unknown").value_counts()
        ax7.bar(status_counts.index.astype(str), status_counts.values)
        ax7.set_ylabel("Count")
        ax7.set_title("Propensity fit status across replicates")
        ax7.tick_params(axis="x", rotation=20)
        figures["fit_status_bar"] = fig7

        # 8) Treated rate vs max weight, and ESS
        if len(wmax):
            # Plot w_max and ess_all by treated_rate
            import seaborn as sns

            # X axis: treated_rate, Y axis: w_max and ess_all, not a pairplot, double y-axis
            # plt.figure(figsize=(8, 5), dpi=150)
            # sns.scatterplot(
            #     data=out["results"],
            #     x="treated_rate",
            #     y="w_max",
            #     label="Max IPTW",
            #     color="blue",
            #     s=30,
            # )
            # sns.scatterplot(
            #     data=out["results"],
            #     x="treated_rate",
            #     y="ess_all",
            #     label="ESS (overall)",
            #     color="orange",
            #     s=30,
            # )
            # plt.xlabel("Treated rate")
            # plt.ylabel("Value")
            # plt.title("Max IPTW and ESS vs Treated Rate")
            # plt.yscale("log")
            # # LEGEND
            # plt.show()
            fig8, ax8 = plt.subplots(figsize=(7, 4), dpi=300)
            sns.scatterplot(
                data=results,
                x="treated_rate",
                y="w_max",
                label="Max IPTW",
                color="blue",
                s=30,
                ax=ax8,
            )
            sns.scatterplot(
                data=results,
                x="treated_rate",
                y="ess_all",
                label="ESS (overall)",
                color="orange",
                s=30,
                ax=ax8,
            )
            ax8.set_xlabel("Treated rate")
            ax8.set_ylabel("Value")
            ax8.set_title("Log(Max IPTW) and Log(ESS) vs Treated Rate")
            ax8.set_yscale("log")
            ax8.legend()
            figures["wmax_ess_vs_treated"] = fig8

        plt.tight_layout()

    return {
        "results": results,
        "counts_by_switch": counts_by_switch,
        "figures": figures,
    }


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    out = bootstrap_instability_of_attribution(
        n_boot=200,
        sample_size=10000,
        stabilized=False,
        truncate=None,  # e.g., (0.01, 10.0) to cap extremes
        seed=123,  # global seed for np.random.* (normal + choice)
        make_plots=True,
    )

    print("\nBootstrap summary (first 10 rows):")
    print(out["results"].head(10))

    print("\nCounts by inferred (c1, c2):")
    print(out["counts_by_switch"])

    print("\nKey instability indicators (quantiles across bootstraps):")
    q = out["results"][
        ["treated_rate", "w_max", "ess_all", "overlap_margin_min"]
    ].quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    print(q)

    # Show plots if any
    for name, fig in out["figures"].items():
        print(f"\nDisplaying figure: {name}")
        fig.show()
    print("\nDone.")
