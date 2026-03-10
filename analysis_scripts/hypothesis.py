#!/usr/bin/env python3
"""
Cursor Technique Study — Statistical Analysis Pipeline
======================================================
Usage:
    python analyse.py <data_directory> [--out <output_directory>]
                      [--outlier-method {sd,iqr,mad,none}]
                      [--outlier-sd FLOAT]
                      [--outlier-iqr FLOAT]
                      [--outlier-mad FLOAT]
                      [--min-mt FLOAT]
                      [--max-mt FLOAT]

Reads all P*_session.json files produced by the cursor study app.
Outputs figures (PNG) and tables (CSV) to the output directory.

Required packages:
    pip install numpy pandas scipy matplotlib pingouin statsmodels
"""

import sys
import os
import json
import glob
import argparse
import warnings
from pathlib import Path
from itertools import combinations, product
from statsmodels.stats.anova import AnovaRM

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import pingouin as pg

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ALPHA      = 0.05
TECHNIQUES = ["BUBBLE", "POINT", "AREA"]
MOVEMENTS  = ["STATIC", "SLOW", "FAST"]
DENSITIES  = ["LOW",    "MEDIUM", "HIGH"]

TECH_COLORS = {"BUBBLE": "#e41a1c", "POINT": "#377eb8", "AREA": "#4daf4a"}
MOV_COLORS  = {"STATIC": "#2ca02c", "SLOW":  "#ff7f0e", "FAST": "#d62728"}
DENS_COLORS = {"LOW":    "#9467bd", "MEDIUM":"#8c564b", "HIGH": "#e377c2"}

PRIMARY_DVS = {
    "throughput_shannon_nominal_bps":   "TP Shannon Nominal (bits/s)",
    "throughput_shannon_effective_bps": "TP Shannon Effective (bits/s)",
    "throughput_hoffmann_bps":          "TP Hoffmann (bits/s)",
    "avgTime_ms":                       "Mean MT (ms)",
    "precisionRate_percent":            "Precision Rate (%)",
    "avgErrorClicks":                   "Mean Error Clicks",
    "avgNormalizedDistance":            "Mean Normalised Distance",
    "avgDistanceFromCenter_px":         "Mean Distance to Centre (px)",
}

SECONDARY_DVS = {
    "medianTime_ms":         "Median MT (ms)",
    "stdDevTime_ms":         "SD MT (ms)",
    "effectiveWidth_We_px":  "Effective Width We (px)",
    "endpointSigma_px":      "Endpoint σ (px)",
    "effectiveAvgID_bits":   "Mean ID_e (bits)",
    "shannonAvgID_bits":     "Mean ID (bits)",
}

ANOVA_SOURCE_ORDER = [
    "technique",
    "movement",
    "clustering",
    "technique * movement",
    "technique * clustering",
    "movement * clustering",
    "technique * movement * clustering",
]

# ─────────────────────────────────────────────────────────────────────────────
# OUTLIER REMOVAL CONSTANTS  (all overridable via CLI flags)
# ─────────────────────────────────────────────────────────────────────────────

OUTLIER_METHOD         = "iqr"      # "sd" | "iqr" | "mad" | "none"
OUTLIER_SD_THRESHOLD   = 3.0        # ±N SD from condition mean
OUTLIER_IQR_MULTIPLIER = 3.0        # N × IQR fence  (3 = conservative Tukey)
OUTLIER_MAD_THRESHOLD  = 3.5        # modified Z-score threshold
OUTLIER_MIN_MT_MS      = 50.0       # absolute floor  — anticipatory clicks
OUTLIER_MAX_MT_MS      = 30_000.0   # absolute ceiling — 30 s inattention


# ── after OUTLIER_MAX_MT_MS = 30_000.0 ───────────────────────────────────────
_TTEST_LOG: list[dict] = []          # populated during hypothesis tests; saved in main()


def _log_t(hypothesis, label, test_type, dv, n, t, p,
           d=np.nan, mean_a=np.nan, mean_b=np.nan, mean_diff=np.nan, condition=""):
    """Append one t-test record to the module-level accumulator."""
    def _r(v, dec):
        f = safe_float(v)
        return round(f, dec) if np.isfinite(f) else None

    p_f = safe_float(p)
    _TTEST_LOG.append({
        "hypothesis": hypothesis,
        "label":      label,
        "condition":  condition,
        "dv":         dv,
        "test_type":  test_type,
        "n":          int(n) if n is not None else None,
        "df":         int(n) - 1 if n else None,
        "mean_a":     _r(mean_a,    4),
        "mean_b":     _r(mean_b,    4),
        "mean_diff":  _r(mean_diff, 4),
        "t":          _r(t,         3),
        "p":          _r(p,         4),
        "cohen_d":    _r(d,         3),
        "sig":        bool(np.isfinite(p_f) and p_f < ALPHA),
        "stars":      sig_stars(p_f).strip() if np.isfinite(p_f) else "",
    })


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def savefig(fig, path, dpi=150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {Path(path).name}")


def safe_float(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def cohen_d(a, b):
    """Cohen's d for two paired arrays."""
    diff = np.asarray(a) - np.asarray(b)
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)


def paired_t(a_series, b_series):
    """
    Paired t-test on two Series sharing an index (participant IDs).
    Returns (t, p, cohen_d, mean_diff, n) or NaN values if insufficient data.
    """
    common = a_series.index.intersection(b_series.index)
    if len(common) < 3:
        return np.nan, np.nan, np.nan, np.nan, len(common)
    a = a_series[common].values
    b = b_series[common].values
    t, p = stats.ttest_rel(a, b)
    d    = cohen_d(a, b)
    return float(t), float(p), float(d), float(np.mean(a - b)), len(common)


def sig_stars(p):
    if not np.isfinite(p):
        return "   "
    if p < 0.001:
        return "***"
    if p < 0.01:
        return " **"
    if p < ALPHA:
        return "  *"
    return "   "


def _gg_epsilon(wide_matrix: np.ndarray) -> float:
    """
    Greenhouse-Geisser epsilon from a participants × levels matrix.
    Formula: ε = tr(S̃)² / [(k-1) · tr(S̃²)]
    where S̃ is the doubly-centred covariance matrix.
    """
    S = np.cov(wide_matrix.T)
    k = S.shape[0]
    if k <= 2:
        return 1.0
    row_m  = S.mean(axis=1, keepdims=True)
    col_m  = S.mean(axis=0, keepdims=True)
    grand  = S.mean()
    S_dc   = S - row_m - col_m + grand
    tr     = np.trace(S_dc)
    tr_sq  = np.trace(S_dc @ S_dc)
    if tr_sq < 1e-12:
        return 1.0
    eps = tr ** 2 / ((k - 1) * tr_sq)
    return float(np.clip(eps, 1.0 / (k - 1), 1.0))


def _gg_epsilon_for_factor(df, dv, subject, factor):
    """
    Marginal GG epsilon for one within-subjects factor.
    Averages over all other factors first (gives the marginal covariance).
    """
    try:
        agg   = df.groupby([subject, factor])[dv].mean().reset_index()
        pivot = agg.pivot(index=subject, columns=factor, values=dv).dropna()
        if pivot.shape[0] < 3 or pivot.shape[1] < 3:
            return 1.0
        return _gg_epsilon(pivot.values)
    except Exception:
        return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# OUTLIER REMOVAL
# ─────────────────────────────────────────────────────────────────────────────

def _outlier_mask_1d(values,
                     method       = OUTLIER_METHOD,
                     sd_thresh    = OUTLIER_SD_THRESHOLD,
                     iqr_mult     = OUTLIER_IQR_MULTIPLIER,
                     mad_thresh   = OUTLIER_MAD_THRESHOLD) -> np.ndarray:
    """
    Return a boolean mask (True = outlier) for a 1-D numeric array.

    Parameters
    ----------
    values      : array-like of floats (NaN-safe; NaN entries are never flagged)
    method      : "sd"  — µ ± N·σ  (N = sd_thresh)
                  "iqr" — Q1 − N·IQR … Q3 + N·IQR  (N = iqr_mult)
                  "mad" — modified Z-score > mad_thresh  (Iglewicz & Hoaglin 1993)
                  "none"— no outliers flagged
    """
    arr  = np.asarray(values, dtype=float)
    mask = np.zeros(len(arr), dtype=bool)
    if method == "none":
        return mask

    finite = arr[np.isfinite(arr)]
    if len(finite) < 4:          # too few points — detection unreliable
        return mask

    if method == "sd":
        mu, sigma = np.mean(finite), np.std(finite, ddof=1)
        if sigma < 1e-12:
            return mask
        mask = np.abs(arr - mu) > sd_thresh * sigma

    elif method == "iqr":
        q1, q3 = np.percentile(finite, [25, 75])
        iqr    = q3 - q1
        if iqr < 1e-12:          # degenerate distribution
            return mask
        lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
        mask   = (arr < lo) | (arr > hi)

    elif method == "mad":
        med = np.median(finite)
        mad = np.median(np.abs(finite - med))
        if mad < 1e-12:
            mad = np.std(finite, ddof=1) * 0.6745   # fallback to scaled σ
        mod_z = 0.6745 * np.abs(arr - med) / (mad + 1e-12)
        mask  = mod_z > mad_thresh

    # Never flag NaN itself as an outlier — it is already missing
    mask[~np.isfinite(arr)] = False
    return mask


def remove_trial_outliers(tdf,
                           method     = OUTLIER_METHOD,
                           sd_thresh  = OUTLIER_SD_THRESHOLD,
                           iqr_mult   = OUTLIER_IQR_MULTIPLIER,
                           mad_thresh = OUTLIER_MAD_THRESHOLD,
                           min_mt     = OUTLIER_MIN_MT_MS,
                           max_mt     = OUTLIER_MAX_MT_MS,
                           out_dir    = None):
    """
    Remove outlier trials from the trial-level DataFrame in two stages.

    Stage 1 — Absolute bounds
        Discard any trial whose movement time falls outside [min_mt, max_mt].
        This catches anticipatory button presses (< 50 ms) and abandoned
        trials / inattention pauses (> 30 000 ms) that would otherwise inflate
        variance and distort Fitts' Law regression slopes.

    Stage 2 — Statistical outliers per participant × condition cell
        Within every (participant, technique, movement, clustering) group,
        flag trials whose MT deviates from the group distribution by more than
        the chosen threshold (see _outlier_mask_1d).  Grouping per cell means
        each participant's own baseline is used, preventing the removal of
        legitimately slow participants.

    Returns
    -------
    cleaned_tdf : pd.DataFrame — trial rows with outliers removed
    report_df   : pd.DataFrame — summary of removal counts by stage / reason
    """
    if method == "none" and (min_mt is None) and (max_mt is None):
        print("  Trial outlier removal: DISABLED")
        return tdf.copy(), pd.DataFrame()

    original_n  = len(tdf)
    df          = tdf.copy()
    df["_out"]  = False
    df["_why"]  = ""
    report_rows = []

    # ── Stage 1: absolute bounds ──────────────────────────────────────────────
    if min_mt is not None:
        mask = df["time_ms"].notna() & (df["time_ms"] < min_mt)
        n    = int(mask.sum())
        if n:
            df.loc[mask, "_out"] = True
            df.loc[mask, "_why"] = f"MT < {min_mt:.0f} ms"
            report_rows.append({"stage": "absolute", "criterion": f"MT < {min_mt:.0f} ms",
                                 "n_removed": n})

    if max_mt is not None:
        mask = df["time_ms"].notna() & (df["time_ms"] > max_mt) & (~df["_out"])
        n    = int(mask.sum())
        if n:
            df.loc[mask, "_out"] = True
            df.loc[mask, "_why"] = f"MT > {max_mt:.0f} ms"
            report_rows.append({"stage": "absolute", "criterion": f"MT > {max_mt:.0f} ms",
                                 "n_removed": n})

    # ── Stage 2: statistical per participant × condition ──────────────────────
    if method != "none":
        group_cols = [c for c in ["participant", "technique", "movement", "clustering"]
                      if c in df.columns]
        stat_count = 0
        for _, grp in df[~df["_out"]].groupby(group_cols):
            mask = _outlier_mask_1d(grp["time_ms"].values, method,
                                    sd_thresh, iqr_mult, mad_thresh)
            if mask.any():
                idx = grp.index[mask]
                df.loc[idx, "_out"] = True
                df.loc[idx, "_why"] = f"statistical_{method}"
                stat_count += int(mask.sum())
        if stat_count:
            report_rows.append({"stage": "statistical",
                                 "criterion": f"method={method}",
                                 "n_removed": stat_count})

    # ── Split and export ──────────────────────────────────────────────────────
    removed = df[df["_out"]].copy()
    cleaned = df[~df["_out"]].drop(columns=["_out", "_why"])
    n_removed = len(removed)
    pct       = 100 * n_removed / max(original_n, 1)

    report_df = pd.DataFrame(report_rows) if report_rows else pd.DataFrame(
        columns=["stage", "criterion", "n_removed"])
    report_df["pct_of_total"] = (report_df["n_removed"] / original_n * 100).round(2)

    if out_dir is not None:
        removed.drop(columns=["_out"], errors="ignore")\
               .to_csv(out_dir / "outliers_removed_trials.csv", index=False)
        report_df.to_csv(out_dir / "outliers_report_trials.csv", index=False)
        # Per-condition tally
        if not removed.empty:
            group_cols = [c for c in ["participant", "technique", "movement", "clustering"]
                          if c in removed.columns]
            if group_cols:
                removed.groupby(group_cols).size()\
                       .reset_index(name="n_removed_trials")\
                       .to_csv(out_dir / "outliers_per_condition_trials.csv", index=False)

    return cleaned, report_df


def remove_condition_outliers(cdf,
                               dvs        = None,
                               method     = OUTLIER_METHOD,
                               sd_thresh  = OUTLIER_SD_THRESHOLD,
                               iqr_mult   = OUTLIER_IQR_MULTIPLIER,
                               mad_thresh = OUTLIER_MAD_THRESHOLD,
                               out_dir    = None):
    """
    Flag outlier aggregate values in the condition-level DataFrame.

    Rather than dropping entire rows (which would break the balanced
    repeated-measures ANOVA structure), individual DV values that are
    statistical outliers *within their condition cell* are set to NaN.
    The RM-ANOVA engine then handles missing cells as incomplete blocks.

    Outlier detection is per (technique × movement × clustering) cell,
    so only participants whose value is extreme relative to the other
    participants *in the same cell* are affected.

    Returns
    -------
    cleaned_cdf : pd.DataFrame — cdf with outlier DV cells set to NaN
    report_df   : pd.DataFrame — per-DV count of NaN'd values
    """
    if method == "none":
        print("  Condition-level outlier NaN-ing: DISABLED")
        return cdf.copy(), pd.DataFrame()

    if dvs is None:
        dvs = list(PRIMARY_DVS.keys())

    df         = cdf.copy()
    cell_cols  = [c for c in ["technique", "movement", "clustering"] if c in df.columns]
    report_rows = []
    total_nulled = 0

    for dv in dvs:
        if dv not in df.columns:
            continue
        dv_nulled = 0
        for _, grp in df.groupby(cell_cols):
            mask = _outlier_mask_1d(grp[dv].values, method,
                                    sd_thresh, iqr_mult, mad_thresh)
            if mask.any():
                idx = grp.index[mask]
                df.loc[idx, dv] = np.nan
                dv_nulled   += int(mask.sum())
                total_nulled += int(mask.sum())
        if dv_nulled:
            report_rows.append({
                "dv":            dv,
                "dv_label":      PRIMARY_DVS.get(dv, dv),
                "n_values_nulled": dv_nulled,
                "method":        method,
            })

    report_df = pd.DataFrame(report_rows) if report_rows else pd.DataFrame(
        columns=["dv", "dv_label", "n_values_nulled", "method"])

    if out_dir is not None and not report_df.empty:
        report_df.to_csv(out_dir / "outliers_report_conditions.csv", index=False)

    return df, report_df


def console_outlier_summary(trial_report, cond_report,
                             n_trials_orig, n_trials_clean,
                             method):
    """Print a concise outlier removal summary to stdout."""
    W = 70
    print(f"\n{'─'*W}")
    print(f"  OUTLIER REMOVAL SUMMARY  (method = {method!r})")
    print(f"{'─'*W}")

    # ── Trial level ───────────────────────────────────────────────────────────
    n_removed = n_trials_orig - n_trials_clean
    pct       = 100 * n_removed / max(n_trials_orig, 1)
    print(f"\n  Trial-level removal: {n_removed} / {n_trials_orig} ({pct:.2f}%)\n")
    if not trial_report.empty:
        for _, row in trial_report.iterrows():
            print(f"    {row['stage']:12s}  {row['criterion']:<35s}  "
                  f"n={row['n_removed']}  ({row['pct_of_total']:.1f}%)")
    else:
        print("    No trials removed.")

    # ── Condition level ───────────────────────────────────────────────────────
    print(f"\n  Condition-level NaN'd DV values:")
    if not cond_report.empty:
        for _, row in cond_report.iterrows():
            print(f"    {row['dv_label']:<45s}  {row['n_values_nulled']} values")
        print(f"    Total: {cond_report['n_values_nulled'].sum()}")
    else:
        print("    No condition values NaN'd.")

    print(f"{'─'*W}")


def fig00_outlier_removal(tdf_orig, tdf_clean, cdf_orig, cdf_clean,
                           trial_report, method, out_dir):
    """
    Figure 0: Outlier removal diagnostic.

    Panel A — MT distribution before vs after trial-level removal.
    Panel B — % trials removed per condition cell.
    Panel C — TP Shannon Nominal distribution before/after condition NaN-ing.
    Panel D — Per-participant trial removal rate.
    Panel E — Removal reason breakdown (bar chart).
    Panel F — MT by technique: box before/after overlay.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Outlier Removal Diagnostic  ·  method = {method!r}  ·  "
        f"{len(tdf_orig) - len(tdf_clean)} trials removed  "
        f"({100*(len(tdf_orig)-len(tdf_clean))/max(len(tdf_orig),1):.1f}%)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    # ── A: MT histogram before / after ───────────────────────────────────────
    ax = axes[0][0]
    mt_b = tdf_orig["time_ms"].dropna().values
    mt_a = tdf_clean["time_ms"].dropna().values
    bins = np.linspace(0, min(mt_b.max(), OUTLIER_MAX_MT_MS * 1.1), 60)
    ax.hist(mt_b, bins=bins, alpha=0.45, color="#e74c3c",
            label=f"Before  n={len(mt_b):,}")
    ax.hist(mt_a, bins=bins, alpha=0.60, color="#2ecc71",
            label=f"After   n={len(mt_a):,}")
    ax.axvline(OUTLIER_MIN_MT_MS, color="black", linewidth=1.2,
               linestyle=":", label=f"min={OUTLIER_MIN_MT_MS:.0f} ms")
    ax.set_xlabel("Movement Time MT (ms)", fontsize=9)
    ax.set_ylabel("Trial count", fontsize=9)
    ax.set_title("MT Distribution: Before vs After", fontsize=9, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── B: per-condition removal rate bar chart ───────────────────────────────
    ax = axes[0][1]
    gcols = [c for c in ["technique", "movement", "clustering"] if c in tdf_orig.columns]
    if gcols:
        nb = tdf_orig.groupby(gcols).size().reset_index(name="nb")
        na = tdf_clean.groupby(gcols).size().reset_index(name="na")
        mg = nb.merge(na, on=gcols, how="left").fillna(0)
        mg["na"] = mg["na"].clip(upper=mg["nb"])
        mg["pct_out"] = 100 * (1 - mg["na"] / mg["nb"].replace(0, np.nan))
        mg["label"]   = mg[gcols].apply(
            lambda r: "/".join(str(v)[:3] for v in r), axis=1)
        mg = mg.sort_values("pct_out", ascending=False)
        x  = np.arange(len(mg))
        bars = ax.bar(x, mg["pct_out"].fillna(0),
                      color="#e67e22", alpha=0.78, edgecolor="black", linewidth=0.4)
        step = max(1, len(x) // 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(mg["label"].iloc[::step].values,
                           rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("% trials removed", fontsize=9)
        ax.set_title("Removal Rate per Condition Cell\n(sorted descending)", fontsize=9,
                     fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # ── C: TP Shannon Nominal before/after condition NaN-ing ─────────────────
    ax = axes[0][2]
    dv = "throughput_shannon_nominal_bps"
    if dv in cdf_orig.columns and dv in cdf_clean.columns:
        bv = cdf_orig[dv].dropna().values
        av = cdf_clean[dv].dropna().values
        lo = min(bv.min(), av.min()); hi = max(bv.max(), av.max())
        bins2 = np.linspace(lo, hi, 25)
        ax.hist(bv, bins=bins2, alpha=0.45, color="#e74c3c",
                label=f"Before  n={len(bv)}")
        ax.hist(av, bins=bins2, alpha=0.60, color="#2ecc71",
                label=f"After   n={len(av)}")
        ax.set_xlabel("TP Shannon Nominal (bits/s)", fontsize=9)
        ax.set_ylabel("Condition count", fontsize=9)
        ax.set_title("Condition-Level TP: Before vs After\n(NaN-ed outlier cells excluded)",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── D: per-participant removal count ─────────────────────────────────────
    ax = axes[1][0]
    if "participant" in tdf_orig.columns:
        removed_idx = tdf_orig.index.difference(tdf_clean.index)
        removed_df  = tdf_orig.loc[removed_idx]
        if not removed_df.empty:
            per_p   = removed_df.groupby("participant").size().reset_index(name="n_out")
            tot_p   = tdf_orig.groupby("participant").size().reset_index(name="n_tot")
            per_p   = per_p.merge(tot_p, on="participant")
            per_p["pct"] = 100 * per_p["n_out"] / per_p["n_tot"]
            x = np.arange(len(per_p))
            ax.bar(x, per_p["pct"],
                   color="#9b59b6", alpha=0.78, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"P{int(p)}" for p in per_p["participant"]], fontsize=9)
            ax.set_ylabel("% trials removed", fontsize=9)
            ax.set_title("Trial Removal Rate per Participant", fontsize=9, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No trials removed", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray")
            ax.set_title("Trial Removal Rate per Participant", fontsize=9, fontweight="bold")

    # ── E: removal reason breakdown ───────────────────────────────────────────
    ax = axes[1][1]
    if not trial_report.empty:
        labels_e = trial_report["criterion"].values
        counts_e = trial_report["n_removed"].values
        colors_e = ["#c0392b", "#e67e22", "#3498db", "#27ae60"][:len(labels_e)]
        ax.barh(range(len(labels_e)), counts_e, color=colors_e,
                edgecolor="black", linewidth=0.5, alpha=0.82)
        ax.set_yticks(range(len(labels_e)))
        ax.set_yticklabels(labels_e, fontsize=8)
        ax.set_xlabel("Trials removed", fontsize=9)
        ax.set_title("Removal Reasons", fontsize=9, fontweight="bold")
        for i, v in enumerate(counts_e):
            ax.text(v + 0.3, i, str(v), va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No trials removed", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="gray")
        ax.set_title("Removal Reasons", fontsize=9, fontweight="bold")

    # ── F: MT box plots by technique — before / after overlay ────────────────
    ax = axes[1][2]
    x  = np.arange(len(TECHNIQUES))
    w  = 0.3
    for offset, df_src, label, col, pat in [
        (-w/2, tdf_orig,  "Before", "#e74c3c", "///"),
        ( w/2, tdf_clean, "After",  "#2ecc71", ""),
    ]:
        data = [df_src[df_src["technique"] == t]["time_ms"].dropna().values
                for t in TECHNIQUES]
        bp = ax.boxplot(data,
                        positions=x + offset,
                        widths=w * 0.85,
                        patch_artist=True,
                        showfliers=True,
                        flierprops=dict(marker=".", markersize=2.5, alpha=0.3),
                        medianprops=dict(color="black", linewidth=1.5),
                        manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
            patch.set_hatch(pat)
        bp["boxes"][0].set_label(label)   # for legend

    ax.set_xticks(x)
    ax.set_xticklabels(TECHNIQUES, fontsize=9)
    ax.set_ylabel("MT (ms)", fontsize=9)
    ax.set_title("MT by Technique: Before vs After\n(box = IQR, line = median)",
                 fontsize=9, fontweight="bold")
    handles = [mpatches.Patch(facecolor="#e74c3c", alpha=0.55, hatch="///", label="Before"),
               mpatches.Patch(facecolor="#2ecc71", alpha=0.55, label="After")]
    ax.legend(handles=handles, fontsize=8, framealpha=0.6)
    ax.grid(axis="y", alpha=0.3)

    savefig(fig, out_dir / "fig00_outlier_removal.png", dpi=140)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_session(path):
    """
    Parse one P*_session.json.
    Returns (cond_rows: list[dict], trial_rows: list[dict]).
    """
    with open(path, encoding="utf-8") as fh:
        session = json.load(fh)

    pid  = session.get("participant", -1)
    cond_rows  = []
    trial_rows = []

    for cond in session.get("conditions", []):
        cs = cond.get("conditionStats") or {}

        row = {
            "participant": int(pid),
            "technique":   cond.get("technique",  "?"),
            "movement":    cond.get("movement",   "?"),
            "clustering":  cond.get("clustering", "?"),
            "techBlock":   cond.get("techBlock"),
            "condInBlock": cond.get("condInBlock"),
        }

        for key in list(PRIMARY_DVS) + list(SECONDARY_DVS):
            row[key] = safe_float(cs.get(key))

        for model in ("fitts_regression", "hoffmann_regression"):
            reg = cs.get(model) or {}
            row[f"{model}_r2"]        = safe_float(reg.get("r2"))
            row[f"{model}_slope"]     = safe_float(reg.get("slope_ms_per_bit"))
            row[f"{model}_intercept"] = safe_float(reg.get("intercept_ms"))
            row[f"{model}_aic"]       = safe_float(reg.get("aic"))
            row[f"{model}_tp"]        = safe_float(reg.get("throughput_from_regression_bps"))
            row[f"{model}_n"]         = reg.get("n")

        row["delta_r2"]  = safe_float(cs.get("delta_r2_hoffmann_vs_shannon"))
        row["delta_aic"] = safe_float(cs.get("delta_aic_hoffmann_vs_shannon"))

        cond_rows.append(row)

        for t in cond.get("trials", []):
            trial_rows.append({
                "participant":  int(pid),
                "technique":    cond.get("technique"),
                "movement":     cond.get("movement"),
                "clustering":   cond.get("clustering"),
                "trial":        t.get("trialNumber"),
                "time_ms":      safe_float(t.get("time_ms")),
                "errorClicks":  safe_float(t.get("errorClickCount", 0)),
                "amplitude_px": safe_float(t.get("amplitude_px")),
                "targetRadius": safe_float(t.get("targetRadius")),
                "clickInside":  bool(t.get("clickedInsideTarget", False)),
                "distNorm":     safe_float(t.get("normalizedDistance")),
                "distPx":       safe_float(t.get("distanceToCenter_px")),
                "targetSpeed":  safe_float(t.get("targetSpeed_px_per_s")),
                "targetVx":     safe_float(t.get("targetVx_px_per_s")),
                "targetVy":     safe_float(t.get("targetVy_px_per_s")),
                "clickX":       safe_float(t.get("clickX")),
                "clickY":       safe_float(t.get("clickY")),
                "targetX":      safe_float(t.get("targetX")),
                "targetY":      safe_float(t.get("targetY")),
                "prevTargetX":  safe_float(t.get("prevTargetX")),
                "prevTargetY":  safe_float(t.get("prevTargetY")),
            })

    return cond_rows, trial_rows


def load_directory(data_dir):
    """Load all session JSONs; return (cdf, tdf, paths)."""
    data_dir = Path(data_dir)
    paths    = sorted(data_dir.glob("P*_session.json"))
    if not paths:
        paths = sorted(data_dir.glob("*session*.json"))
    if not paths:
        sys.exit(f"[ERROR] No session JSON files found in {data_dir}")

    all_cond, all_trial = [], []
    for p in paths:
        cr, tr = load_session(p)
        all_cond.extend(cr)
        all_trial.extend(tr)
        print(f"  ✓  {p.name}  ({len(cr)} conditions, {len(tr)} trials)")

    cdf = pd.DataFrame(all_cond)
    tdf = pd.DataFrame(all_trial)

    for df in (cdf, tdf):
        if "technique" in df.columns:
            df["technique"] = pd.Categorical(df["technique"], TECHNIQUES)
        if "movement"  in df.columns:
            df["movement"]  = pd.Categorical(df["movement"],  MOVEMENTS)
        if "clustering" in df.columns:
            df["clustering"] = pd.Categorical(df["clustering"], DENSITIES)

    return cdf, tdf, paths


# ─────────────────────────────────────────────────────────────────────────────
# DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def descriptive_stats(cdf, out_dir):
    dvs = list(PRIMARY_DVS)
    grand = cdf[dvs].describe().T.round(3)
    grand.to_csv(out_dir / "desc_grand.csv")

    for factor in ("technique", "movement", "clustering"):
        grp = cdf.groupby(factor)[dvs].agg(["mean", "std", "median", "sem"]).round(3)
        grp.to_csv(out_dir / f"desc_by_{factor}.csv")

    for f1, f2 in [("technique", "movement"),
                   ("technique", "clustering"),
                   ("movement",  "clustering")]:
        grp = cdf.groupby([f1, f2])[dvs].mean().round(3)
        grp.to_csv(out_dir / f"desc_{f1}_x_{f2}.csv")

    cdf.groupby(["technique", "movement", "clustering"])[dvs].mean().round(3)\
       .to_csv(out_dir / "desc_3way_cell_means.csv")

    print("  Descriptive CSVs written.")
    return grand


# ─────────────────────────────────────────────────────────────────────────────
# NORMALITY TESTING
# ─────────────────────────────────────────────────────────────────────────────

def normality_tests(cdf, out_dir):
    rows = []
    for tech, mov, dens in product(TECHNIQUES, MOVEMENTS, DENSITIES):
        cell = cdf[
            (cdf.technique == tech) &
            (cdf.movement  == mov)  &
            (cdf.clustering == dens)
        ]
        for dv in PRIMARY_DVS:
            vals = cell[dv].dropna().values
            if len(vals) >= 3:
                W, p = stats.shapiro(vals)
                rows.append({
                    "technique": tech, "movement": mov, "clustering": dens,
                    "dv": dv, "n": len(vals),
                    "W": round(W, 4), "p_shapiro": round(p, 4),
                    "normal_at_05": p >= ALPHA,
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "normality_shapiro_wilk.csv", index=False)

    print("\n  Shapiro-Wilk — fraction of cells passing normality (p ≥ .05):")
    for dv in PRIMARY_DVS:
        sub = df[df.dv == dv]
        pct = 100 * sub["normal_at_05"].mean()
        print(f"    {dv:<45s} {sub.normal_at_05.sum():2d}/{len(sub)} ({pct:.0f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3-WAY REPEATED-MEASURES ANOVA
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_complete_participants(df, dv):
    """Keep only participants with all 27 within-subjects cells filled."""
    expected = len(TECHNIQUES) * len(MOVEMENTS) * len(DENSITIES)
    counts   = df.groupby("participant").size()
    good     = counts[counts == expected].index
    n_dropped = df["participant"].nunique() - len(good)
    return df[df["participant"].isin(good)].copy(), n_dropped


def rm_anova_3way(cdf, dv):
    """
    3-way fully within-subjects RM ANOVA via statsmodels AnovaRM.
    GG epsilon computed per main-effect factor; interaction epsilons use
    the Box (1954) product approximation.  Corrected p-values are derived
    from scipy.stats.f with GG-adjusted degrees of freedom.
    """
    WITHIN = ["technique", "movement", "clustering"]

    df = cdf[["participant"] + WITHIN + [dv]].dropna().copy()
    df, n_dropped = _ensure_complete_participants(df, dv)

    n = df["participant"].nunique()
    if n < 3:
        return None, f"insufficient complete participants (n={n})"

    for col in WITHIN:
        df[col] = df[col].astype(str)

    try:
        res = AnovaRM(
            data    = df,
            depvar  = dv,
            subject = "participant",
            within  = WITHIN,
        ).fit()
        aov = res.anova_table.reset_index()
        aov.columns = ["Source", "F", "ddof1", "ddof2", "p-unc"]
    except Exception as exc:
        return None, str(exc)

    _src_map = {
        "technique":                     "technique",
        "movement":                      "movement",
        "clustering":                    "clustering",
        "technique:movement":            "technique * movement",
        "technique:clustering":          "technique * clustering",
        "movement:clustering":           "movement * clustering",
        "technique:movement:clustering": "technique * movement * clustering",
    }
    aov["Source"] = aov["Source"].str.strip().map(
        lambda s: _src_map.get(s, s)
    )

    aov["np2"] = (aov["F"] * aov["ddof1"]) / \
                 (aov["F"] * aov["ddof1"] + aov["ddof2"])

    eps_main = {f: _gg_epsilon_for_factor(df, dv, "participant", f)
                for f in WITHIN}

    def _combined_eps(source_name: str) -> float:
        parts = [p.strip() for p in source_name.split("*")]
        eps   = float(np.prod([eps_main.get(p, 1.0) for p in parts]))
        return min(max(eps, 0.0), 1.0)

    eps_col, p_gg_col = [], []
    for _, row in aov.iterrows():
        src      = row["Source"]
        eps      = _combined_eps(src) if src in _src_map.values() else 1.0
        df1_adj  = max(eps * row["ddof1"], 1e-6)
        df2_adj  = max(eps * row["ddof2"], 1e-6)
        F_val    = safe_float(row["F"])
        p_gg     = (float(stats.f.sf(F_val, df1_adj, df2_adj))
                    if np.isfinite(F_val) and F_val >= 0
                    else safe_float(row["p-unc"]))
        eps_col.append(round(eps, 4))
        p_gg_col.append(round(float(p_gg), 4))

    aov["eps"]       = eps_col
    aov["p-GG-corr"] = p_gg_col
    aov.insert(0, "dv", dv)

    return aov.round(4), None


def run_all_anovas(cdf, out_dir):
    results = {}
    print()
    for dv, label in PRIMARY_DVS.items():
        aov, err = rm_anova_3way(cdf, dv)
        if aov is not None:
            results[dv] = aov
            aov.to_csv(out_dir / f"anova_{dv}.csv", index=False)
            print(f"  ✓  ANOVA: {label}")
        else:
            print(f"  ✗  ANOVA: {label}  [{err}]")

    if results:
        pd.concat(list(results.values())).to_csv(out_dir / "anova_all_dvs.csv", index=False)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SPHERICITY SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def sphericity_summary(anova_results, out_dir):
    rows = []
    for dv, aov in anova_results.items():
        if aov is None:
            continue
        for _, row in aov.iterrows():
            eps = row.get("eps")
            if eps is not None and not np.isnan(safe_float(eps)):
                rows.append({
                    "dv":     dv,
                    "source": row["Source"],
                    "eps":    round(safe_float(eps), 4),
                    "sphericity_violated": safe_float(eps) < 0.9,
                    "correction_applied":  safe_float(eps) < 1.0,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_dir / "sphericity_epsilon.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# EFFECT SIZES
# ─────────────────────────────────────────────────────────────────────────────

def extract_effect_sizes(anova_results, out_dir):
    rows = []
    for dv, aov in anova_results.items():
        if aov is None:
            continue
        p_col = "p-GG-corr" if "p-GG-corr" in aov.columns else "p-unc"
        for _, row in aov.iterrows():
            eta2 = row.get("np2") or row.get("ng2")
            pval = safe_float(row.get(p_col))
            rows.append({
                "dv":       dv,
                "dv_label": PRIMARY_DVS.get(dv, dv)[:35],
                "source":   row["Source"],
                "partial_eta2": safe_float(eta2),
                "p_GG_corr":    pval,
                "sig":      pval < ALPHA,
                "effect_size_label":
                    "large"  if safe_float(eta2) >= 0.14 else
                    "medium" if safe_float(eta2) >= 0.06 else
                    "small"  if safe_float(eta2) >= 0.01 else "negligible",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_dir / "effect_sizes_partial_eta2.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# POST-HOC TESTS (Bonferroni-corrected pairwise)
# ─────────────────────────────────────────────────────────────────────────────

def _significant_main_effects(aov):
    if aov is None:
        return []
    p_col = "p-GG-corr" if "p-GG-corr" in aov.columns else "p-unc"
    return aov[aov[p_col] < ALPHA]["Source"].tolist()


def posthoc_for_dv(cdf, dv, aov, out_dir):
    results = {}
    df = cdf[["participant", "technique", "movement", "clustering", dv]].dropna()
    df, _ = _ensure_complete_participants(df, dv)
    if df["participant"].nunique() < 3:
        return results

    sig_effects = _significant_main_effects(aov)
    single_factors = {"technique": "technique", "movement": "movement", "clustering": "clustering"}

    for effect in sig_effects:
        if effect in single_factors:
            factor = single_factors[effect]
            try:
                ph = pg.pairwise_tests(
                    data=df, dv=dv, within=factor,
                    subject="participant", padjust="bonf", parametric=True,
                )
                ph.insert(0, "dv", dv); ph.insert(1, "effect", effect)
                results[effect] = ph
                ph.to_csv(out_dir / f"posthoc_{dv}_{factor}.csv", index=False)
            except Exception:
                pass
        elif " * " in effect:
            parts = [p.strip() for p in effect.split("*")]
            if len(parts) == 2 and all(p in single_factors for p in parts):
                try:
                    ph = pg.pairwise_tests(
                        data=df, dv=dv, within=parts,
                        subject="participant", padjust="bonf", parametric=True,
                    )
                    ph.insert(0, "dv", dv); ph.insert(1, "effect", effect)
                    safe_name = effect.replace(" * ", "_x_")
                    results[effect] = ph
                    ph.to_csv(out_dir / f"posthoc_{dv}_{safe_name}.csv", index=False)
                except Exception:
                    pass

    return results


def run_all_posthoc(cdf, anova_results, out_dir):
    posthoc = {}
    print()
    for dv in anova_results:
        posthoc[dv] = posthoc_for_dv(cdf, dv, anova_results[dv], out_dir)
        n = sum(len(v) for v in posthoc[dv].values())
        print(f"  Post-hoc: {PRIMARY_DVS.get(dv, dv)[:45]}  [{n} comparisons]")
    return posthoc


# ─────────────────────────────────────────────────────────────────────────────
# FITTS' LAW REGRESSION  (trial-level)
# ─────────────────────────────────────────────────────────────────────────────

def fitts_regression(tdf, out_dir):
    elig = tdf.dropna(subset=["amplitude_px", "time_ms", "targetRadius"]).copy()
    elig = elig[(elig.amplitude_px > 0) & (elig.time_ms > 0) & (elig.targetRadius > 0)]
    elig = elig[elig.trial.fillna(0) > 1]

    elig["ID"]   = np.log2(elig.amplitude_px / (2 * elig.targetRadius) + 1)
    elig["MT_s"] = elig.time_ms / 1000.0

    dx = elig.targetX - elig.prevTargetX.fillna(elig.targetX)
    dy = elig.targetY - elig.prevTargetY.fillna(elig.targetY)
    dist_moved = np.maximum(np.sqrt(dx**2 + dy**2), 1)
    vt_proj = (elig.targetVx.fillna(0) * dx / dist_moved +
               elig.targetVy.fillna(0) * dy / dist_moved)
    elig["vt"] = np.abs(vt_proj)
    W_nom = 2 * elig.targetRadius
    W_eff = np.maximum(W_nom - elig.vt * elig.MT_s, W_nom * 0.1)
    elig["ID_hoffmann"] = np.log2(elig.amplitude_px / W_eff + 1)

    results_tech, results_tm = {}, {}

    for tech in TECHNIQUES:
        sub = elig[elig.technique == tech]
        if len(sub) < 5:
            continue
        for id_col, model_name in [("ID", "fitts"), ("ID_hoffmann", "hoffmann")]:
            x, y = sub[id_col].values, sub["time_ms"].values
            slope, intercept, r, p_r, se = stats.linregress(x, y)
            results_tech[f"{tech}_{model_name}"] = {
                "technique": tech, "model": model_name, "n": len(sub),
                "slope_ms": round(slope, 2), "intercept_ms": round(intercept, 2),
                "r": round(r, 4), "r2": round(r**2, 4), "p": round(p_r, 6),
                "tp_bps": round(1000 / slope, 3) if slope > 0 else np.nan,
                "se": round(se, 4),
            }

    pd.DataFrame(results_tech).T.to_csv(out_dir / "fitts_regression_by_technique.csv")

    for tech in TECHNIQUES:
        for mov in MOVEMENTS:
            sub = elig[(elig.technique == tech) & (elig.movement == mov)]
            if len(sub) < 5:
                continue
            x, y = sub["ID"].values, sub["time_ms"].values
            slope, intercept, r, p_r, se = stats.linregress(x, y)
            results_tm[f"{tech}_{mov}"] = {
                "technique": tech, "movement": mov, "n": len(sub),
                "slope_ms": round(slope, 2), "intercept_ms": round(intercept, 2),
                "r2": round(r**2, 4),
                "tp_bps": round(1000 / slope, 3) if slope > 0 else np.nan,
            }

    pd.DataFrame(results_tm).T.to_csv(out_dir / "fitts_regression_by_tech_movement.csv")
    return elig, results_tech, results_tm


# ─────────────────────────────────────────────────────────────────────────────
# THROUGHPUT CONCORDANCE
# ─────────────────────────────────────────────────────────────────────────────

def throughput_concordance(cdf, out_dir):
    tp_cols = [
        "throughput_shannon_nominal_bps",
        "throughput_shannon_effective_bps",
        "throughput_hoffmann_bps",
    ]
    labels = ["TP-Shannon-Nominal", "TP-Shannon-Effective", "TP-Hoffmann"]
    sub = cdf[tp_cols].dropna()
    sub.corr().round(4).to_csv(out_dir / "throughput_pearson_matrix.csv")

    pair_rows = []
    for (c1, l1), (c2, l2) in combinations(zip(tp_cols, labels), 2):
        x, y = sub[c1].values, sub[c2].values
        r, p_r = stats.pearsonr(x, y)
        diff   = x - y
        pair_rows.append({
            "pair":    f"{l1} vs {l2}",
            "pearson_r": round(r, 4), "p_pearson": round(p_r, 6),
            "mean_diff": round(np.mean(diff), 4), "sd_diff": round(np.std(diff, ddof=1), 4),
            "loa_lower": round(np.mean(diff) - 1.96 * np.std(diff, ddof=1), 4),
            "loa_upper": round(np.mean(diff) + 1.96 * np.std(diff, ddof=1), 4),
        })
    pd.DataFrame(pair_rows).to_csv(
        out_dir / "throughput_concordance_bland_altman.csv", index=False)
    return pd.DataFrame(pair_rows)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def model_comparison_stats(cdf, out_dir):
    df = cdf[["fitts_regression_r2", "hoffmann_regression_r2",
              "fitts_regression_aic", "hoffmann_regression_aic",
              "delta_r2", "delta_aic",
              "technique", "movement", "clustering"]].dropna(
                  subset=["delta_r2", "delta_aic"])
    rows = []
    for var, col in [("Δr²", "delta_r2"), ("ΔAIC", "delta_aic")]:
        vals = df[col].dropna().values
        if len(vals) < 3:
            continue
        t_stat, t_p = stats.ttest_1samp(vals, 0.0)
        w_stat, w_p = stats.wilcoxon(vals)
        rows.append({
            "metric": var, "n_conditions": len(vals),
            "mean": round(np.mean(vals), 4), "sd": round(np.std(vals, ddof=1), 4),
            "median": round(np.median(vals), 4),
            "t_vs_0": round(t_stat, 3), "p_t": round(t_p, 4),
            "W_wilcoxon": round(w_stat, 3), "p_wilcoxon": round(w_p, 4),
            "sig_at_05": t_p < ALPHA,
        })
    pd.DataFrame(rows).to_csv(out_dir / "model_comparison_delta_stats.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# HYPOTHESIS TESTING
# ─────────────────────────────────────────────────────────────────────────────

def _technique_pairwise_in_subset(cdf, dv, filter_col=None, filter_val=None,
                                   _log_hyp="", _log_cond=""):
    sub = cdf.copy()
    if filter_col and filter_val:
        sub = sub[sub[filter_col] == filter_val]

    means = sub.groupby(["participant", "technique"])[dv].mean().reset_index()
    pivot = means.pivot(index="participant", columns="technique", values=dv)
    pivot.columns = pivot.columns.astype(str)

    results = {}
    for t1, t2 in combinations(TECHNIQUES, 2):
        if t1 not in pivot.columns or t2 not in pivot.columns:
            continue
        pair = pivot[[t1, t2]].dropna()
        if len(pair) < 2:
            continue
        n = len(pair)

        if n >= 3:
            t_stat, p = stats.ttest_rel(pair[t1].values, pair[t2].values)
            d = cohen_d(pair[t1].values, pair[t2].values)

            if _log_hyp:
                _log_t(_log_hyp, f"{t1}_vs_{t2}", "paired_t", dv, n,
                       t_stat, p, d=d,
                       mean_a=pair[t1].mean(), mean_b=pair[t2].mean(),
                       mean_diff=(pair[t1] - pair[t2]).mean(),
                       condition=_log_cond)
        else:
            t_stat = p = d = np.nan

        results[f"{t1}_vs_{t2}"] = {
            "n": n,
            "mean_A": round(pair[t1].mean(), 4),
            "mean_B": round(pair[t2].mean(), 4),
            "mean_diff": round((pair[t1] - pair[t2]).mean(), 4),
            "t": round(t_stat, 3) if np.isfinite(t_stat) else None,
            "p": round(p, 4) if np.isfinite(p) else None,
            "cohen_d": round(d, 3) if np.isfinite(d) else None,
            "sig": bool(np.isfinite(p) and p < ALPHA),
        }
    return results


def test_h1_cursor_technique(cdf, anova_results, out_dir):
    results = {}

    h1a_overall = {}
    for dv in ["throughput_shannon_nominal_bps",
               "throughput_shannon_effective_bps",
               "throughput_hoffmann_bps",
               "avgTime_ms",
               "precisionRate_percent",
               "avgErrorClicks"]:
        h1a_overall[dv] = _technique_pairwise_in_subset(
            cdf, dv, _log_hyp="H1a", _log_cond="all_conditions")
    results["H1a_overall_pairwise"] = h1a_overall

    dv_primary    = "throughput_shannon_nominal_bps"
    h1a_primary   = _technique_pairwise_in_subset(cdf, dv_primary)
    bub_vs_pt     = h1a_primary.get("BUBBLE_vs_POINT", {})
    area_entry    = (h1a_primary.get("AREA_vs_POINT") or
                     h1a_primary.get("POINT_vs_AREA") or {})
    results["H1a1_verdict"] = {
        "bubble_vs_point_mean_diff": bub_vs_pt.get("mean_diff"),
        "bubble_vs_point_p":         bub_vs_pt.get("p"),
        "bubble_vs_point_cohen_d":   bub_vs_pt.get("cohen_d"),
        "bubble_vs_point_sig":       bub_vs_pt.get("sig", False),
        "area_vs_point_p":           area_entry.get("p"),
        "area_vs_point_cohen_d":     area_entry.get("cohen_d"),
        "area_vs_point_sig":         area_entry.get("sig", False),
        "H1a1_supported": bool(bub_vs_pt.get("sig") or area_entry.get("sig")),
        "note": "H1a1: Bubble and/or Area significantly outperform Point on TP Shannon Nominal",
    }

    dv  = "throughput_shannon_nominal_bps"
    h1b = {}

    for dens in DENSITIES:
        pairs = _technique_pairwise_in_subset(
            cdf, dv, "clustering", dens,
            _log_hyp="H1b", _log_cond=f"clustering={dens}")
        cell  = {}

        ap_key = ("AREA_vs_POINT"  if "AREA_vs_POINT"  in pairs else
                  "POINT_vs_AREA"  if "POINT_vs_AREA"  in pairs else None)
        if ap_key:
            ap   = pairs[ap_key]
            sign = 1 if ap_key == "AREA_vs_POINT" else -1
            cell["area_minus_point"]      = round(sign * ap["mean_diff"], 4)
            cell["area_vs_point_t"]       = ap.get("t")
            cell["area_vs_point_p"]       = ap.get("p")
            cell["area_vs_point_cohen_d"] = ap.get("cohen_d")
            cell["area_vs_point_sig"]     = ap.get("sig")

        if "BUBBLE_vs_POINT" in pairs:
            bp = pairs["BUBBLE_vs_POINT"]
            cell["bubble_minus_point"]      = round(bp["mean_diff"], 4)
            cell["bubble_vs_point_t"]       = bp.get("t")
            cell["bubble_vs_point_p"]       = bp.get("p")
            cell["bubble_vs_point_cohen_d"] = bp.get("cohen_d")
            cell["bubble_vs_point_sig"]     = bp.get("sig")

        if "BUBBLE_vs_AREA" in pairs:
            ba = pairs["BUBBLE_vs_AREA"]
            cell["bubble_minus_area"]      = round(ba["mean_diff"], 4)
            cell["bubble_vs_area_t"]       = ba.get("t")
            cell["bubble_vs_area_p"]       = ba.get("p")
            cell["bubble_vs_area_sig"]     = ba.get("sig")

        h1b[dens] = cell

    def _adv_series(df_sub, dv, tech_a, tech_b):
        m   = df_sub.groupby(["participant", "technique"])[dv].mean().reset_index()
        piv = m.pivot(index="participant", columns="technique", values=dv)
        piv.columns = piv.columns.astype(str)
        if tech_a in piv.columns and tech_b in piv.columns:
            return (piv[tech_a] - piv[tech_b]).dropna()
        return pd.Series(dtype=float)

    for tech_a, tech_b, label in [("AREA",   "POINT", "area_vs_point"),
                                   ("BUBBLE", "POINT", "bubble_vs_point")]:
        adv_lo = _adv_series(cdf[cdf.clustering == "LOW"],  dv, tech_a, tech_b)
        adv_hi = _adv_series(cdf[cdf.clustering == "HIGH"], dv, tech_a, tech_b)
        common = adv_lo.index.intersection(adv_hi.index)
        if len(common) >= 3:
            t, p = stats.ttest_rel(adv_lo[common].values, adv_hi[common].values)
            d    = cohen_d(adv_lo[common].values, adv_hi[common].values)
            _log_t("H1b", f"attenuation_{label}", "paired_t", dv,          # ← ADD
                   len(common), t, p, d=d,                                  # ← ADD
                   mean_a=adv_lo[common].mean(), mean_b=adv_hi[common].mean(), # ← ADD
                   mean_diff=(adv_lo[common] - adv_hi[common]).mean(),      # ← ADD
                   condition="advantage_LOW_vs_HIGH")                       # ← ADD
            h1b[f"attenuation_{label}_LOW_vs_HIGH"] = {
                "mean_advantage_LOW":  round(adv_lo[common].mean(), 4),
                "mean_advantage_HIGH": round(adv_hi[common].mean(), 4),
                "t": round(t, 3), "p": round(p, 4), "cohen_d": round(d, 3),
                "sig": bool(p < ALPHA),
                "H1b1_attenuation_confirmed": bool(
                    p < ALPHA and adv_lo[common].mean() > adv_hi[common].mean()
                ),
            }
        else:
            h1b[f"attenuation_{label}_LOW_vs_HIGH"] = {
                "n_common": len(common),
                "note": "insufficient paired observations for t-test",
            }

    results["H1b_clustering_interaction"] = h1b

    dv  = "throughput_shannon_nominal_bps"
    h1c = {}

    for dens in DENSITIES:
        for mov in MOVEMENTS:
            sub   = cdf[(cdf.clustering == dens) & (cdf.movement == mov)]
            if sub.empty:
                continue
            means = sub.groupby(["participant", "technique"])[dv].mean().reset_index()
            pivot = means.pivot(index="participant", columns="technique", values=dv)
            pivot.columns = pivot.columns.astype(str)
            cell  = {}
            for t1, t2 in [("BUBBLE", "POINT"), ("BUBBLE", "AREA")]:
                if t1 not in pivot.columns or t2 not in pivot.columns:
                    continue
                pair = pivot[[t1, t2]].dropna()
                n    = len(pair)
                if n >= 3:
                    t_stat, p = stats.ttest_rel(pair[t1].values, pair[t2].values)
                    d         = cohen_d(pair[t1].values, pair[t2].values)
                    _log_t("H1c", f"{t1}_vs_{t2}", "paired_t", dv, n,      # ← ADD
                           t_stat, p, d=d,                                  # ← ADD
                           mean_a=pair[t1].mean(), mean_b=pair[t2].mean(), # ← ADD
                           mean_diff=(pair[t1] - pair[t2]).mean(),         # ← ADD
                           condition=f"clustering={dens}_movement={mov}")  # ← ADD
                else:
                    t_stat = p = d = np.nan
                cell[f"{t1}_vs_{t2}"] = {
                    "n": n,
                    "mean_diff": round((pair[t1] - pair[t2]).mean(), 4) if n > 0 else np.nan,
                    "t":        round(t_stat, 3) if np.isfinite(t_stat) else None,
                    "p":        round(p, 4)       if np.isfinite(p)      else None,
                    "cohen_d":  round(d, 3)        if np.isfinite(d)     else None,
                    "sig": bool(np.isfinite(p) and p < ALPHA),
                }
            if cell:
                h1c[f"{dens}_{mov}"] = cell

    bp_cell_vals = {
        key: d["BUBBLE_vs_POINT"]["mean_diff"]
        for key, d in h1c.items()
        if isinstance(d, dict) and "BUBBLE_vs_POINT" in d
        and np.isfinite(d["BUBBLE_vs_POINT"].get("mean_diff", np.nan))
    }
    if bp_cell_vals:
        fh_val     = bp_cell_vals.get("HIGH_FAST", np.nan)
        other_vals = [v for k, v in bp_cell_vals.items() if k != "HIGH_FAST"]
        h1c["_H1c1_summary"] = {
            "bubble_minus_point_HIGH_FAST": fh_val if np.isfinite(fh_val) else None,
            "mean_bubble_minus_point_other_cells": round(np.mean(other_vals), 4)
            if other_vals else None,
            "H1c1_diminishment_at_fast_high": bool(
                np.isfinite(fh_val) and other_vals and
                fh_val < np.mean(other_vals)
            ),
        }

    results["H1c_motion_moderation"] = h1c

    rows = []
    for hyp, top in results.items():
        for k1, v1 in top.items():
            if isinstance(v1, dict):
                for k2, v2 in v1.items():
                    if isinstance(v2, dict):
                        rows.append({"hypothesis": hyp, "level1": k1, "level2": k2, **v2})
                    else:
                        rows.append({"hypothesis": hyp, "level1": k1, "level2": k2,
                                     "value": v2})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "hypothesis_H1_results.csv", index=False)

    return results


def test_h2_fitts_law(cdf, elig, out_dir):
    results = {}

    h2a = {}
    if not elig.empty:
        static_elig = elig[elig.movement == "STATIC"]
        for tech in TECHNIQUES + ["ALL"]:
            sub = static_elig if tech == "ALL" else static_elig[static_elig.technique == tech]
            if len(sub) < 5:
                h2a[tech] = {"note": "insufficient data", "n": len(sub)}
                continue
            x, y   = sub["ID"].values, sub["time_ms"].values
            slope, intercept, r, p_r, _ = stats.linregress(x, y)
            h2a[tech] = {
                "n": len(sub),
                "r2": round(r**2, 4), "r": round(r, 4), "p": round(p_r, 6),
                "slope_ms_per_bit": round(slope, 2),
                "intercept_ms":     round(intercept, 2),
                "tp_bps": round(1000 / slope, 3) if slope > 0 else np.nan,
                "H2a1_supported": bool(r**2 > 0.3 and p_r < ALPHA),
            }

    static_cdf = cdf[cdf.movement == "STATIC"][["fitts_regression_r2"]].dropna()
    h2a["_condition_level_mean_r2_STATIC"] = (
        round(static_cdf["fitts_regression_r2"].mean(), 4)
        if not static_cdf.empty else np.nan
    )
    results["H2a_static_regression"] = h2a

    h2b      = {}
    slow_cdf = cdf[cdf.movement == "SLOW"].copy()

    s_r2 = slow_cdf["fitts_regression_r2"].dropna().values
    h_r2 = slow_cdf["hoffmann_regression_r2"].dropna().values
    dr2  = slow_cdf["delta_r2"].dropna().values
    daic = slow_cdf["delta_aic"].dropna().values

    both_slow = slow_cdf[["fitts_regression_r2", "hoffmann_regression_r2"]].dropna()
    if len(both_slow) >= 3:
        t_r2, p_r2 = stats.ttest_rel(both_slow["hoffmann_regression_r2"].values,
                                     both_slow["fitts_regression_r2"].values)
        d_r2 = cohen_d(both_slow["hoffmann_regression_r2"].values,
                       both_slow["fitts_regression_r2"].values)
        _log_t("H2b", "hoffmann_vs_fitts_r2", "paired_t",                  # ← ADD
               "hoffmann_r2 vs fitts_r2", len(both_slow), t_r2, p_r2,     # ← ADD
               d=d_r2,                                                      # ← ADD
               mean_a=both_slow["hoffmann_regression_r2"].mean(),          # ← ADD
               mean_b=both_slow["fitts_regression_r2"].mean(),             # ← ADD
               mean_diff=(both_slow["hoffmann_regression_r2"]              # ← ADD
                          - both_slow["fitts_regression_r2"]).mean(),      # ← ADD
               condition="movement=SLOW")                                   # ← ADD
    else:
        t_r2 = p_r2 = d_r2 = np.nan

    hoffmann_better = (
        bool(np.mean(h_r2) > np.mean(s_r2))
        if (len(h_r2) and len(s_r2)) else None
    )
    h2b0_supported = bool(
        np.isfinite(p_r2) and p_r2 < ALPHA and hoffmann_better
    ) if hoffmann_better is not None else False

    h2b["r2_comparison_SLOW"] = {
        "n_conditions":     len(both_slow),
        "mean_fitts_r2":    round(np.mean(s_r2), 4) if len(s_r2) else np.nan,
        "mean_hoffmann_r2": round(np.mean(h_r2), 4) if len(h_r2) else np.nan,
        "mean_delta_r2":    round(np.mean(dr2),  4) if len(dr2)  else np.nan,
        "t_paired_r2":  round(t_r2, 3) if np.isfinite(t_r2) else None,
        "p_paired_r2":  round(p_r2, 4) if np.isfinite(p_r2) else None,
        "cohen_d_r2":   round(d_r2, 3) if np.isfinite(d_r2) else None,
        "hoffmann_better_than_fitts": hoffmann_better,
        "H2b0_supported": h2b0_supported,
        "H2b1_supported": not h2b0_supported,
    }

    if len(daic) >= 3:
        t_daic, p_daic = stats.ttest_1samp(daic, 0)
        _log_t("H2b", "delta_aic_vs_zero", "one_sample_t",                 # ← ADD
               "delta_aic", len(daic), t_daic, p_daic,                    # ← ADD
               mean_a=np.mean(daic), mean_b=0, mean_diff=np.mean(daic),   # ← ADD
               condition="movement=SLOW")                                   # ← ADD
        h2b["aic_comparison_SLOW"] = {
            "mean_daic": round(np.mean(daic), 3),
            "t_vs_0":    round(t_daic, 3),
            "p_vs_0":    round(p_daic, 4),
            "sig":       bool(p_daic < ALPHA),
            "hoffmann_preferred_aic": bool(np.mean(daic) < 0),
            "H2b0_aic_supported": bool(p_daic < ALPHA and np.mean(daic) < 0),
        }

    results["H2b_hoffmann_vs_shannon_slow"] = h2b

    h2c      = {}
    fast_cdf = cdf[cdf.movement == "FAST"].copy()

    s_r2_f = fast_cdf["fitts_regression_r2"].dropna().values
    h_r2_f = fast_cdf["hoffmann_regression_r2"].dropna().values
    dr2_f  = fast_cdf["delta_r2"].dropna().values
    daic_f = fast_cdf["delta_aic"].dropna().values

    both_fast = fast_cdf[["fitts_regression_r2", "hoffmann_regression_r2"]].dropna()
    if len(both_fast) >= 3:
        t_r2_f, p_r2_f = stats.ttest_rel(both_fast["hoffmann_regression_r2"].values,
                                          both_fast["fitts_regression_r2"].values)
        d_r2_f = cohen_d(both_fast["hoffmann_regression_r2"].values,
                         both_fast["fitts_regression_r2"].values)
        _log_t("H2c", "hoffmann_vs_fitts_r2", "paired_t",                  # ← ADD
               "hoffmann_r2 vs fitts_r2", len(both_fast), t_r2_f, p_r2_f, # ← ADD
               d=d_r2_f,                                                    # ← ADD
               mean_a=both_fast["hoffmann_regression_r2"].mean(),          # ← ADD
               mean_b=both_fast["fitts_regression_r2"].mean(),             # ← ADD
               mean_diff=(both_fast["hoffmann_regression_r2"]              # ← ADD
                          - both_fast["fitts_regression_r2"]).mean(),      # ← ADD
               condition="movement=FAST")                                   # ← ADD
    else:
        t_r2_f = p_r2_f = d_r2_f = np.nan

    h2c["r2_comparison_FAST"] = {
        "n_conditions":     len(both_fast),
        "mean_fitts_r2":    round(np.mean(s_r2_f), 4) if len(s_r2_f) else np.nan,
        "mean_hoffmann_r2": round(np.mean(h_r2_f), 4) if len(h_r2_f) else np.nan,
        "mean_delta_r2":    round(np.mean(dr2_f),  4) if len(dr2_f)  else np.nan,
        "t_paired_r2":  round(t_r2_f, 3) if np.isfinite(t_r2_f) else None,
        "p_paired_r2":  round(p_r2_f, 4) if np.isfinite(p_r2_f) else None,
        "cohen_d_r2":   round(d_r2_f, 3) if np.isfinite(d_r2_f) else None,
        "H2c0_retained":  bool(not (np.isfinite(p_r2_f) and p_r2_f < ALPHA))
        if np.isfinite(p_r2_f) else None,
        "H2c1_supported": bool(np.isfinite(p_r2_f) and p_r2_f < ALPHA),
    }

    if len(daic_f) >= 3:
        t_daic_f, p_daic_f = stats.ttest_1samp(daic_f, 0)
        _log_t("H2c", "delta_aic_vs_zero", "one_sample_t",                 # ← ADD
               "delta_aic", len(daic_f), t_daic_f, p_daic_f,              # ← ADD
               mean_a=np.mean(daic_f), mean_b=0, mean_diff=np.mean(daic_f), # ← ADD
               condition="movement=FAST")                                   # ← ADD
        h2c["aic_comparison_FAST"] = {
            "mean_daic": round(np.mean(daic_f), 3),
            "t_vs_0":    round(t_daic_f, 3),
            "p_vs_0":    round(p_daic_f, 4),
            "H2c0_aic_retained": bool(p_daic_f >= ALPHA),
        }

    if len(daic) >= 3 and len(daic_f) >= 3:
        t_comp, p_comp = stats.ttest_ind(daic, daic_f)
        _log_t("H2c", "delta_aic_SLOW_vs_FAST", "independent_t",           # ← ADD
               "delta_aic", len(daic) + len(daic_f), t_comp, p_comp,      # ← ADD
               mean_a=np.mean(daic), mean_b=np.mean(daic_f),              # ← ADD
               mean_diff=np.mean(daic) - np.mean(daic_f),                 # ← ADD
               condition="SLOW_vs_FAST")                                    # ← ADD
        h2c["daic_SLOW_vs_FAST"] = {
            "mean_daic_SLOW": round(np.mean(daic), 3),
            "mean_daic_FAST": round(np.mean(daic_f), 3),
            "t": round(t_comp, 3), "p": round(p_comp, 4),
            "sig": bool(p_comp < ALPHA),
        }

    if len(s_r2) >= 3 and len(s_r2_f) >= 3:
        t_sf, p_sf = stats.ttest_ind(s_r2, s_r2_f)
        _log_t("H2c", "fitts_r2_SLOW_vs_FAST", "independent_t",            # ← ADD
               "fitts_regression_r2", len(s_r2) + len(s_r2_f), t_sf, p_sf, # ← ADD
               mean_a=np.mean(s_r2), mean_b=np.mean(s_r2_f),              # ← ADD
               mean_diff=np.mean(s_r2) - np.mean(s_r2_f),                 # ← ADD
               condition="SLOW_vs_FAST")                                    # ← ADD
        h2c["fitts_r2_SLOW_vs_FAST"] = {
            "mean_SLOW": round(np.mean(s_r2), 4),
            "mean_FAST": round(np.mean(s_r2_f), 4),
            "t": round(t_sf, 3), "p": round(p_sf, 4),
            "sig": bool(p_sf < ALPHA),
        }

    results["H2c_random_walk"] = h2c

    rows = []
    for hyp, top in results.items():
        for k, v in top.items():
            if isinstance(v, dict):
                rows.append({"hypothesis": hyp, "condition": k, **v})
            else:
                rows.append({"hypothesis": hyp, "condition": k, "value": v})
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "hypothesis_H2_results.csv", index=False)

    return results

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 16: HYPOTHESIS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def fig16_hypothesis_tests(cdf, h1_results, h2_results, elig, out_dir):
    fig = plt.figure(figsize=(20, 24))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    dv_tp    = "throughput_shannon_nominal_bps"
    label_tp = "TP Shannon Nominal (bits/s)"

    ax = fig.add_subplot(gs[0, 0])
    x  = np.arange(len(MOVEMENTS))
    w  = 0.25
    for k, tech in enumerate(TECHNIQUES):
        means = [cdf[(cdf.technique == tech) & (cdf.movement == m)][dv_tp].mean()
                 for m in MOVEMENTS]
        sems  = [cdf[(cdf.technique == tech) & (cdf.movement == m)][dv_tp].sem()
                 for m in MOVEMENTS]
        ax.bar(x + k * w, means, w, yerr=sems, label=tech,
               color=TECH_COLORS[tech], edgecolor="black", linewidth=0.6,
               alpha=0.8, capsize=3)
    ax.set_xticks(x + w); ax.set_xticklabels(MOVEMENTS, fontsize=9)
    ax.set_ylabel(label_tp, fontsize=8)
    ax.set_title("H1a: Technique × Motion\nBubble/Area vs Point?", fontsize=9,
                 fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.6); ax.grid(axis="y", alpha=0.3)

    ax   = fig.add_subplot(gs[0, 1])
    h1b  = h1_results.get("H1b_clustering_interaction", {})
    x    = np.arange(len(DENSITIES))
    w    = 0.32
    for offset, adv_key, label_line, col in [
        (-w/2, "area_minus_point",   "Area − Point",   TECH_COLORS["AREA"]),
        ( w/2, "bubble_minus_point", "Bubble − Point", TECH_COLORS["BUBBLE"]),
    ]:
        vals = [h1b.get(dens, {}).get(adv_key, np.nan) for dens in DENSITIES]
        bars = ax.bar(x + offset, vals, w, label=label_line,
                      color=col, edgecolor="black", linewidth=0.6, alpha=0.8)
        p_key = adv_key.replace("minus", "vs").replace("_point", "_point_p")
        for bi, (bar, dens) in enumerate(zip(bars, DENSITIES)):
            p_val = h1b.get(dens, {}).get(p_key)
            v     = vals[bi]
            if p_val is not None and np.isfinite(v):
                ypos = v + np.sign(v) * 0.01 * max(
                    (abs(h1b.get(d, {}).get(adv_key, 0) or 0) for d in DENSITIES), default=0.01
                )
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        sig_stars(p_val).strip() or "ns",
                        ha="center", fontsize=9, fontweight="bold",
                        color="red" if p_val < ALPHA else "gray")
    ax.axhline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(DENSITIES, fontsize=9)
    ax.set_ylabel("Advantage over Point (bits/s)", fontsize=8)
    ax.set_title("H1b: Area & Bubble Advantages\nby Clustering",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.6); ax.grid(axis="y", alpha=0.3)

    ax  = fig.add_subplot(gs[0, 2])
    h1c = h1_results.get("H1c_motion_moderation", {})
    mat = np.full((len(MOVEMENTS), len(DENSITIES)), np.nan)
    for di, dens in enumerate(DENSITIES):
        for mi, mov in enumerate(MOVEMENTS):
            bp = h1c.get(f"{dens}_{mov}", {}).get("BUBBLE_vs_POINT", {})
            if bp:
                mat[mi, di] = bp.get("mean_diff", np.nan)

    if not np.all(np.isnan(mat)):
        vext = np.nanmax(np.abs(mat))
        im   = ax.imshow(mat, cmap="RdYlGn", vmin=-vext, vmax=vext,
                         aspect="auto", interpolation="nearest")
        ax.set_xticks(range(len(DENSITIES)));  ax.set_xticklabels(DENSITIES, fontsize=8)
        ax.set_yticks(range(len(MOVEMENTS)));  ax.set_yticklabels(MOVEMENTS, fontsize=8)
        for mi, mov in enumerate(MOVEMENTS):
            for di, dens in enumerate(DENSITIES):
                v  = mat[mi, di]
                if np.isnan(v):
                    continue
                p  = h1c.get(f"{dens}_{mov}", {}).get("BUBBLE_vs_POINT", {}).get("p")
                star = sig_stars(p if p else 1.0).strip()
                txt_col = "white" if abs(v) > vext * 0.6 else "black"
                ax.text(di, mi, f"{v:.2f}\n{star}",
                        ha="center", va="center", fontsize=7, color=txt_col)
                if dens == "HIGH" and mov == "FAST":
                    ax.add_patch(plt.Rectangle(
                        (di - 0.5, mi - 0.5), 1, 1,
                        fill=False, edgecolor="black", linewidth=2.5, zorder=5
                    ))
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="bits/s")
    ax.set_title("H1c: Bubble − Point TP\n(★ HIGH×FAST = H1c1 key cell)", fontsize=9,
                 fontweight="bold")
    ax.set_xlabel("Clustering", fontsize=8); ax.set_ylabel("Motion", fontsize=8)

    ax  = fig.add_subplot(gs[1, 0])
    h2a = h2_results.get("H2a_static_regression", {})
    if not elig.empty:
        static_elig = elig[elig.movement == "STATIC"]
        for tech in TECHNIQUES:
            sub = static_elig[static_elig.technique == tech]
            if len(sub) < 5:
                continue
            ax.scatter(sub["ID"], sub["time_ms"], color=TECH_COLORS[tech],
                       alpha=0.25, s=10)
            d = h2a.get(tech, {})
            if "slope_ms_per_bit" in d and "intercept_ms" in d:
                xr = np.linspace(sub["ID"].min(), sub["ID"].max(), 80)
                ok = "✓" if d.get("H2a1_supported") else "✗"
                ax.plot(xr, d["slope_ms_per_bit"] * xr + d["intercept_ms"],
                        color=TECH_COLORS[tech], linewidth=2,
                        label=f"{tech}  r²={d.get('r2','?')}  {ok}")
    ax.set_xlabel("ID (bits)", fontsize=8); ax.set_ylabel("MT (ms)", fontsize=8)
    ax.set_title("H2a: Fitts Law — STATIC\nr² > 0.3 → H2a1 supported", fontsize=9,
                 fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.6); ax.grid(alpha=0.3)

    ax       = fig.add_subplot(gs[1, 1])
    r2_fitts = [cdf[cdf.movement == m]["fitts_regression_r2"].dropna().mean()
                for m in MOVEMENTS]
    r2_hoff  = [cdf[cdf.movement == m]["hoffmann_regression_r2"].dropna().mean()
                for m in MOVEMENTS]
    x = np.arange(len(MOVEMENTS))
    ax.bar(x - 0.2, r2_fitts, 0.38, label="Fitts (Shannon)", color="#3498db",
           edgecolor="black", linewidth=0.6, alpha=0.8)
    ax.bar(x + 0.2, r2_hoff,  0.38, label="Hoffmann",        color="#e74c3c",
           edgecolor="black", linewidth=0.6, alpha=0.8)
    for xi, mov in enumerate(MOVEMENTS):
        sub = cdf[cdf.movement == mov][
            ["fitts_regression_r2", "hoffmann_regression_r2"]].dropna()
        if len(sub) >= 3:
            _, p = stats.ttest_rel(sub["hoffmann_regression_r2"].values,
                                   sub["fitts_regression_r2"].values)
            top  = max(r2_fitts[xi], r2_hoff[xi]) + 0.02
            lbl  = sig_stars(p).strip() or "ns"
            ax.text(xi, top, lbl, ha="center", fontsize=10,
                    color="purple", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(MOVEMENTS, fontsize=9)
    ax.set_ylabel("Mean r² per condition", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.axhline(0.3, color="gray", linewidth=1, linestyle=":",
               label="r²=0.3 threshold (H2a)")
    ax.set_title("H2b/H2c: r² Fitts vs Hoffmann",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.6); ax.grid(axis="y", alpha=0.3)

    ax   = fig.add_subplot(gs[1, 2])
    x    = np.arange(len(MOVEMENTS))
    for dens in DENSITIES:
        vals = []
        for mov in MOVEMENTS:
            bp = h1c.get(f"{dens}_{mov}", {}).get("BUBBLE_vs_POINT", {})
            vals.append(bp.get("mean_diff", np.nan))
        ax.plot(x, vals, "o-", color=DENS_COLORS[dens], linewidth=1.8,
                markersize=6, label=dens, alpha=0.85)
    grand = []
    for mov in MOVEMENTS:
        cell_vals = [
            h1c.get(f"{dens}_{mov}", {}).get("BUBBLE_vs_POINT", {}).get("mean_diff", np.nan)
            for dens in DENSITIES
        ]
        finite = [v for v in cell_vals if np.isfinite(v)]
        grand.append(np.mean(finite) if finite else np.nan)
    ax.plot(x, grand, "k--o", linewidth=2.2, markersize=8,
            label="Grand mean", zorder=10)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xticks(x); ax.set_xticklabels(MOVEMENTS, fontsize=9)
    ax.set_ylabel("Bubble − Point (bits/s)", fontsize=8)
    ax.set_title("H1c: Bubble Advantage\nby Motion × Density",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.6); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    for mov, col, pat in [("SLOW", "#2ecc71", "//"), ("FAST", "#e74c3c", "\\\\")]:
        vals = cdf[cdf.movement == mov]["delta_aic"].dropna().values
        if len(vals):
            ax.hist(vals, bins=10, alpha=0.55,
                    label=f"{mov}  (n={len(vals)})",
                    color=col, edgecolor="white", hatch=pat)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="ΔAIC = 0")
    ax.set_xlabel("ΔAIC  (Hoffmann − Fitts)", fontsize=8)
    ax.set_ylabel("Conditions", fontsize=8)
    ax.set_title("H2b/H2c: ΔAIC by Motion", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.6); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    static_r2 = cdf[cdf.movement == "STATIC"]["fitts_regression_r2"].dropna().values
    if len(static_r2):
        ax.hist(static_r2, bins=10, color="#9b59b6", alpha=0.7, edgecolor="white")
        ax.axvline(0.3, color="red", linewidth=1.8, linestyle="--",
                   label="r²=0.3 adequacy threshold")
        ax.axvline(np.mean(static_r2), color="black", linewidth=1.5,
                   label=f"Mean r² = {np.mean(static_r2):.3f}")
    ax.set_xlabel("r² per STATIC condition", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title("H2a: Fitts r² Distribution (STATIC)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.6); ax.grid(alpha=0.3)

    ax      = fig.add_subplot(gs[2, 2])
    h1a_pw  = h1_results.get("H1a_overall_pairwise", {})
    pairs_  = ["BUBBLE_vs_POINT", "BUBBLE_vs_AREA", "POINT_vs_AREA"]
    dv_short = {
        "throughput_shannon_nominal_bps":   "TP-Nom",
        "throughput_hoffmann_bps":          "TP-Hoff",
        "avgTime_ms":                       "MT",
        "precisionRate_percent":            "Prec%",
        "avgErrorClicks":                   "ErrClk",
    }
    dvs_h1 = list(dv_short.keys())
    d_mat  = np.full((len(dvs_h1), len(pairs_)), np.nan)
    for ri, dv in enumerate(dvs_h1):
        pd_ = h1a_pw.get(dv, {})
        for ci, pair in enumerate(pairs_):
            d_mat[ri, ci] = (pd_.get(pair) or {}).get("cohen_d", np.nan)

    if not np.all(np.isnan(d_mat)):
        vext = max(np.nanmax(np.abs(d_mat)), 0.1)
        im   = ax.imshow(d_mat, cmap="RdBu_r", vmin=-vext, vmax=vext,
                         aspect="auto", interpolation="nearest")
        for ri in range(len(dvs_h1)):
            for ci in range(len(pairs_)):
                v = d_mat[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                            fontsize=8,
                            color="black" if abs(v) < vext * 0.6 else "white")
        ax.set_xticks(range(len(pairs_)))
        ax.set_xticklabels([p.replace("_vs_", "\nvs ") for p in pairs_], fontsize=7)
        ax.set_yticks(range(len(dvs_h1)))
        ax.set_yticklabels([dv_short[d] for d in dvs_h1], fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="Cohen's d")
    ax.set_title("H1a: Effect Size (Cohen's d)\nTechnique Pairwise — all DVs", fontsize=9,
                 fontweight="bold")

    fig.suptitle(
        "Hypothesis Test Summary  "
        "(H1: Cursor Technique  ·  H2: Fitts' Law & Hoffmann Correction)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    savefig(fig, out_dir / "fig16_hypothesis_tests.png", dpi=140)


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE SUMMARIES
# ─────────────────────────────────────────────────────────────────────────────

def console_hypothesis_summary(h1_results, h2_results, anova_results):
    W   = 90
    SEP = "─" * W
    print(f"\n{'='*W}")
    print("  HYPOTHESIS TEST RESULTS")
    print(f"{'='*W}")

    print(f"\n  {'─'*38}  H1: CURSOR TECHNIQUE  {'─'*27}")

    print("\n  H1a0  Cursor type has no significant effect on throughput or error rate.")
    print("  H1a1  Cursor type affects throughput/MT; Bubble and Area outperform Point.")
    print(f"\n  ANOVA — technique main effect:")
    print(f"  {'DV':<42} {'F':>8}  {'p(GG)':>10}  {'η²p':>7}  {'sig':>4}  verdict")
    print(f"  {SEP}")
    for dv in ["throughput_shannon_nominal_bps", "throughput_hoffmann_bps",
               "avgTime_ms", "precisionRate_percent", "avgErrorClicks"]:
        aov = anova_results.get(dv)
        if aov is None:
            continue
        p_col = "p-GG-corr" if "p-GG-corr" in aov.columns else "p-unc"
        match = aov[aov["Source"].str.strip() == "technique"]
        if match.empty:
            continue
        r       = match.iloc[0]
        F       = safe_float(r.get("F"))
        pval    = safe_float(r.get(p_col))
        eta2    = safe_float(r.get("np2") or r.get("ng2"))
        verdict = "Reject H1a0" if pval < ALPHA else "Retain H1a0"
        print(f"  {PRIMARY_DVS.get(dv,''):<42} {F:>8.2f}  {pval:>10.4f}  "
              f"{eta2:>7.3f}  {sig_stars(pval):>4}  {verdict}")

    v1 = h1_results.get("H1a1_verdict", {})
    print(f"\n  H1a1 check — Bubble/Area vs Point on TP Shannon Nominal:")
    print(f"    Bubble vs Point sig: {v1.get('bubble_vs_point_sig')}  "
          f"p={v1.get('bubble_vs_point_p')}  d={v1.get('bubble_vs_point_cohen_d')}")
    print(f"    Area   vs Point sig: {v1.get('area_vs_point_sig')}  "
          f"p={v1.get('area_vs_point_p')}  d={v1.get('area_vs_point_cohen_d')}")
    print(f"    → H1a1 {'✓ Supported' if v1.get('H1a1_supported') else '✗ Not supported'}")

    print(f"\n  {'─'*W}")
    print("  H1b0  Area cursor TP does not differ significantly from Point cursor TP.")
    print("  H1b1  Cursor × clustering interaction: Area AND Bubble lose advantage at HIGH.")
    print(f"\n  Per-clustering pairwise (TP Shannon Nominal):")
    h1b = h1_results.get("H1b_clustering_interaction", {})
    print(f"  {'Density':<10} {'Area−Pt':>9} {'p(A>P)':>9} {'d(A>P)':>8}  "
          f"{'Bub−Pt':>9} {'p(B>P)':>9} {'d(B>P)':>8}")
    print(f"  {SEP[:75]}")
    for dens in DENSITIES:
        d     = h1b.get(dens, {})
        ap    = d.get("area_minus_point", np.nan)
        ap_p  = d.get("area_vs_point_p");    ap_d = d.get("area_vs_point_cohen_d")
        bp    = d.get("bubble_minus_point", np.nan)
        bp_p  = d.get("bubble_vs_point_p"); bp_d = d.get("bubble_vs_point_cohen_d")
        ap_s  = f"{ap:.4f}"  if np.isfinite(ap) else "   n/a"
        bp_s  = f"{bp:.4f}"  if np.isfinite(bp) else "   n/a"
        ap_ps = f"{ap_p:.4f}" if ap_p is not None else "   n/a"
        bp_ps = f"{bp_p:.4f}" if bp_p is not None else "   n/a"
        ap_ds = f"{ap_d:.3f}" if ap_d is not None else "  n/a"
        bp_ds = f"{bp_d:.3f}" if bp_d is not None else "  n/a"
        print(f"  {dens:<10} {ap_s:>9} {ap_ps:>9} {ap_ds:>8}  "
              f"{bp_s:>9} {bp_ps:>9} {bp_ds:>8}")

    print(f"\n  H1b1 attenuation tests (advantage LOW vs HIGH, paired t):")
    for label in ["area_vs_point", "bubble_vs_point"]:
        att = h1b.get(f"attenuation_{label}_LOW_vs_HIGH", {})
        if not att or "insufficient" in att.get("note", ""):
            print(f"    {label}: insufficient data for paired t-test")
            continue
        adv_lo = att.get("mean_advantage_LOW"); adv_hi = att.get("mean_advantage_HIGH")
        t_  = att.get("t");  p_ = att.get("p");  d_ = att.get("cohen_d")
        confirmed = att.get("H1b1_attenuation_confirmed", False)
        print(f"    {label:<20}  adv_LOW={adv_lo:.4f}  adv_HIGH={adv_hi:.4f}  "
              f"t={t_:.3f}  p={p_:.4f}  d={d_:.3f}  {sig_stars(p_)}"
              f"  → H1b1 {'✓ confirmed' if confirmed else '✗ not confirmed'}")

    print(f"\n  {'─'*W}")
    print("  H1c0  Bubble TP does not differ from Area or Point across all conditions.")
    print("  H1c1  Motion moderates Bubble effect: advantage diminishes at FAST × HIGH.")
    print(f"\n  Bubble − Point (bits/s) per density × motion cell:")
    h1c = h1_results.get("H1c_motion_moderation", {})
    print(f"  {'Condition':<20} {'B−P Δ':>9} {'p':>8} {'d':>7}  {'B−A Δ':>9} {'p':>8}")
    print(f"  {SEP[:65]}")
    for dens in DENSITIES:
        for mov in MOVEMENTS:
            key  = f"{dens}_{mov}"
            cell = h1c.get(key, {})
            bp   = cell.get("BUBBLE_vs_POINT", {})
            ba   = cell.get("BUBBLE_vs_AREA",  {})
            tag  = "  ← H1c1 key" if (dens == "HIGH" and mov == "FAST") else ""
            print(f"  {key:<20} "
                  f"{(str(round(bp.get('mean_diff',np.nan),4)) if bp else 'n/a'):>9}  "
                  f"{(str(bp.get('p')) if bp.get('p') is not None else 'n/a'):>7}  "
                  f"{(str(bp.get('cohen_d')) if bp.get('cohen_d') is not None else 'n/a'):>6}  "
                  f"{(str(round(ba.get('mean_diff',np.nan),4)) if ba else 'n/a'):>9}  "
                  f"{(str(ba.get('p')) if ba.get('p') is not None else 'n/a'):>7}"
                  f"{tag}")
    s = h1c.get("_H1c1_summary", {})
    if s:
        print(f"\n  H1c1 directional check:")
        print(f"    Bubble−Point at HIGH×FAST = {s.get('bubble_minus_point_HIGH_FAST')}")
        print(f"    Mean Bubble−Point other 8 cells = {s.get('mean_bubble_minus_point_other_cells')}")
        dim = s.get("H1c1_diminishment_at_fast_high")
        print(f"    → H1c1 {'✓ Diminishment observed' if dim else '✗ Not observed'}")

    print(f"\n\n  {'─'*36}  H2: FITTS' LAW  {'─'*35}")

    print("\n  H2a0  ID does not predict MT under static conditions.")
    print("  H2a1  ID is a fair linear predictor of MT under static (r² > 0.3, p < .05).")
    h2a = h2_results.get("H2a_static_regression", {})
    print(f"\n  {'Technique':<12} {'n':>6} {'r²':>8} {'p':>12} {'Slope(ms/bit)':>14} "
          f"{'TP(b/s)':>9}  H2a1?")
    print(f"  {SEP[:72]}")
    for tech in TECHNIQUES + ["ALL"]:
        d = h2a.get(tech, {})
        if not d or "note" in d:
            print(f"  {tech:<12}  — {d.get('note','')}")
            continue
        ok = "✓ Supported" if d.get("H2a1_supported") else "✗ Not met"
        print(f"  {tech:<12} {d.get('n',0):>6} {d.get('r2',np.nan):>8.4f} "
              f"{d.get('p',np.nan):>12.6f} {d.get('slope_ms_per_bit',np.nan):>14.2f} "
              f"{d.get('tp_bps',np.nan):>9.3f}  {ok}")
    print(f"\n  Condition-level mean r² (STATIC): "
          f"{h2a.get('_condition_level_mean_r2_STATIC', 'n/a')}")

    print(f"\n  {'─'*W}")
    print("  H2b0  Hoffmann correction fits BETTER than Shannon under SLOW motion.")
    print("  H2b1  Hoffmann correction does NOT fit better than Shannon under SLOW motion.")
    h2b = h2_results.get("H2b_hoffmann_vs_shannon_slow", {})
    rc  = h2b.get("r2_comparison_SLOW", {})
    print(f"\n  r² under SLOW:  Fitts = {rc.get('mean_fitts_r2','n/a')}  "
          f"Hoffmann = {rc.get('mean_hoffmann_r2','n/a')}  "
          f"Δ = {rc.get('mean_delta_r2','n/a')}  "
          f"(n conditions = {rc.get('n_conditions','n/a')})")
    t_ = rc.get("t_paired_r2"); p_ = rc.get("p_paired_r2"); d_ = rc.get("cohen_d_r2")
    if t_ is not None:
        print(f"  Paired t (Hoffmann vs Fitts r²): t={t_:.3f}  p={p_:.4f}  d={d_:.3f}"
              f"  {sig_stars(p_)}")
    print(f"  Hoffmann better than Fitts: {rc.get('hoffmann_better_than_fitts')}")
    verdict_b0 = rc.get("H2b0_supported", False)
    print(f"  → H2b0 {'✓ Supported' if verdict_b0 else '✗ Not supported'}  "
          f"/ H2b1 {'✓ Supported' if not verdict_b0 else '✗ Not supported'}")
    ac = h2b.get("aic_comparison_SLOW", {})
    if ac:
        print(f"  ΔAIC (SLOW): mean={ac.get('mean_daic','n/a')}  "
              f"t vs 0 = {ac.get('t_vs_0','n/a')}  p={ac.get('p_vs_0','n/a')}  "
              f"{'Hoffmann preferred (neg. ΔAIC)' if ac.get('hoffmann_preferred_aic') else 'Fitts preferred'}  "
              f"H2b0 AIC: {'✓' if ac.get('H2b0_aic_supported') else '✗'}")

    print(f"\n  {'─'*W}")
    print("  H2c0  Hoffmann and Shannon r² do NOT differ in the random-walk (FAST) condition.")
    print("  H2c1  One model demonstrates improved performance in random walk.")
    h2c  = h2_results.get("H2c_random_walk", {})
    rcf  = h2c.get("r2_comparison_FAST", {})
    print(f"\n  r² under FAST:  Fitts = {rcf.get('mean_fitts_r2','n/a')}  "
          f"Hoffmann = {rcf.get('mean_hoffmann_r2','n/a')}  "
          f"Δ = {rcf.get('mean_delta_r2','n/a')}  "
          f"(n conditions = {rcf.get('n_conditions','n/a')})")
    t_ = rcf.get("t_paired_r2"); p_ = rcf.get("p_paired_r2"); d_ = rcf.get("cohen_d_r2")
    if t_ is not None:
        print(f"  Paired t (Hoffmann vs Fitts r²): t={t_:.3f}  p={p_:.4f}  d={d_:.3f}"
              f"  {sig_stars(p_)}")
    c0_retained  = rcf.get("H2c0_retained")
    c1_supported = rcf.get("H2c1_supported")
    print(f"  → H2c0 {'✓ Retained (p ≥ .05)' if c0_retained else '✗ Rejected'}"
          f"  /  H2c1 {'✓ Supported' if c1_supported else '✗ Not supported'}")
    acf = h2c.get("aic_comparison_FAST", {})
    if acf:
        print(f"  ΔAIC (FAST): mean={acf.get('mean_daic','n/a')}  "
              f"t vs 0 = {acf.get('t_vs_0','n/a')}  p={acf.get('p_vs_0','n/a')}  "
              f"H2c0 AIC: {'✓ retained' if acf.get('H2c0_aic_retained') else '✗ rejected'}")
    csf = h2c.get("daic_SLOW_vs_FAST", {})
    if csf:
        print(f"\n  Context — ΔAIC SLOW vs FAST:  "
              f"SLOW={csf.get('mean_daic_SLOW'):.3f}  FAST={csf.get('mean_daic_FAST'):.3f}  "
              f"t={csf.get('t'):.3f}  p={csf.get('p'):.4f}  {sig_stars(csf.get('p', 1.0))}")
        print(f"  (Significant result would confirm Hoffmann gain is motion-type-specific)")
    sf = h2c.get("fitts_r2_SLOW_vs_FAST", {})
    if sf:
        print(f"  Context — Fitts r² SLOW vs FAST:  "
              f"SLOW={sf.get('mean_SLOW'):.4f}  FAST={sf.get('mean_FAST'):.4f}  "
              f"t={sf.get('t'):.3f}  p={sf.get('p'):.4f}  {sig_stars(sf.get('p', 1.0))}")

    print(f"\n{'='*W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING  (Figures 1–15, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def fig1_boxplots(cdf, out_dir):
    dvs   = list(PRIMARY_DVS.items())
    ncols = 4
    nrows = int(np.ceil(len(dvs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    axes = axes.flatten()
    for ax, (dv, label) in zip(axes, dvs):
        data = [cdf[cdf.technique == t][dv].dropna().values for t in TECHNIQUES]
        bp   = ax.boxplot(data, patch_artist=True, notch=False,
                          widths=0.55, showfliers=True,
                          flierprops=dict(marker=".", markersize=3, alpha=0.4))
        for patch, c in zip(bp["boxes"], [TECH_COLORS[t] for t in TECHNIQUES]):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set(color="black", linewidth=1.5)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(TECHNIQUES, fontsize=9)
        ax.set_ylabel(label, fontsize=7)
        ax.set_title(label, fontsize=8, fontweight="bold", pad=3)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[len(dvs):]:
        ax.set_visible(False)
    fig.suptitle("Primary DVs by Technique  (condition-level observations)", fontsize=12)
    savefig(fig, out_dir / "fig01_boxplots_by_technique.png")


def fig2_main_effects(cdf, out_dir):
    dvs     = list(PRIMARY_DVS.items())[:6]
    factors = [
        ("technique",  TECHNIQUES, list(TECH_COLORS.values())),
        ("movement",   MOVEMENTS,  list(MOV_COLORS.values())),
        ("clustering", DENSITIES,  list(DENS_COLORS.values())),
    ]
    fig, axes = plt.subplots(len(dvs), 3, figsize=(16, 3.5 * len(dvs)))
    for row_i, (dv, label) in enumerate(dvs):
        for col_i, (factor, levels, colors) in enumerate(factors):
            ax    = axes[row_i][col_i]
            means = [cdf[cdf[factor] == l][dv].mean() for l in levels]
            sems  = [cdf[cdf[factor] == l][dv].sem()  for l in levels]
            ax.bar(range(len(levels)), means, yerr=sems, capsize=4,
                   color=colors, edgecolor="black", linewidth=0.6,
                   alpha=0.78, error_kw={"elinewidth": 1.2, "ecolor": "black"})
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels, fontsize=8)
            if col_i == 0:
                ax.set_ylabel(label, fontsize=7)
            ax.set_title(f"{label[:22]}\n× {factor}", fontsize=7)
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Main Effects: Mean ± SEM per Factor", fontsize=12)
    savefig(fig, out_dir / "fig02_main_effects_bars.png")


def fig3_interactions_2way(cdf, out_dir):
    tp_dvs = [
        ("throughput_shannon_nominal_bps",   "TP Shannon Nominal (bits/s)"),
        ("throughput_shannon_effective_bps", "TP Shannon Effective (bits/s)"),
        ("throughput_hoffmann_bps",          "TP Hoffmann (bits/s)"),
        ("avgTime_ms",                       "Mean MT (ms)"),
    ]
    interactions = [
        ("technique",  "movement",   TECHNIQUES, MOVEMENTS,  TECH_COLORS, "Technique"),
        ("technique",  "clustering", TECHNIQUES, DENSITIES,  TECH_COLORS, "Technique"),
        ("movement",   "clustering", MOVEMENTS,  DENSITIES,  MOV_COLORS,  "Motion"),
    ]
    fig, axes = plt.subplots(len(tp_dvs), 3, figsize=(16, 4.5 * len(tp_dvs)))
    for row_i, (dv, label) in enumerate(tp_dvs):
        for col_i, (f1, f2, lev1, lev2, col_map, leg_title) in enumerate(interactions):
            ax = axes[row_i][col_i]
            for l1 in lev1:
                sub   = cdf[cdf[f1] == l1]
                means = [sub[sub[f2] == l2][dv].mean() for l2 in lev2]
                sems  = [sub[sub[f2] == l2][dv].sem()  for l2 in lev2]
                c     = col_map.get(l1, "#888")
                ax.errorbar(range(len(lev2)), means, yerr=sems,
                            label=l1, color=c, marker="o", markersize=5,
                            linewidth=1.8, capsize=3, capthick=1.2)
            ax.set_xticks(range(len(lev2)))
            ax.set_xticklabels(lev2, fontsize=8)
            ax.set_xlabel(f2.capitalize(), fontsize=8)
            if col_i == 0:
                ax.set_ylabel(label, fontsize=7)
            ax.set_title(f"{f1} × {f2}", fontsize=8)
            ax.legend(title=leg_title, fontsize=7, title_fontsize=7,
                      loc="best", framealpha=0.6)
            ax.grid(alpha=0.3)
    fig.suptitle("Two-Way Interaction Profiles  (Mean ± SEM)", fontsize=12)
    savefig(fig, out_dir / "fig03_2way_interactions.png")


def fig4_heatmaps_3way(cdf, out_dir):
    dvs_plot = [
        ("throughput_shannon_nominal_bps",  "TP Shannon Nominal (bits/s)"),
        ("throughput_hoffmann_bps",          "TP Hoffmann (bits/s)"),
        ("avgTime_ms",                       "Mean MT (ms)"),
        ("precisionRate_percent",            "Precision Rate (%)"),
    ]
    fig, axes = plt.subplots(len(dvs_plot), len(DENSITIES),
                             figsize=(13, 4.2 * len(dvs_plot)))
    for row_i, (dv, label) in enumerate(dvs_plot):
        mats = []
        for dens in DENSITIES:
            mat = np.full((len(MOVEMENTS), len(TECHNIQUES)), np.nan)
            for mi, mov in enumerate(MOVEMENTS):
                for ti, tech in enumerate(TECHNIQUES):
                    v = cdf[(cdf.technique == tech) &
                             (cdf.movement  == mov)  &
                             (cdf.clustering == dens)][dv].mean()
                    mat[mi, ti] = v
            mats.append(mat)
        vmin = np.nanmin(mats); vmax = np.nanmax(mats)
        cmap = "RdYlGn" if "throughput" in dv or "precision" in dv else "RdYlGn_r"
        for col_i, (dens, mat) in enumerate(zip(DENSITIES, mats)):
            ax = axes[row_i][col_i]
            im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="auto", interpolation="nearest")
            ax.set_xticks(range(len(TECHNIQUES))); ax.set_xticklabels(TECHNIQUES, fontsize=8)
            ax.set_yticks(range(len(MOVEMENTS)));  ax.set_yticklabels(MOVEMENTS,  fontsize=8)
            ax.set_title(f"Density = {dens}", fontsize=9)
            if col_i == 0:
                ax.set_ylabel(label, fontsize=7)
            for mi in range(len(MOVEMENTS)):
                for ti in range(len(TECHNIQUES)):
                    v = mat[mi, ti]
                    if not np.isnan(v):
                        brightness = (v - vmin) / max(vmax - vmin, 1e-6)
                        txt_color  = "white" if brightness < 0.35 or brightness > 0.75 else "black"
                        ax.text(ti, mi, f"{v:.1f}", ha="center", va="center",
                                fontsize=7.5, color=txt_color, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.suptitle("3-Way Condition Cell Means  (Technique × Motion, faceted by Density)",
                 fontsize=11)
    savefig(fig, out_dir / "fig04_3way_heatmaps.png")


def fig5_fitts_scatter(elig, reg_tech, out_dir):
    if elig.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, tech in zip(axes, TECHNIQUES):
        sub = elig[elig.technique == tech]
        for mov in MOVEMENTS:
            sm = sub[sub.movement == mov]
            ax.scatter(sm["ID"], sm["time_ms"],
                       color=MOV_COLORS[mov], alpha=0.25, s=10, label=mov)
        key = f"{tech}_fitts"
        if key in reg_tech:
            r  = reg_tech[key]
            xr = np.linspace(sub["ID"].min(), sub["ID"].max(), 200)
            ax.plot(xr, r["slope_ms"] * xr + r["intercept_ms"],
                    "k-", linewidth=2.2,
                    label=f"OLS  r²={r['r2']}  TP={r['tp_bps']} b/s")
        ax.set_title(f"{tech}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Index of Difficulty  ID (bits)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Movement Time  MT (ms)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.6)
        ax.grid(alpha=0.3)
    fig.suptitle("Fitts' Law Regression  MT ~ log₂(A/W + 1)  by Technique", fontsize=11)
    savefig(fig, out_dir / "fig05_fitts_regression.png")


def fig6_fitts_grid(elig, reg_tm, out_dir):
    if elig.empty:
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for row_i, tech in enumerate(TECHNIQUES):
        for col_i, mov in enumerate(MOVEMENTS):
            ax  = axes[row_i][col_i]
            sub = elig[(elig.technique == tech) & (elig.movement == mov)]
            if len(sub) < 5:
                ax.set_title(f"{tech} × {mov}", fontsize=8)
                ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")
                continue
            for dens in DENSITIES:
                d = sub[sub.clustering == dens]
                ax.scatter(d["ID"], d["time_ms"],
                           color=DENS_COLORS[dens], alpha=0.3, s=12, label=dens)
            key = f"{tech}_{mov}"
            if key in reg_tm:
                r  = reg_tm[key]
                xr = np.linspace(sub["ID"].min(), sub["ID"].max(), 80)
                ax.plot(xr, r["slope_ms"] * xr + r["intercept_ms"],
                        "k--", linewidth=1.6,
                        label=f"r²={r['r2']},  TP={r['tp_bps']}")
            ax.set_title(f"{tech} × {mov}", fontsize=8, fontweight="bold")
            ax.set_xlabel("ID (bits)", fontsize=7)
            ax.set_ylabel("MT (ms)", fontsize=7)
            ax.legend(fontsize=6, loc="upper left", framealpha=0.6)
            ax.grid(alpha=0.25)
    fig.suptitle("Fitts' Law  per Technique × Motion  (colour = density)", fontsize=11)
    savefig(fig, out_dir / "fig06_fitts_grid.png")


def fig7_tp_violins(cdf, out_dir):
    tp_specs = [
        ("throughput_shannon_nominal_bps",   "Shannon Nominal"),
        ("throughput_shannon_effective_bps", "Shannon Effective"),
        ("throughput_hoffmann_bps",          "Hoffmann"),
    ]
    factors = [
        ("technique",  TECHNIQUES, TECH_COLORS),
        ("movement",   MOVEMENTS,  MOV_COLORS),
        ("clustering", DENSITIES,  DENS_COLORS),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for row_i, (dv, dv_label) in enumerate(tp_specs):
        for col_i, (factor, levels, col_map) in enumerate(factors):
            ax   = axes[row_i][col_i]
            data = [cdf[cdf[factor] == l][dv].dropna().values for l in levels]
            parts = ax.violinplot(data, positions=range(len(levels)),
                                  showmeans=True, showmedians=True,
                                  showextrema=True, widths=0.7)
            for pc, l in zip(parts["bodies"], levels):
                pc.set_facecolor(col_map.get(l, "#aaa")); pc.set_alpha(0.55)
            for pos, (vals, l) in enumerate(zip(data, levels)):
                jitter = np.random.uniform(-0.08, 0.08, len(vals))
                ax.scatter(pos + jitter, vals, color=col_map.get(l, "#aaa"),
                           alpha=0.55, s=16, zorder=3)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels, fontsize=8)
            if col_i == 0:
                ax.set_ylabel(f"{dv_label} (bits/s)", fontsize=8)
            ax.set_title(f"{dv_label}\n× {factor}", fontsize=8)
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Throughput Distributions  (Violin + Strip)", fontsize=12)
    savefig(fig, out_dir / "fig07_throughput_violins.png")


def fig8_error_precision(cdf, out_dir):
    factors = [
        ("technique",  TECHNIQUES, TECH_COLORS),
        ("movement",   MOVEMENTS,  MOV_COLORS),
        ("clustering", DENSITIES,  DENS_COLORS),
    ]
    dvs_acc = [
        ("precisionRate_percent",  "Precision Rate (%)"),
        ("avgErrorClicks",         "Mean Error Clicks / Trial"),
        ("avgNormalizedDistance",  "Mean Normalised Distance"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for row_i, (dv, label) in enumerate(dvs_acc):
        for col_i, (factor, levels, col_map) in enumerate(factors):
            ax    = axes[row_i][col_i]
            means = [cdf[cdf[factor] == l][dv].mean() for l in levels]
            sems  = [cdf[cdf[factor] == l][dv].sem()  for l in levels]
            ax.bar(range(len(levels)), means, yerr=sems, capsize=4,
                   color=[col_map.get(l, "#aaa") for l in levels],
                   edgecolor="black", linewidth=0.6, alpha=0.78)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels, fontsize=8)
            if col_i == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.set_title(f"{label[:22]}\n× {factor}", fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            if "Precision" in label:
                ax.set_ylim(0, 107)
                ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
    fig.suptitle("Accuracy Metrics by Factor  (Mean ± SEM)", fontsize=12)
    savefig(fig, out_dir / "fig08_accuracy_metrics.png")


def fig9_effect_sizes(eta2_df, out_dir):
    if eta2_df.empty:
        return
    focus_dvs = [
        ("throughput_shannon_nominal_bps",   "TP Shannon Nominal"),
        ("throughput_shannon_effective_bps", "TP Shannon Effective"),
        ("throughput_hoffmann_bps",          "TP Hoffmann"),
        ("avgTime_ms",                       "Mean MT"),
        ("precisionRate_percent",            "Precision Rate"),
    ]
    effect_labels = {
        "technique":                         "Technique (A)",
        "movement":                          "Movement (B)",
        "clustering":                        "Clustering (C)",
        "technique * movement":              "A × B",
        "technique * clustering":            "A × C",
        "movement * clustering":             "B × C",
        "technique * movement * clustering": "A × B × C",
    }
    fig, axes = plt.subplots(1, len(focus_dvs),
                             figsize=(4.2 * len(focus_dvs), 6), sharey=True)
    for ax, (dv, short_label) in zip(axes, focus_dvs):
        sub = eta2_df[(eta2_df.dv == dv) & eta2_df.partial_eta2.notna()].copy()
        sub["partial_eta2"] = pd.to_numeric(sub["partial_eta2"], errors="coerce")
        sub = sub.dropna(subset=["partial_eta2"])
        etas, labels_out, colors_out = [], [], []
        for src in ANOVA_SOURCE_ORDER:
            match = sub[sub["source"].str.strip() == src]
            if match.empty:
                continue
            eta = safe_float(match.iloc[0]["partial_eta2"])
            sig = bool(match.iloc[0]["sig"])
            etas.append(eta); labels_out.append(effect_labels.get(src, src))
            colors_out.append("#c0392b" if sig else "#95a5a6")
        y_pos = np.arange(len(etas))
        ax.barh(y_pos, etas, color=colors_out, edgecolor="black",
                linewidth=0.6, height=0.6, alpha=0.85)
        for y, eta in zip(y_pos, etas):
            ax.text(eta + 0.002, y, f"{eta:.3f}", va="center", ha="left", fontsize=7.5)
        for thresh, col, lab in [(0.01, "steelblue", "small (.01)"),
                                  (0.06, "darkorange", "medium (.06)"),
                                  (0.14, "darkred", "large (.14)")]:
            ax.axvline(thresh, color=col, linewidth=0.8, linestyle=":", label=lab)
        ax.set_yticks(y_pos); ax.set_yticklabels(labels_out, fontsize=8)
        ax.set_xlabel("Partial η²", fontsize=8)
        ax.set_title(short_label, fontsize=9, fontweight="bold")
        ax.set_xlim(left=0); ax.grid(axis="x", alpha=0.25)
    axes[-1].legend(fontsize=7, loc="lower right")
    red_p  = mpatches.Patch(color="#c0392b", label="Significant (p<.05)")
    grey_p = mpatches.Patch(color="#95a5a6", label="Non-significant")
    axes[0].legend(handles=[red_p, grey_p], fontsize=7, loc="lower right")
    fig.suptitle("Effect Sizes  (Partial η²)  ·  Dotted lines = Cohen thresholds",
                 fontsize=11)
    savefig(fig, out_dir / "fig09_effect_sizes.png")


def fig10_anova_table(anova_results, out_dir):
    focus_dvs = {
        "throughput_shannon_nominal_bps":   "TP-Nom",
        "throughput_shannon_effective_bps": "TP-Eff",
        "throughput_hoffmann_bps":          "TP-Hoff",
        "avgTime_ms":                       "MT",
        "precisionRate_percent":            "Prec%",
        "avgErrorClicks":                   "Err",
    }
    col_headers = ["Effect"] + list(focus_dvs.values())
    cell_data   = []
    for src in ANOVA_SOURCE_ORDER:
        row_vals = [src]
        for dv in focus_dvs:
            aov = anova_results.get(dv)
            if aov is None:
                row_vals.append("—"); continue
            match = aov[aov["Source"].str.strip() == src]
            if match.empty:
                row_vals.append("—"); continue
            r    = match.iloc[0]
            F    = safe_float(r.get("F"))
            df1  = r.get("ddof1", r.get("DF", ""))
            df2  = r.get("ddof2", "")
            p_col= "p-GG-corr" if "p-GG-corr" in aov.columns else "p-unc"
            pval = safe_float(r.get(p_col))
            eta2 = safe_float(r.get("np2") or r.get("ng2"))
            sig  = "*" if pval < ALPHA else ""
            try:
                df_str = f"({int(df1)},{int(df2)})"
            except (ValueError, TypeError):
                df_str = ""
            row_vals.append(
                f"F{df_str}={F:.2f}\np={pval:.3f} η²={eta2:.3f}{sig}"
                if not np.isnan(F) else "—"
            )
        cell_data.append(row_vals)

    fig, ax = plt.subplots(figsize=(18, max(5, 1.4 * len(ANOVA_SOURCE_ORDER) + 2)))
    ax.axis("off")
    col_w = [0.18] + [0.135] * len(focus_dvs)
    tbl   = ax.table(cellText=cell_data, colLabels=col_headers,
                     cellLoc="center", loc="center", colWidths=col_w)
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1, 2.6)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f6f7")
        else:
            cell.set_facecolor("#ffffff")
        if row > 0:
            txt = cell.get_text().get_text()
            if "p=" in txt:
                try:
                    p_val = float(txt.split("p=")[1].split(" ")[0].split("\n")[0])
                    if p_val < ALPHA:
                        cell.set_facecolor("#fde8e8")
                except (ValueError, IndexError):
                    pass
    fig.suptitle("3-Way RM ANOVA Summary  (GG-corrected  ·  * p < .05)", fontsize=11)
    savefig(fig, out_dir / "fig10_anova_summary_table.png", dpi=120)


def fig11_posthoc(cdf, posthoc_results, out_dir):
    focus_dvs = [
        ("throughput_shannon_nominal_bps", "TP Shannon Nominal (bits/s)"),
        ("throughput_hoffmann_bps",        "TP Hoffmann (bits/s)"),
        ("avgTime_ms",                     "Mean MT (ms)"),
        ("precisionRate_percent",          "Precision Rate (%)"),
    ]
    factors = [
        ("technique",  TECHNIQUES, TECH_COLORS),
        ("movement",   MOVEMENTS,  MOV_COLORS),
        ("clustering", DENSITIES,  DENS_COLORS),
    ]
    fig, axes = plt.subplots(len(focus_dvs), 3, figsize=(16, 4.5 * len(focus_dvs)))
    for row_i, (dv, label) in enumerate(focus_dvs):
        for col_i, (factor, levels, col_map) in enumerate(factors):
            ax      = axes[row_i][col_i]
            ph_dict = posthoc_results.get(dv, {})
            ph      = ph_dict.get(factor)
            means   = [cdf[cdf[factor] == l][dv].mean() for l in levels]
            sems    = [cdf[cdf[factor] == l][dv].sem()  for l in levels]
            ax.bar(range(len(levels)), means, yerr=sems, capsize=4,
                   color=[col_map.get(l, "#aaa") for l in levels],
                   edgecolor="black", linewidth=0.6, alpha=0.78)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels, fontsize=8)
            if col_i == 0:
                ax.set_ylabel(label, fontsize=7)
            ax.set_title(f"{label[:20]}\n× {factor}", fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            if ph is None or ph.empty:
                continue
            p_col = next((c for c in ["p-corr", "p-bonf", "p-unc"] if c in ph.columns), None)
            if p_col is None:
                continue
            valid_top = [m + s for m, s in zip(means, sems)
                         if not (np.isnan(m) or np.isnan(s))]
            if not valid_top:
                continue
            y_base    = max(valid_top) * 1.05
            y_step    = max(valid_top) * 0.055
            bracket_h = max(valid_top) * 0.025
            current_y = y_base
            for _, ph_row in ph.iterrows():
                a    = ph_row.get("A"); b = ph_row.get("B")
                pval = safe_float(ph_row.get(p_col, 1.0))
                if a not in levels or b not in levels:
                    continue
                if np.isnan(pval) or pval >= ALPHA * 3:
                    continue
                xa, xb  = levels.index(a), levels.index(b)
                sig_str = ("***" if pval < 0.001 else "**" if pval < 0.01 else
                           "*" if pval < ALPHA else "ns")
                ax.plot([xa, xa, xb, xb],
                        [current_y, current_y + bracket_h,
                         current_y + bracket_h, current_y], "k-", linewidth=0.9)
                ax.text((xa + xb) / 2, current_y + bracket_h * 1.1,
                        sig_str, ha="center", va="bottom", fontsize=8.5)
                current_y += y_step
    fig.suptitle(
        "Post-hoc Pairwise Comparisons  (Bonferroni  ·  * p<.05  ** p<.01  *** p<.001)",
        fontsize=10)
    savefig(fig, out_dir / "fig11_posthoc_brackets.png")


def fig12_tp_concordance(cdf, out_dir):
    tp_cols = [
        ("throughput_shannon_nominal_bps",   "TP Shannon Nominal"),
        ("throughput_shannon_effective_bps", "TP Shannon Effective"),
        ("throughput_hoffmann_bps",          "TP Hoffmann"),
    ]
    sub = cdf[[c for c, _ in tp_cols]].dropna()
    if len(sub) < 3:
        return
    pairs = list(combinations(range(len(tp_cols)), 2))
    fig, axes = plt.subplots(2, len(pairs), figsize=(5 * len(pairs), 10))
    for col_i, (i, j) in enumerate(pairs):
        c1, l1 = tp_cols[i]; c2, l2 = tp_cols[j]
        x, y   = sub[c1].values, sub[c2].values
        ax = axes[0][col_i]
        ax.scatter(x, y, alpha=0.4, s=18, color="#3498db", edgecolors="none")
        m, b, r, _, _ = stats.linregress(x, y)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, m * xr + b, "r-", linewidth=1.5,
                label=f"r={r:.3f},  p={stats.pearsonr(x,y)[1]:.4f}")
        ax.plot(xr, xr, "k--", linewidth=1, alpha=0.4, label="identity")
        ax.set_xlabel(l1, fontsize=8); ax.set_ylabel(l2, fontsize=8)
        ax.set_title("Correlation", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax = axes[1][col_i]
        mean_ = (x + y) / 2; diff_ = x - y
        md = np.mean(diff_); sd = np.std(diff_, ddof=1)
        ax.scatter(mean_, diff_, alpha=0.4, s=18, color="#2ecc71", edgecolors="none")
        ax.axhline(md, color="red", linewidth=1.5, label=f"Mean diff = {md:.3f}")
        ax.axhline(md + 1.96*sd, color="orange", linewidth=1, linestyle="--",
                   label=f"+1.96SD = {md+1.96*sd:.3f}")
        ax.axhline(md - 1.96*sd, color="orange", linewidth=1, linestyle="--",
                   label=f"−1.96SD = {md-1.96*sd:.3f}")
        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax.set_xlabel("Mean (bits/s)", fontsize=8)
        ax.set_ylabel("Difference (bits/s)", fontsize=8)
        ax.set_title(f"Bland-Altman: {l1[:15]} − {l2[:15]}", fontsize=8)
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)
    fig.suptitle("Throughput Measure Concordance  (Scatter + Bland-Altman)", fontsize=11)
    savefig(fig, out_dir / "fig12_tp_concordance.png")


def fig13_participant_profiles(cdf, out_dir):
    dv   = "throughput_shannon_nominal_bps"
    pids = sorted(cdf["participant"].unique())
    if len(pids) < 2:
        return
    factors = [("technique", TECHNIQUES), ("movement", MOVEMENTS), ("clustering", DENSITIES)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap_pid  = plt.cm.tab20
    for ax, (factor, levels) in zip(axes, factors):
        for k, pid in enumerate(pids):
            sub   = cdf[cdf.participant == pid]
            means = [sub[sub[factor] == l][dv].mean() for l in levels]
            ax.plot(range(len(levels)), means, "o-",
                    color=cmap_pid(k / max(len(pids) - 1, 1)),
                    alpha=0.55, linewidth=1.2, markersize=4, label=f"P{pid}")
        grand = [cdf[cdf[factor] == l][dv].mean() for l in levels]
        ax.plot(range(len(levels)), grand, "k-o",
                linewidth=2.8, markersize=8, label="Grand mean", zorder=10)
        ax.set_xticks(range(len(levels))); ax.set_xticklabels(levels, fontsize=9)
        ax.set_title(f"TP Shannon Nominal × {factor}", fontsize=9)
        ax.set_ylabel("bits/s", fontsize=8)
        ax.legend(fontsize=6, loc="best", framealpha=0.6, ncol=max(1, len(pids) // 8))
        ax.grid(alpha=0.3)
    fig.suptitle("Individual Participant Profiles  (bold = grand mean)", fontsize=11)
    savefig(fig, out_dir / "fig13_participant_profiles.png")


def fig14_model_comparison(cdf, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    for col, lab, c in [("fitts_regression_r2",    "Fitts (nominal)",              "#3498db"),
                         ("hoffmann_regression_r2", "Hoffmann (velocity-corrected)", "#e74c3c")]:
        vals = cdf[col].dropna().values
        if len(vals):
            ax.hist(vals, bins=14, alpha=0.55, label=lab, color=c, edgecolor="white")
    ax.set_xlabel("r² (OLS fit, per condition)", fontsize=9)
    ax.set_ylabel("Number of conditions", fontsize=9)
    ax.set_title("r² per Condition: Fitts vs Hoffmann", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax  = axes[1]
    dr2 = cdf["delta_r2"].dropna().values
    if len(dr2):
        ax.hist(dr2, bins=14, color="#27ae60", alpha=0.72, edgecolor="white")
        ax.axvline(0, color="red", linewidth=1.8, linestyle="--", label="Δr²= 0")
        ax.axvline(np.mean(dr2), color="black", linewidth=1.5,
                   label=f"Mean = {np.mean(dr2):.4f}")
        t, p = stats.ttest_1samp(dr2, 0)
        ax.set_title(f"Δr²  (Hoffmann − Fitts)\nt = {t:.2f},  p = {p:.4f}", fontsize=9)
    ax.set_xlabel("Δr²", fontsize=9); ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.annotate("positive → Hoffmann fits better", xy=(0.5, 0.93),
                xycoords="axes fraction", ha="center", fontsize=7.5, color="gray")
    ax   = axes[2]
    daic = cdf["delta_aic"].dropna().values
    if len(daic):
        ax.hist(daic, bins=14, color="#e67e22", alpha=0.72, edgecolor="white")
        ax.axvline(0, color="red", linewidth=1.8, linestyle="--", label="ΔAIC = 0")
        ax.axvline(np.mean(daic), color="black", linewidth=1.5,
                   label=f"Mean = {np.mean(daic):.2f}")
        t, p = stats.ttest_1samp(daic, 0)
        ax.set_title(f"ΔAIC  (Hoffmann − Fitts)\nt = {t:.2f},  p = {p:.4f}", fontsize=9)
    ax.set_xlabel("ΔAIC", fontsize=9); ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.annotate("negative → Hoffmann preferred", xy=(0.5, 0.93),
                xycoords="axes fraction", ha="center", fontsize=7.5, color="gray")
    fig.suptitle("Regression Model Comparison  (H₀: Δ = 0,  one-sample t-test)", fontsize=11)
    savefig(fig, out_dir / "fig14_model_comparison.png")


def fig15_diagnostics(cdf, out_dir):
    dvs   = list(PRIMARY_DVS.items())[:6]
    fig, axes = plt.subplots(2, len(dvs), figsize=(4.5 * len(dvs), 9))
    for col_i, (dv, label) in enumerate(dvs):
        vals = cdf[dv].dropna().values
        ax   = axes[0][col_i]
        if len(vals) >= 3:
            (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm")
            ax.scatter(osm, osr, s=12, color="#3498db", alpha=0.6)
            ax.plot(osm, slope * np.array(osm) + intercept, "r-",
                    linewidth=1.5, label=f"r={r:.3f}")
        ax.set_title(f"Q-Q: {label[:22]}", fontsize=8)
        ax.set_xlabel("Theoretical quantiles", fontsize=7)
        ax.set_ylabel("Sample quantiles", fontsize=7)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax   = axes[1][col_i]
        if len(vals) >= 3:
            fitted = np.full_like(vals, vals.mean())
            resid  = (vals - fitted) / (vals.std(ddof=1) + 1e-9)
            ax.scatter(fitted, resid, s=10, alpha=0.5, color="#2ecc71")
            ax.axhline(0,  color="red",  linewidth=1.2, linestyle="--")
            ax.axhline( 2, color="gray", linewidth=0.7, linestyle=":")
            ax.axhline(-2, color="gray", linewidth=0.7, linestyle=":")
        ax.set_title(f"Residuals: {label[:22]}", fontsize=8)
        ax.set_xlabel("Fitted (grand mean)", fontsize=7)
        ax.set_ylabel("Standardised residual", fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Diagnostic Plots  (Q-Q normality  +  standardised residuals)", fontsize=11)
    savefig(fig, out_dir / "fig15_diagnostics.png")


def console_anova_summary(anova_results):
    SEP = "─" * 88
    print(f"\n{'='*88}")
    print("  3-WAY REPEATED-MEASURES ANOVA  (Greenhouse-Geisser correction applied)")
    print(f"{'='*88}")
    for dv, aov in anova_results.items():
        if aov is None:
            continue
        label = PRIMARY_DVS.get(dv, dv)
        p_col = "p-GG-corr" if "p-GG-corr" in aov.columns else "p-unc"
        print(f"\n  DV: {label}")
        print(f"  {SEP}")
        print(f"  {'Effect':<40} {'F':>8} {'df1':>5} {'df2':>7}  {'p (GG-corr)':>13}  {'η²p':>7}  sig")
        print(f"  {SEP}")
        for src in ANOVA_SOURCE_ORDER:
            match = aov[aov["Source"].str.strip() == src]
            if match.empty:
                continue
            r    = match.iloc[0]
            F    = safe_float(r.get("F"))
            df1  = r.get("ddof1", r.get("DF", ""))
            df2  = r.get("ddof2", "")
            pval = safe_float(r.get(p_col))
            eta2 = safe_float(r.get("np2") or r.get("ng2"))
            F_s   = f"{F:8.2f}"      if np.isfinite(F)    else f"{'':>8}"
            df1_s = f"{int(df1):5d}" if isinstance(df1, (float, int, np.floating)) else f"{str(df1):>5}"
            df2_s = f"{int(df2):7d}" if isinstance(df2, (float, int, np.floating)) else f"{str(df2):>7}"
            p_s   = f"{pval:13.4f}"  if np.isfinite(pval) else f"{'':>13}"
            e_s   = f"{eta2:7.3f}"   if np.isfinite(eta2) else f"{'':>7}"
            print(f"  {src:<40} {F_s} {df1_s} {df2_s}  {p_s}  {e_s}  "
                  f"{sig_stars(pval)}")
    print(f"\n{'='*88}\n")


def console_posthoc_summary(posthoc_results):
    print("\n  POST-HOC SUMMARY  (Bonferroni-corrected, significant pairs only)")
    print("  " + "─" * 60)
    for dv, ph_dict in posthoc_results.items():
        label = PRIMARY_DVS.get(dv, dv)
        for effect, ph in ph_dict.items():
            if ph is None or ph.empty:
                continue
            p_col = next((c for c in ["p-corr", "p-bonf", "p-unc"]
                          if c in ph.columns), None)
            if p_col is None:
                continue
            sig_rows = ph[ph[p_col] < ALPHA]
            if sig_rows.empty:
                continue
            print(f"\n  [{label[:30]}]  {effect}")
            for _, row in sig_rows.iterrows():
                A = row.get("A", "?"); B = row.get("B", "?")
                p = safe_float(row.get(p_col, 1.0))
                d = safe_float(row.get("cohen-d") or row.get("cohen_d") or np.nan)
                d_s = f"  d={d:.3f}" if np.isfinite(d) else ""
                print(f"    {A:10s} vs {B:10s}  p={p:.4f}{d_s}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cursor Technique Study — Statistical Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("data_dir",
                        help="Directory containing P*_session.json files")
    parser.add_argument("--out", default="analysis_output",
                        help="Output directory  [default: ./analysis_output]")

    # ── Outlier removal CLI flags ─────────────────────────────────────────────
    parser.add_argument(
        "--outlier-method",
        default=OUTLIER_METHOD,
        choices=["sd", "iqr", "mad", "none"],
        help=(
            "Statistical outlier detection method applied per participant × condition cell.  "
            "'sd'  = ±N standard deviations (N set by --outlier-sd).  "
            "'iqr' = Tukey fences at Q1−N·IQR / Q3+N·IQR (N set by --outlier-iqr).  "
            "'mad' = modified Z-score (threshold set by --outlier-mad).  "
            "'none'= disable statistical detection; absolute bounds still apply.  "
            "[default: iqr]"
        ),
    )
    parser.add_argument(
        "--outlier-sd", type=float, default=OUTLIER_SD_THRESHOLD,
        metavar="N",
        help="SD multiplier for --outlier-method=sd  [default: 3.0]",
    )
    parser.add_argument(
        "--outlier-iqr", type=float, default=OUTLIER_IQR_MULTIPLIER,
        metavar="N",
        help="IQR multiplier for --outlier-method=iqr  [default: 3.0]",
    )
    parser.add_argument(
        "--outlier-mad", type=float, default=OUTLIER_MAD_THRESHOLD,
        metavar="N",
        help="Modified Z-score threshold for --outlier-method=mad  [default: 3.5]",
    )
    parser.add_argument(
        "--min-mt", type=float, default=OUTLIER_MIN_MT_MS,
        metavar="MS",
        help="Absolute minimum MT in ms; shorter trials are removed  [default: 50]",
    )
    parser.add_argument(
        "--max-mt", type=float, default=OUTLIER_MAX_MT_MS,
        metavar="MS",
        help="Absolute maximum MT in ms; longer trials are removed  [default: 30000]",
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Cursor Technique Study — Statistical Analysis")
    print(f"  Input:            {args.data_dir}")
    print(f"  Output:           {out_dir}")
    print(f"  α =               {ALPHA}")
    print(f"  Outlier method:   {args.outlier_method}")
    if args.outlier_method == "sd":
        print(f"  Outlier SD:       ±{args.outlier_sd}")
    elif args.outlier_method == "iqr":
        print(f"  Outlier IQR mult: {args.outlier_iqr}×")
    elif args.outlier_method == "mad":
        print(f"  Outlier MAD thr:  {args.outlier_mad}")
    print(f"  MT bounds:        [{args.min_mt:.0f} ms, {args.max_mt:.0f} ms]")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n► Loading session data...")
    cdf, tdf, paths = load_directory(args.data_dir)
    n_p = cdf["participant"].nunique()
    print(f"\n  {len(paths)} file(s) · {n_p} participant(s) · "
          f"{len(cdf)} condition records · {len(tdf)} trial records")

    if n_p == 0:
        sys.exit("[ERROR] No usable participant data found.")

    # ── Outlier removal ───────────────────────────────────────────────────────
    # Must come before all analyses so every downstream function sees clean data.
    print("\n► Outlier removal...")
    tdf_orig = tdf.copy()
    cdf_orig = cdf.copy()

    tdf, trial_outlier_report = remove_trial_outliers(
        tdf,
        method     = args.outlier_method,
        sd_thresh  = args.outlier_sd,
        iqr_mult   = args.outlier_iqr,
        mad_thresh = args.outlier_mad,
        min_mt     = args.min_mt,
        max_mt     = args.max_mt,
        out_dir    = out_dir,
    )

    cdf, cond_outlier_report = remove_condition_outliers(
        cdf,
        method     = args.outlier_method,
        sd_thresh  = args.outlier_sd,
        iqr_mult   = args.outlier_iqr,
        mad_thresh = args.outlier_mad,
        out_dir    = out_dir,
    )

    console_outlier_summary(
        trial_outlier_report, cond_outlier_report,
        n_trials_orig  = len(tdf_orig),
        n_trials_clean = len(tdf),
        method         = args.outlier_method,
    )

    # Save the cleaned data tables for reference
    cdf.to_csv(out_dir / "data_conditions_clean.csv", index=False)
    tdf.to_csv(out_dir / "data_trials_clean.csv",     index=False)

    print(f"\n  Retained: {len(tdf):,} trials · {len(cdf)} condition records "
          f"(after outlier removal)")

    # ── Descriptive stats ─────────────────────────────────────────────────────
    print("\n► Descriptive statistics...")
    descriptive_stats(cdf, out_dir)

    # ── Normality ─────────────────────────────────────────────────────────────
    print("\n► Shapiro-Wilk normality tests...")
    normality_tests(cdf, out_dir)

    # ── 3-way RM ANOVA ────────────────────────────────────────────────────────
    print("\n► 3-way repeated-measures ANOVA...")
    anova_results = run_all_anovas(cdf, out_dir)

    # ── Sphericity ────────────────────────────────────────────────────────────
    sphericity_summary(anova_results, out_dir)

    # ── Effect sizes ──────────────────────────────────────────────────────────
    print("\n► Effect sizes (partial η²)...")
    eta2_df = extract_effect_sizes(anova_results, out_dir)

    # ── Post-hoc ──────────────────────────────────────────────────────────────
    print("\n► Post-hoc pairwise tests (Bonferroni)...")
    posthoc_results = run_all_posthoc(cdf, anova_results, out_dir)

    # ── Throughput concordance ────────────────────────────────────────────────
    print("\n► Throughput measure concordance...")
    throughput_concordance(cdf, out_dir)

    # ── Fitts regression ──────────────────────────────────────────────────────
    print("\n► Fitts' Law regression (trial-level)...")
    elig, reg_tech, reg_tm = fitts_regression(tdf, out_dir)

    # ── Model comparison ──────────────────────────────────────────────────────
    print("\n► Fitts vs Hoffmann model comparison...")
    model_comparison_stats(cdf, out_dir)

    # ── Hypothesis tests ──────────────────────────────────────────────────────
    print("\n► Testing H1 (cursor technique)...")
    h1_results = test_h1_cursor_technique(cdf, anova_results, out_dir)

    print("\n► Testing H2 (Fitts' Law validity)...")
    h2_results = test_h2_fitts_law(cdf, elig, out_dir)
    if _TTEST_LOG:
        ttest_csv = out_dir / "all_ttests.csv"
        pd.DataFrame(_TTEST_LOG).to_csv(ttest_csv, index=False)
        print(f"  t-test log: {len(_TTEST_LOG)} tests → {ttest_csv.name}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n► Generating figures...")

    # fig00 uses both pre- and post-cleaning DataFrames
    fig00_outlier_removal(tdf_orig, tdf, cdf_orig, cdf,
                          trial_outlier_report, args.outlier_method, out_dir)

    fig1_boxplots(cdf, out_dir)
    fig2_main_effects(cdf, out_dir)
    fig3_interactions_2way(cdf, out_dir)
    fig4_heatmaps_3way(cdf, out_dir)

    if len(elig) > 10:
        fig5_fitts_scatter(elig, reg_tech, out_dir)
        fig6_fitts_grid(elig, reg_tm, out_dir)
    else:
        print("    (skipping Fitts scatter: insufficient trial-level data)")

    fig7_tp_violins(cdf, out_dir)
    fig8_error_precision(cdf, out_dir)
    fig9_effect_sizes(eta2_df, out_dir)
    fig10_anova_table(anova_results, out_dir)
    fig11_posthoc(cdf, posthoc_results, out_dir)
    fig12_tp_concordance(cdf, out_dir)
    fig13_participant_profiles(cdf, out_dir)
    fig14_model_comparison(cdf, out_dir)
    fig15_diagnostics(cdf, out_dir)
    fig16_hypothesis_tests(cdf, h1_results, h2_results, elig, out_dir)

    # ── Console summaries ─────────────────────────────────────────────────────
    console_anova_summary(anova_results)
    console_posthoc_summary(posthoc_results)
    console_hypothesis_summary(h1_results, h2_results, anova_results)

    # ── Final tally ───────────────────────────────────────────────────────────
    n_figs = len(list(out_dir.glob("*.png")))
    n_csvs = len(list(out_dir.glob("*.csv")))
    print(f"\n{'='*60}")
    print(f"  Done.  Saved to: {out_dir}/")
    print(f"  Figures : {n_figs} PNG files")
    print(f"  Tables  : {n_csvs} CSV files")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()