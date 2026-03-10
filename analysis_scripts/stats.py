#!/usr/bin/env python3
"""
Cursor Technique Study — Statistical Analysis Pipeline
======================================================
Usage:
    python analyse.py <data_directory> [--out <output_directory>]

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

def _gg_epsilon(wide_matrix: np.ndarray) -> float:
    """
    Greenhouse-Geisser epsilon from a participants × levels matrix.
    Formula: ε = tr(S̃)² / [(k-1) · tr(S̃²)]
    where S̃ is the doubly-centred covariance matrix.
    """
    S = np.cov(wide_matrix.T)
    k = S.shape[0]
    if k <= 2:
        return 1.0                        # df = 1 → sphericity trivially holds
    row_m  = S.mean(axis=1, keepdims=True)
    col_m  = S.mean(axis=0, keepdims=True)
    grand  = S.mean()
    S_dc   = S - row_m - col_m + grand   # doubly-centred
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

        # Regression model stats
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

        # Individual trial records (for Fitts regression)
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

    if not paths:                                        # fallback
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
    """Condition-cell means, SDs, medians; save as CSV files."""
    dvs = list(PRIMARY_DVS)

    # Grand summary across all conditions
    grand = cdf[dvs].describe().T.round(3)
    grand.to_csv(out_dir / "desc_grand.csv")

    # Per-factor marginal summaries
    for factor in ("technique", "movement", "clustering"):
        grp = cdf.groupby(factor)[dvs].agg(["mean", "std", "median", "sem"]).round(3)
        grp.to_csv(out_dir / f"desc_by_{factor}.csv")

    # All two-way cell means
    for f1, f2 in [("technique", "movement"),
                   ("technique", "clustering"),
                   ("movement",  "clustering")]:
        grp = cdf.groupby([f1, f2])[dvs].mean().round(3)
        grp.to_csv(out_dir / f"desc_{f1}_x_{f2}.csv")

    # Full 3-way cell means
    cdf.groupby(["technique", "movement", "clustering"])[dvs].mean().round(3)\
       .to_csv(out_dir / "desc_3way_cell_means.csv")

    print("  Descriptive CSVs written.")
    return grand


# ─────────────────────────────────────────────────────────────────────────────
# NORMALITY TESTING  (Shapiro-Wilk per DV per condition cell)
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
    """
    Keep only participants with all 27 within-subjects cells filled.
    Returns filtered DataFrame and number dropped.
    """
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

    # statsmodels requires plain strings / object dtype for within-factors
    for col in WITHIN:
        df[col] = df[col].astype(str)

    # ── Run ANOVA ─────────────────────────────────────────────────────────────
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

    # ── Normalise source names to match the rest of the script ────────────────
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

    # ── Partial eta²  η²p = F·df₁ / (F·df₁ + df₂) ───────────────────────────
    aov["np2"] = (aov["F"] * aov["ddof1"]) / \
                 (aov["F"] * aov["ddof1"] + aov["ddof2"])

    # ── GG epsilon per main-effect factor  ───────────────────────────────────
    eps_main = {f: _gg_epsilon_for_factor(df, dv, "participant", f)
                for f in WITHIN}

    def _combined_eps(source_name: str) -> float:
        """
        Main effect  → marginal epsilon.
        Interaction  → product of marginal epsilons (Box 1954 approximation).
        Bounded to [1/(k-1), 1].
        """
        parts = [p.strip() for p in source_name.split("*")]
        eps   = float(np.prod([eps_main.get(p, 1.0) for p in parts]))
        return min(max(eps, 0.0), 1.0)

    # ── Apply GG correction ───────────────────────────────────────────────────
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

    # Concatenated table
    if results:
        pd.concat(list(results.values())).to_csv(out_dir / "anova_all_dvs.csv", index=False)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAUCHLY SPHERICITY SUMMARY
# (pingouin rm_anova reports eps; eps < 1 means sphericity violated)
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
    """Return list of factor names with significant main or 2-way effects."""
    if aov is None:
        return []
    p_col = "p-GG-corr" if "p-GG-corr" in aov.columns else "p-unc"
    sig   = aov[aov[p_col] < ALPHA]["Source"].tolist()
    return sig


def posthoc_for_dv(cdf, dv, aov, out_dir):
    results = {}
    df = cdf[["participant", "technique", "movement", "clustering", dv]].dropna()
    df, _ = _ensure_complete_participants(df, dv)
    if df["participant"].nunique() < 3:
        return results

    sig_effects = _significant_main_effects(aov)

    single_factors = {
        "technique":  "technique",
        "movement":   "movement",
        "clustering": "clustering",
    }

    for effect in sig_effects:
        # ── Main effects ──────────────────────────────────────────────────
        if effect in single_factors:
            factor = single_factors[effect]
            try:
                ph = pg.pairwise_tests(
                    data       = df,
                    dv         = dv,
                    within     = factor,
                    subject    = "participant",
                    padjust    = "bonf",
                    parametric = True,
                )
                ph.insert(0, "dv",     dv)
                ph.insert(1, "effect", effect)
                results[effect] = ph
                ph.to_csv(out_dir / f"posthoc_{dv}_{factor}.csv", index=False)
            except Exception:
                pass

        # ── 2-way interactions ────────────────────────────────────────────
        elif " * " in effect:
            parts = [p.strip() for p in effect.split("*")]
            if len(parts) == 2 and all(p in single_factors for p in parts):
                try:
                    ph = pg.pairwise_tests(
                        data       = df,
                        dv         = dv,
                        within     = parts,
                        subject    = "participant",
                        padjust    = "bonf",
                        parametric = True,
                    )
                    ph.insert(0, "dv",     dv)
                    ph.insert(1, "effect", effect)
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
    """
    Per-technique and per-technique × movement OLS regression of
    MT ~ log2(A/W + 1).  Excludes trial 1 of each condition (no amplitude).
    """
    elig = tdf.dropna(subset=["amplitude_px", "time_ms", "targetRadius"]).copy()
    elig = elig[(elig.amplitude_px > 0) & (elig.time_ms > 0) & (elig.targetRadius > 0)]
    elig = elig[elig.trial.fillna(0) > 1]   # drop first trial (no predecessor)

    elig["ID"]   = np.log2(elig.amplitude_px / (2 * elig.targetRadius) + 1)
    elig["MT_s"] = elig.time_ms / 1000.0

    # Also compute Hoffmann-corrected ID where applicable
    elig["vt"] = np.abs(
        elig.targetVx.fillna(0) * (elig.targetX - elig.prevTargetX.fillna(elig.targetX)) /
        np.maximum(np.sqrt((elig.targetX - elig.prevTargetX.fillna(elig.targetX))**2 +
                           (elig.targetY - elig.prevTargetY.fillna(elig.targetY))**2), 1) +
        elig.targetVy.fillna(0) * (elig.targetY - elig.prevTargetY.fillna(elig.targetY)) /
        np.maximum(np.sqrt((elig.targetX - elig.prevTargetX.fillna(elig.targetX))**2 +
                           (elig.targetY - elig.prevTargetY.fillna(elig.targetY))**2), 1)
    )
    W_nom = 2 * elig.targetRadius
    W_eff = np.maximum(W_nom - elig.vt * elig.MT_s, W_nom * 0.1)
    elig["ID_hoffmann"] = np.log2(elig.amplitude_px / W_eff + 1)

    results_tech  = {}
    results_tm    = {}

    # ── Per-technique ─────────────────────────────────────────────────────────
    for tech in TECHNIQUES:
        sub = elig[elig.technique == tech]
        if len(sub) < 5:
            continue
        for id_col, model_name in [("ID", "fitts"), ("ID_hoffmann", "hoffmann")]:
            x, y = sub[id_col].values, sub["time_ms"].values
            slope, intercept, r, p_r, se = stats.linregress(x, y)
            key = f"{tech}_{model_name}"
            results_tech[key] = {
                "technique": tech, "model": model_name,
                "n": len(sub), "slope_ms": round(slope, 2),
                "intercept_ms": round(intercept, 2),
                "r": round(r, 4), "r2": round(r**2, 4),
                "p": round(p_r, 6),
                "tp_bps": round(1000 / slope, 3) if slope > 0 else np.nan,
                "se": round(se, 4),
            }

    pd.DataFrame(results_tech).T.to_csv(
        out_dir / "fitts_regression_by_technique.csv")

    # ── Per-technique × movement ──────────────────────────────────────────────
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

    pd.DataFrame(results_tm).T.to_csv(
        out_dir / "fitts_regression_by_tech_movement.csv")

    return elig, results_tech, results_tm


# ─────────────────────────────────────────────────────────────────────────────
# THROUGHPUT MEASURE CONCORDANCE
# ─────────────────────────────────────────────────────────────────────────────

def throughput_concordance(cdf, out_dir):
    tp_cols = [
        "throughput_shannon_nominal_bps",
        "throughput_shannon_effective_bps",
        "throughput_hoffmann_bps",
    ]
    labels = ["TP-Shannon-Nominal", "TP-Shannon-Effective", "TP-Hoffmann"]
    sub = cdf[tp_cols].dropna()

    # Pearson r matrix
    sub.corr().round(4).to_csv(out_dir / "throughput_pearson_matrix.csv")

    # Pairwise: Pearson r, ICC(2,1), Bland-Altman stats
    pair_rows = []
    for (c1, l1), (c2, l2) in combinations(zip(tp_cols, labels), 2):
        x, y = sub[c1].values, sub[c2].values
        r, p_r = stats.pearsonr(x, y)
        diff   = x - y
        pair_rows.append({
            "pair":    f"{l1} vs {l2}",
            "pearson_r": round(r, 4),
            "p_pearson": round(p_r, 6),
            "mean_diff": round(np.mean(diff), 4),
            "sd_diff":   round(np.std(diff, ddof=1), 4),
            "loa_lower": round(np.mean(diff) - 1.96 * np.std(diff, ddof=1), 4),
            "loa_upper": round(np.mean(diff) + 1.96 * np.std(diff, ddof=1), 4),
        })
    pd.DataFrame(pair_rows).to_csv(
        out_dir / "throughput_concordance_bland_altman.csv", index=False)

    return pd.DataFrame(pair_rows)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL COMPARISON  (Fitts vs Hoffmann  per condition)
# ─────────────────────────────────────────────────────────────────────────────

def model_comparison_stats(cdf, out_dir):
    """
    Per-condition Δr² and ΔAIC; paired t-test and Wilcoxon across conditions.
    """
    df = cdf[["fitts_regression_r2",    "hoffmann_regression_r2",
              "fitts_regression_aic",   "hoffmann_regression_aic",
              "delta_r2", "delta_aic",
              "technique", "movement",  "clustering"]].dropna(
                  subset=["delta_r2", "delta_aic"])

    rows = []
    for var, col in [("Δr²",  "delta_r2"), ("ΔAIC", "delta_aic")]:
        vals = df[col].dropna().values
        if len(vals) < 3:
            continue
        t_stat, t_p = stats.ttest_1samp(vals, 0.0)
        w_stat, w_p = stats.wilcoxon(vals)
        rows.append({
            "metric":       var,
            "n_conditions": len(vals),
            "mean":         round(np.mean(vals), 4),
            "sd":           round(np.std(vals, ddof=1), 4),
            "median":       round(np.median(vals), 4),
            "t_vs_0":       round(t_stat, 3),
            "p_t":          round(t_p, 4),
            "W_wilcoxon":   round(w_stat, 3),
            "p_wilcoxon":   round(w_p, 4),
            "sig_at_05":    t_p < ALPHA,
        })
    pd.DataFrame(rows).to_csv(out_dir / "model_comparison_delta_stats.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

# ── Figure 1: Grand descriptive box plots ────────────────────────────────────

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


# ── Figure 2: Main-effect mean ± SEM bar charts ──────────────────────────────

def fig2_main_effects(cdf, out_dir):
    dvs     = list(PRIMARY_DVS.items())[:6]           # top 6 DVs
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


# ── Figure 3: 2-way interaction line plots ───────────────────────────────────

def fig3_interactions_2way(cdf, out_dir):
    tp_dvs = [
        ("throughput_shannon_nominal_bps",   "TP Shannon Nominal (bits/s)"),
        ("throughput_shannon_effective_bps", "TP Shannon Effective (bits/s)"),
        ("throughput_hoffmann_bps",          "TP Hoffmann (bits/s)"),
        ("avgTime_ms",                       "Mean MT (ms)"),
    ]
    interactions = [
        ("technique",  "movement",   TECHNIQUES, MOVEMENTS,
         TECH_COLORS,  "Technique"),
        ("technique",  "clustering", TECHNIQUES, DENSITIES,
         TECH_COLORS,  "Technique"),
        ("movement",   "clustering", MOVEMENTS,  DENSITIES,
         MOV_COLORS,   "Motion"),
    ]

    fig, axes = plt.subplots(len(tp_dvs), 3, figsize=(16, 4.5 * len(tp_dvs)))

    for row_i, (dv, label) in enumerate(tp_dvs):
        for col_i, (f1, f2, lev1, lev2, col_map, leg_title) in enumerate(interactions):
            ax = axes[row_i][col_i]
            for l1 in lev1:
                sub  = cdf[cdf[f1] == l1]
                means = [sub[sub[f2] == l2][dv].mean() for l2 in lev2]
                sems  = [sub[sub[f2] == l2][dv].sem()  for l2 in lev2]
                c    = col_map.get(l1, "#888")
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


# ── Figure 4: 3-way heatmaps ─────────────────────────────────────────────────

def fig4_heatmaps_3way(cdf, out_dir):
    dvs_plot = [
        ("throughput_shannon_nominal_bps",   "TP Shannon Nominal (bits/s)"),
        ("throughput_hoffmann_bps",           "TP Hoffmann (bits/s)"),
        ("avgTime_ms",                        "Mean MT (ms)"),
        ("precisionRate_percent",             "Precision Rate (%)"),
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
            ax.set_xticks(range(len(TECHNIQUES)))
            ax.set_xticklabels(TECHNIQUES, fontsize=8)
            ax.set_yticks(range(len(MOVEMENTS)))
            ax.set_yticklabels(MOVEMENTS, fontsize=8)
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


# ── Figure 5: Fitts' Law scatter + OLS per technique ─────────────────────────

def fig5_fitts_scatter(elig, reg_tech, out_dir):
    if elig.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, tech in zip(axes, TECHNIQUES):
        sub = elig[elig.technique == tech]
        for mov in MOVEMENTS:
            sm  = sub[sub.movement == mov]
            ax.scatter(sm["ID"], sm["time_ms"],
                       color=MOV_COLORS[mov], alpha=0.25, s=10, label=mov)

        key = f"{tech}_fitts"
        if key in reg_tech:
            r   = reg_tech[key]
            xr  = np.linspace(sub["ID"].min(), sub["ID"].max(), 200)
            ax.plot(xr, r["slope_ms"] * xr + r["intercept_ms"],
                    "k-", linewidth=2.2,
                    label=f"OLS  r²={r['r2']}  TP={r['tp_bps']} b/s")

        ax.set_title(f"{tech}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Index of Difficulty  ID (bits)", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Movement Time  MT (ms)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.6)
        ax.grid(alpha=0.3)

    fig.suptitle("Fitts' Law Regression  MT ~ log₂(A/W + 1)  by Technique "
                 "(colour = motion)", fontsize=11)
    savefig(fig, out_dir / "fig05_fitts_regression.png")


# ── Figure 6: Fitts' regression  3×3 grid (technique × movement) ─────────────

def fig6_fitts_grid(elig, reg_tm, out_dir):
    if elig.empty:
        return

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharey=False)

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


# ── Figure 7: Throughput violin + strip ──────────────────────────────────────

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
            ax = axes[row_i][col_i]
            data_by_level = [cdf[cdf[factor] == l][dv].dropna().values
                             for l in levels]

            parts = ax.violinplot(data_by_level, positions=range(len(levels)),
                                  showmeans=True, showmedians=True,
                                  showextrema=True, widths=0.7)
            for pc, l in zip(parts["bodies"], levels):
                pc.set_facecolor(col_map.get(l, "#aaa"))
                pc.set_alpha(0.55)

            for pos, (vals, l) in enumerate(zip(data_by_level, levels)):
                jitter = np.random.uniform(-0.08, 0.08, len(vals))
                ax.scatter(pos + jitter, vals,
                           color=col_map.get(l, "#aaa"),
                           alpha=0.55, s=16, zorder=3)

            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels, fontsize=8)
            if col_i == 0:
                ax.set_ylabel(f"{dv_label} (bits/s)", fontsize=8)
            ax.set_title(f"{dv_label}\n× {factor}", fontsize=8)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Throughput Distributions  (Violin + Strip)", fontsize=12)
    savefig(fig, out_dir / "fig07_throughput_violins.png")


# ── Figure 8: Precision and error analysis ───────────────────────────────────

def fig8_error_precision(cdf, out_dir):
    factors = [
        ("technique",  TECHNIQUES, TECH_COLORS),
        ("movement",   MOVEMENTS,  MOV_COLORS),
        ("clustering", DENSITIES,  DENS_COLORS),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))

    dvs_acc = [
        ("precisionRate_percent",    "Precision Rate (%)"),
        ("avgErrorClicks",           "Mean Error Clicks / Trial"),
        ("avgNormalizedDistance",    "Mean Normalised Distance"),
    ]

    for row_i, (dv, label) in enumerate(dvs_acc):
        for col_i, (factor, levels, col_map) in enumerate(factors):
            ax    = axes[row_i][col_i]
            means = [cdf[cdf[factor] == l][dv].mean() for l in levels]
            sems  = [cdf[cdf[factor] == l][dv].sem()  for l in levels]
            colors_b = [col_map.get(l, "#aaa") for l in levels]

            ax.bar(range(len(levels)), means, yerr=sems, capsize=4,
                   color=colors_b, edgecolor="black", linewidth=0.6,
                   alpha=0.78)
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


# ── Figure 9: Effect size forest plot ────────────────────────────────────────

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

    fig, axes = plt.subplots(1, len(focus_dvs),
                             figsize=(4.2 * len(focus_dvs), 6), sharey=True)

    effect_labels = {
        "technique":                        "Technique (A)",
        "movement":                         "Movement (B)",
        "clustering":                       "Clustering (C)",
        "technique * movement":             "A × B",
        "technique * clustering":           "A × C",
        "movement * clustering":            "B × C",
        "technique * movement * clustering":"A × B × C",
    }

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
            ax.text(eta + 0.002, y, f"{eta:.3f}",
                    va="center", ha="left", fontsize=7.5)

        # Effect size thresholds
        ax.axvline(0.01, color="steelblue",   linewidth=0.8,
                   linestyle=":", label="small (.01)")
        ax.axvline(0.06, color="darkorange",  linewidth=0.8,
                   linestyle=":", label="medium (.06)")
        ax.axvline(0.14, color="darkred",     linewidth=0.8,
                   linestyle=":", label="large (.14)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_out, fontsize=8)
        ax.set_xlabel("Partial η²", fontsize=8)
        ax.set_title(short_label, fontsize=9, fontweight="bold")
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=0.25)

    axes[-1].legend(fontsize=7, loc="lower right")
    red_p  = mpatches.Patch(color="#c0392b", label="Significant (p<.05)")
    grey_p = mpatches.Patch(color="#95a5a6", label="Non-significant")
    axes[0].legend(handles=[red_p, grey_p], fontsize=7, loc="lower right")

    fig.suptitle("Effect Sizes  (Partial η²)  ·  Dotted lines = Cohen thresholds",
                 fontsize=11)
    savefig(fig, out_dir / "fig09_effect_sizes.png")


# ── Figure 10: ANOVA results summary table ───────────────────────────────────

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
    # For each DV × effect: "F(df1,df2) = x.xx, p = .xxx *"
    cell_data = []

    for src in ANOVA_SOURCE_ORDER:
        row_vals = [src]
        for dv in focus_dvs:
            aov = anova_results.get(dv)
            if aov is None:
                row_vals.append("—")
                continue
            match = aov[aov["Source"].str.strip() == src]
            if match.empty:
                row_vals.append("—")
                continue
            r    = match.iloc[0]
            F    = safe_float(r.get("F"))
            df1  = r.get("ddof1", r.get("DF", ""))
            df2  = r.get("ddof2", "")
            p_col= "p-GG-corr" if "p-GG-corr" in aov.columns else "p-unc"
            pval = safe_float(r.get(p_col))
            eta2 = safe_float(r.get("np2") or r.get("ng2"))
            sig  = "*" if pval < ALPHA else ""

            df_str = (f"({int(df1)},{int(df2)})"
                      if isinstance(df1, (float,int,np.floating)) and
                         isinstance(df2, (float,int,np.floating)) else "")
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
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 2.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f4f6f7")
        else:
            cell.set_facecolor("#ffffff")
        # Highlight significant cells
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


# ── Figure 11: Post-hoc significance brackets ────────────────────────────────

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

    fig, axes = plt.subplots(len(focus_dvs), 3,
                             figsize=(16, 4.5 * len(focus_dvs)))

    for row_i, (dv, label) in enumerate(focus_dvs):
        for col_i, (factor, levels, col_map) in enumerate(factors):
            ax = axes[row_i][col_i]
            ph_dict = posthoc_results.get(dv, {})
            ph      = ph_dict.get(factor)

            means  = [cdf[cdf[factor] == l][dv].mean() for l in levels]
            sems   = [cdf[cdf[factor] == l][dv].sem()  for l in levels]
            colors_b = [col_map.get(l, "#aaa") for l in levels]

            ax.bar(range(len(levels)), means, yerr=sems, capsize=4,
                   color=colors_b, edgecolor="black", linewidth=0.6, alpha=0.78)
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels, fontsize=8)
            if col_i == 0:
                ax.set_ylabel(label, fontsize=7)
            ax.set_title(f"{label[:20]}\n× {factor}", fontsize=7)
            ax.grid(axis="y", alpha=0.3)

            if ph is None or ph.empty:
                continue

            # Significance brackets
            p_col = next((c for c in ["p-corr", "p-bonf", "p-unc"]
                          if c in ph.columns), None)
            if p_col is None:
                continue

            valid_top = [m + s for m, s in zip(means, sems)
                         if not (np.isnan(m) or np.isnan(s))]
            if not valid_top:
                continue

            y_base   = max(valid_top) * 1.05
            y_step   = max(valid_top) * 0.055
            bracket_h = max(valid_top) * 0.025
            current_y = y_base

            for _, ph_row in ph.iterrows():
                a     = ph_row.get("A")
                b     = ph_row.get("B")
                pval  = safe_float(ph_row.get(p_col, 1.0))
                if a not in levels or b not in levels:
                    continue
                if np.isnan(pval) or pval >= ALPHA * 3:  # show only if < 0.15
                    continue
                xa, xb = levels.index(a), levels.index(b)
                sig_str = ("***" if pval < 0.001 else
                           "**"  if pval < 0.01  else
                           "*"   if pval < ALPHA  else "ns")
                ax.plot([xa, xa, xb, xb],
                        [current_y, current_y + bracket_h,
                         current_y + bracket_h, current_y],
                        "k-", linewidth=0.9)
                ax.text((xa + xb) / 2, current_y + bracket_h * 1.1,
                        sig_str, ha="center", va="bottom", fontsize=8.5)
                current_y += y_step

    fig.suptitle(
        "Post-hoc Pairwise Comparisons  (Bonferroni  ·  * p<.05  ** p<.01  *** p<.001)",
        fontsize=10)
    savefig(fig, out_dir / "fig11_posthoc_brackets.png")


# ── Figure 12: Throughput pairwise scatter + Bland-Altman ────────────────────

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

        # Scatter + regression
        ax = axes[0][col_i]
        ax.scatter(x, y, alpha=0.4, s=18, color="#3498db", edgecolors="none")
        m, b, r, _, _ = stats.linregress(x, y)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, m * xr + b, "r-", linewidth=1.5,
                label=f"r={r:.3f},  p={stats.pearsonr(x,y)[1]:.4f}")
        ax.plot(xr, xr, "k--", linewidth=1, alpha=0.4, label="identity")
        ax.set_xlabel(l1, fontsize=8); ax.set_ylabel(l2, fontsize=8)
        ax.set_title(f"Correlation", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Bland-Altman
        ax = axes[1][col_i]
        mean_ = (x + y) / 2
        diff_ = x - y
        md    = np.mean(diff_)
        sd    = np.std(diff_, ddof=1)
        ax.scatter(mean_, diff_, alpha=0.4, s=18, color="#2ecc71", edgecolors="none")
        ax.axhline(md,           color="red",    linewidth=1.5, label=f"Mean diff = {md:.3f}")
        ax.axhline(md + 1.96*sd, color="orange", linewidth=1, linestyle="--",
                   label=f"+1.96SD = {md+1.96*sd:.3f}")
        ax.axhline(md - 1.96*sd, color="orange", linewidth=1, linestyle="--",
                   label=f"−1.96SD = {md-1.96*sd:.3f}")
        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax.set_xlabel("Mean of measures (bits/s)", fontsize=8)
        ax.set_ylabel("Difference (bits/s)", fontsize=8)
        ax.set_title(f"Bland-Altman: {l1[:15]} − {l2[:15]}", fontsize=8)
        ax.legend(fontsize=6.5); ax.grid(alpha=0.3)

    fig.suptitle("Throughput Measure Concordance  (Scatter + Bland-Altman)", fontsize=11)
    savefig(fig, out_dir / "fig12_tp_concordance.png")


# ── Figure 13: Individual participant profiles ───────────────────────────────

def fig13_participant_profiles(cdf, out_dir):
    dv     = "throughput_shannon_nominal_bps"
    pids   = sorted(cdf["participant"].unique())
    if len(pids) < 2:
        return

    factors = [
        ("technique",  TECHNIQUES),
        ("movement",   MOVEMENTS),
        ("clustering", DENSITIES),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap_pid  = plt.cm.tab20

    for ax, (factor, levels) in zip(axes, factors):
        for k, pid in enumerate(pids):
            sub   = cdf[cdf.participant == pid]
            means = [sub[sub[factor] == l][dv].mean() for l in levels]
            ax.plot(range(len(levels)), means, "o-",
                    color=cmap_pid(k / max(len(pids) - 1, 1)),
                    alpha=0.55, linewidth=1.2, markersize=4,
                    label=f"P{pid}")

        grand = [cdf[cdf[factor] == l][dv].mean() for l in levels]
        ax.plot(range(len(levels)), grand, "k-o",
                linewidth=2.8, markersize=8, label="Grand mean", zorder=10)

        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, fontsize=9)
        ax.set_title(f"TP Shannon Nominal × {factor}", fontsize=9)
        ax.set_ylabel("bits/s", fontsize=8)
        ax.legend(fontsize=6, loc="best", framealpha=0.6,
                  ncol=max(1, len(pids) // 8))
        ax.grid(alpha=0.3)

    fig.suptitle("Individual Participant Profiles  (bold = grand mean)", fontsize=11)
    savefig(fig, out_dir / "fig13_participant_profiles.png")


# ── Figure 14: Model comparison (Δr², ΔAIC) ──────────────────────────────────

def fig14_model_comparison(cdf, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # r² distributions
    ax = axes[0]
    for col, lab, c in [("fitts_regression_r2",    "Fitts (nominal)",             "#3498db"),
                         ("hoffmann_regression_r2", "Hoffmann (velocity-corrected)","#e74c3c")]:
        vals = cdf[col].dropna().values
        if len(vals):
            ax.hist(vals, bins=14, alpha=0.55, label=lab, color=c, edgecolor="white")
    ax.set_xlabel("r² (OLS fit, per condition)", fontsize=9)
    ax.set_ylabel("Number of conditions", fontsize=9)
    ax.set_title("r² per Condition: Fitts vs Hoffmann", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Δr²
    ax = axes[1]
    dr2 = cdf["delta_r2"].dropna().values
    if len(dr2):
        ax.hist(dr2, bins=14, color="#27ae60", alpha=0.72, edgecolor="white")
        ax.axvline(0, color="red", linewidth=1.8, linestyle="--", label="Δr²= 0")
        ax.axvline(np.mean(dr2), color="black", linewidth=1.5, linestyle="-",
                   label=f"Mean = {np.mean(dr2):.4f}")
        t, p = stats.ttest_1samp(dr2, 0)
        ax.set_title(f"Δr²  (Hoffmann − Fitts)\nt = {t:.2f},  p = {p:.4f}", fontsize=9)
    ax.set_xlabel("Δr²", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.annotate("positive → Hoffmann fits better", xy=(0.5, 0.93),
                xycoords="axes fraction", ha="center", fontsize=7.5, color="gray")

    # ΔAIC
    ax = axes[2]
    daic = cdf["delta_aic"].dropna().values
    if len(daic):
        ax.hist(daic, bins=14, color="#e67e22", alpha=0.72, edgecolor="white")
        ax.axvline(0, color="red", linewidth=1.8, linestyle="--", label="ΔAIC = 0")
        ax.axvline(np.mean(daic), color="black", linewidth=1.5, linestyle="-",
                   label=f"Mean = {np.mean(daic):.2f}")
        t, p = stats.ttest_1samp(daic, 0)
        ax.set_title(f"ΔAIC  (Hoffmann − Fitts)\nt = {t:.2f},  p = {p:.4f}", fontsize=9)
    ax.set_xlabel("ΔAIC", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.annotate("negative → Hoffmann preferred", xy=(0.5, 0.93),
                xycoords="axes fraction", ha="center", fontsize=7.5, color="gray")

    fig.suptitle("Regression Model Comparison  (H₀: Δ = 0,  one-sample t-test)", fontsize=11)
    savefig(fig, out_dir / "fig14_model_comparison.png")


# ── Figure 15: Q-Q and residual diagnostics ──────────────────────────────────

def fig15_diagnostics(cdf, out_dir):
    """
    Q-Q normality plots for primary DVs aggregated across all conditions,
    plus residual plots for the Fitts regression.
    """
    dvs    = list(PRIMARY_DVS.items())[:6]
    fig, axes = plt.subplots(2, len(dvs), figsize=(4.5 * len(dvs), 9))

    for col_i, (dv, label) in enumerate(dvs):
        vals = cdf[dv].dropna().values

        # Q-Q plot
        ax = axes[0][col_i]
        if len(vals) >= 3:
            (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm")
            ax.scatter(osm, osr, s=12, color="#3498db", alpha=0.6)
            ax.plot(osm, slope * np.array(osm) + intercept,
                    "r-", linewidth=1.5, label=f"r={r:.3f}")
        ax.set_title(f"Q-Q: {label[:22]}", fontsize=8)
        ax.set_xlabel("Theoretical quantiles", fontsize=7)
        ax.set_ylabel("Sample quantiles", fontsize=7)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Standardised residuals from grand mean
        ax = axes[1][col_i]
        if len(vals) >= 3:
            fitted = np.full_like(vals, vals.mean())
            resid  = (vals - fitted) / (vals.std(ddof=1) + 1e-9)
            ax.scatter(fitted, resid, s=10, alpha=0.5, color="#2ecc71")
            ax.axhline(0, color="red", linewidth=1.2, linestyle="--")
            ax.axhline( 2, color="gray", linewidth=0.7, linestyle=":")
            ax.axhline(-2, color="gray", linewidth=0.7, linestyle=":")
        ax.set_title(f"Residuals: {label[:22]}", fontsize=8)
        ax.set_xlabel("Fitted (grand mean)", fontsize=7)
        ax.set_ylabel("Standardised residual", fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Diagnostic Plots  (Q-Q normality  +  standardised residuals)", fontsize=11)
    savefig(fig, out_dir / "fig15_diagnostics.png")


# ─────────────────────────────────────────────────────────────────────────────
# CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

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
        hdr = f"  {'Effect':<40} {'F':>8} {'df1':>5} {'df2':>7}  {'p (GG-corr)':>13}  {'η²p':>7}  sig"
        print(hdr)
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
            df1_s = f"{int(df1):5d}" if isinstance(df1, (float,int,np.floating)) else f"{str(df1):>5}"
            df2_s = f"{int(df2):7d}" if isinstance(df2, (float,int,np.floating)) else f"{str(df2):>7}"
            p_s   = f"{pval:13.4f}"  if np.isfinite(pval) else f"{'':>13}"
            e_s   = f"{eta2:7.3f}"   if np.isfinite(eta2) else f"{'':>7}"
            sig   = "  *" if (np.isfinite(pval) and pval < ALPHA) else "   "

            print(f"  {src:<40} {F_s} {df1_s} {df2_s}  {p_s}  {e_s} {sig}")

    print(f"\n{'='*88}\n")


def console_posthoc_summary(posthoc_results):
    print("\n  POST-HOC SUMMARY  (Bonferroni-corrected, significant pairs only)")
    print("  " + "─"*60)
    for dv, ph_dict in posthoc_results.items():
        label = PRIMARY_DVS.get(dv, dv)
        for effect, ph in ph_dict.items():
            if ph is None or ph.empty:
                continue
            p_col = next((c for c in ["p-corr","p-bonf","p-unc"] if c in ph.columns), None)
            if p_col is None:
                continue
            sig_rows = ph[ph[p_col] < ALPHA] if not ph.empty else ph
            if sig_rows.empty:
                continue
            print(f"\n  [{label[:30]}]  {effect}")
            for _, row in sig_rows.iterrows():
                A = row.get("A","?"); B = row.get("B","?")
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
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Cursor Technique Study — Statistical Analysis")
    print(f"  Input:   {args.data_dir}")
    print(f"  Output:  {out_dir}")
    print(f"  α =      {ALPHA}")
    print(f"{'='*60}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n► Loading session data...")
    cdf, tdf, paths = load_directory(args.data_dir)
    n_p = cdf["participant"].nunique()
    print(f"\n  {len(paths)} file(s) · {n_p} participant(s) · "
          f"{len(cdf)} condition records · {len(tdf)} trial records")

    if n_p == 0:
        sys.exit("[ERROR] No usable participant data found.")

    # ── Descriptive stats ─────────────────────────────────────────────────────
    print("\n► Descriptive statistics...")
    descriptive_stats(cdf, out_dir)

    # ── Normality ─────────────────────────────────────────────────────────────
    print("\n► Shapiro-Wilk normality tests...")
    normality_df = normality_tests(cdf, out_dir)

    # ── 3-way RM ANOVA ────────────────────────────────────────────────────────
    print("\n► 3-way repeated-measures ANOVA...")
    anova_results = run_all_anovas(cdf, out_dir)

    # ── Sphericity ────────────────────────────────────────────────────────────
    sph_df = sphericity_summary(anova_results, out_dir)

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

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n► Generating figures...")
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

    # ── Console summaries ─────────────────────────────────────────────────────
    console_anova_summary(anova_results)
    console_posthoc_summary(posthoc_results)

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