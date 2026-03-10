#!/usr/bin/env python3
"""
analysis.py — Cursor Technique Study · Full Analysis Pipeline
═════════════════════════════════════════════════════════════
Reads one or more study JSON files (auto-detected format) and produces:

  <output>/
  ├── data/
  │   ├── trials.csv          one row per trial, all numeric metrics
  │   ├── conditions.csv      aggregated per participant × condition
  │   └── summary.csv         per participant × technique
  ├── stats/
  │   ├── descriptives.csv    mean, SD, median, IQR  per tech × mov × clust
  │   ├── fitts.csv           OLS fit (a, b, R², p)  per technique
  │   └── pairwise_mt.csv     Welch t-tests + Cohen's d between techniques
  └── figures/
      ├── performance/
      │   ├── mt_distributions.png    violin + strip: MT by each IV
      │   ├── precision.png           click-inside-target rate
      │   └── throughput.png          Shannon TP violin + TP vs clustering
      ├── fitts/
      │   └── id_vs_mt.png            Fitts scatter + OLS per technique
      ├── interactions/
      │   └── interaction_plots.png   pointplots + error heatmap
      └── trajectories/
          ├── overview.png            XY · 3-D · closure · lateral  (4-panel)
          ├── closure_faceted.png     movement × clustering grid
          └── lateral_faceted.png     one row per technique

Accepted JSON formats
─────────────────────
  Block save  — top-level keys include "conditionsInBlock", "techBlock",
                "technique", "participant".  Produced after each technique
                block and on study completion.

  Session     — top-level key "conditions" with conditionStats present.
                (older / completion save variant)

  Trajectories — flat "trials" list.  (trajectory-only log)

  Multiple files / participants may be passed together.
  Duplicate trials (same participant × technique × movement × clustering ×
  trialNumber) are automatically deduplicated — so you can safely pass all
  block saves, the session save, and the trajectories file for every
  participant without double-counting.

Usage
─────
  python analysis.py                                         # synthetic demo
  python analysis.py P1_block1_POINT.json
  python analysis.py data/                                   # whole folder
  python analysis.py P*.json --output results/
  python analysis.py P1_block1_POINT.json --no-trajectories
  python analysis.py P1_block1_POINT.json --no-stats
  python analysis.py P1_block1_POINT.json --bins 60 --views xy closure lateral
"""

import argparse
import json
import math
import os
import random
import sys
import warnings
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  CONSTANTS & PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "BUBBLE": "#60a5fa",
    "POINT":  "#34d399",
    "AREA":   "#fb923c",
}
MOVEMENT_PALETTE   = {"STATIC": "#64748b", "SLOW": "#38bdf8", "FAST": "#f472b6"}
CLUSTERING_PALETTE = {"LOW": "#4ade80", "MEDIUM": "#facc15", "HIGH": "#f87171"}
FALLBACK_COLORS    = ["#a78bfa", "#f472b6", "#facc15", "#94a3b8",
                      "#38bdf8", "#4ade80", "#f87171"]

TECHNIQUE_ORDER  = ["BUBBLE", "POINT",  "AREA"]
MOVEMENT_ORDER   = ["STATIC", "SLOW",   "FAST"]
CLUSTERING_ORDER = ["LOW",    "MEDIUM", "HIGH"]

DARK = {
    "bg":      "#0f172a",
    "panel":   "#0d1a2e",
    "grid":    "#1e293b",
    "tick":    "#475569",
    "text":    "#94a3b8",
    "bright":  "#cbd5e1",
    "target":  "#f87171",
    "alpha_i": 0.055,
    "agg_lw":  2.4,
    "dot_n":   6,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  STYLE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _dark_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  DARK["bg"],
        "axes.facecolor":    DARK["panel"],
        "axes.edgecolor":    DARK["grid"],
        "axes.labelcolor":   DARK["text"],
        "xtick.color":       DARK["tick"],
        "ytick.color":       DARK["tick"],
        "text.color":        DARK["bright"],
        "grid.color":        DARK["grid"],
        "grid.linewidth":    0.6,
        "grid.linestyle":    "--",
        "legend.facecolor":  DARK["panel"],
        "legend.edgecolor":  DARK["grid"],
        "legend.labelcolor": DARK["bright"],
        "font.family":       "monospace",
        "font.size":         9,
        "axes.titlesize":    11,
        "axes.titlecolor":   DARK["bright"],
    })


def _light_style() -> None:
    plt.rcParams.update(matplotlib.rcParamsDefault)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        100,
    })


def _tech_color(tech: str, extra: dict = None) -> str:
    return PALETTE.get(tech) or (extra or {}).get(tech, "#94a3b8")


def _extra_colors(techs: list) -> dict:
    known = set(PALETTE)
    c = cycle(FALLBACK_COLORS)
    return {t: next(c) for t in techs if t not in known}


def _save(fig: plt.Figure, path: Path, dpi: int = 180, fc=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"dpi": dpi, "bbox_inches": "tight"}
    if fc:
        kw["facecolor"] = fc
    fig.savefig(path, **kw)
    plt.close(fig)
    print(f"    ✓ {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _strip(d: dict) -> dict:
    """Remove _README_* and _note metadata keys injected by the study app."""
    return {k: v for k, v in d.items()
            if not k.startswith("_README_") and k != "_note"}


def _detect(data: dict) -> str:
    """
    Return one of:
      'block'        — intermediate / completion block-save
                       (has 'conditionsInBlock' list)
      'session'      — older completion save
                       (has 'conditions' list with conditionStats)
      'trajectories' — flat trial list
                       (has 'trials' list)
      'unknown'
    """
    if "conditionsInBlock" in data and isinstance(data["conditionsInBlock"], list):
        return "block"
    if "conditions" in data and isinstance(data["conditions"], list):
        conds = data["conditions"]
        if conds and "conditionStats" in conds[0]:
            return "session"
    if "trials" in data and isinstance(data["trials"], list):
        return "trajectories"
    return "unknown"


def _shannon_id(amplitude_px, target_radius) -> float | None:
    """ID = log2(A / (2·r) + 1)  — Shannon formulation."""
    if amplitude_px and amplitude_px > 0 and target_radius and target_radius > 0:
        return math.log2(amplitude_px / (2 * target_radius) + 1)
    return None


# ── Format parsers ─────────────────────────────────────────────────────────────

def _parse_trial(t: dict, pid, ci, tech, mov, clust, tech_block) -> tuple[dict, dict | None]:
    """
    Parse a single trial dict into (trial_row, traj_row | None).
    traj_row is None when normalizedTrajectory is absent or null.
    """
    t   = _strip(t)
    amp = t.get("amplitude_px")
    rad = t.get("targetRadius", 0)

    trial_row = {
        "participant":          pid,
        "technique":            tech,
        "movement":             mov,
        "clustering":           clust,
        "conditionIndex":       ci,
        "techBlock":            tech_block,
        "trialNumber":          t.get("trialNumber"),
        "time_ms":              t.get("time_ms"),
        "errorClickCount":      t.get("errorClickCount", 0),
        "clickX":               t.get("clickX"),
        "clickY":               t.get("clickY"),
        "targetX":              t.get("targetX"),
        "targetY":              t.get("targetY"),
        "distanceToCenter_px":  t.get("distanceToCenter_px"),
        "normalizedDistance":   t.get("normalizedDistance"),
        "clickedInsideTarget":  t.get("clickedInsideTarget"),
        "amplitude_px":         amp,
        "targetRadius":         rad,
        "targetVx_px_per_s":    t.get("targetVx_px_per_s", 0),
        "targetVy_px_per_s":    t.get("targetVy_px_per_s", 0),
        "targetSpeed_px_per_s": t.get("targetSpeed_px_per_s", 0),
        "shannon_id_bits":      _shannon_id(amp, rad),
    }

    nt = t.get("normalizedTrajectory")
    cc = t.get("closureCurve")

    # normalizedTrajectory is explicitly null for trial 1 — treat as absent
    traj_row = None
    if isinstance(nt, list) and len(nt) >= 2:
        traj_row = {
            "participant":          pid,
            "conditionIndex":       ci,
            "technique":            tech,
            "movement":             mov,
            "clustering":           clust,
            "trialNumber":          t.get("trialNumber"),
            "normalizedTrajectory": nt,
            "closureCurve":         cc if isinstance(cc, list) else [],
        }
    elif isinstance(cc, list) and len(cc) >= 2:
        # No normalizedTrajectory but closureCurve present — keep for closure plots
        traj_row = {
            "participant":          pid,
            "conditionIndex":       ci,
            "technique":            tech,
            "movement":             mov,
            "clustering":           clust,
            "trialNumber":          t.get("trialNumber"),
            "normalizedTrajectory": None,   # will be skipped in XY / lateral
            "closureCurve":         cc,
        }

    return trial_row, traj_row


def _parse_block(data: dict) -> tuple[list, list]:
    """
    Parse a block-save JSON.
    Top-level keys of interest:
      participant, techBlock, technique
      conditionsInBlock[]:
        conditionIndex, technique, movement, clustering,
        techBlock, condInBlock, conditionStats, trials[]
    """
    data     = _strip(data)
    pid      = data.get("participant", 0)
    tb_top   = data.get("techBlock", -1)

    trial_rows, traj_rows = [], []

    for cond in data.get("conditionsInBlock", []):
        cond  = _strip(cond)
        tech  = cond.get("technique",  data.get("technique", "UNKNOWN"))
        mov   = cond.get("movement",   "UNKNOWN")
        clust = cond.get("clustering", "UNKNOWN")
        ci    = cond.get("conditionIndex", -1)
        tb    = cond.get("techBlock", tb_top)

        for t in cond.get("trials", []):
            if not isinstance(t, dict):
                continue
            tr, tj = _parse_trial(t, pid, ci, tech, mov, clust, tb)
            trial_rows.append(tr)
            if tj is not None:
                traj_rows.append(tj)

    return trial_rows, traj_rows


def _parse_session(data: dict) -> tuple[list, list]:
    """Parse an older session/completion-save JSON (key: 'conditions')."""
    data     = _strip(data)
    pid      = data.get("participant", 0)
    trial_rows, traj_rows = [], []

    for cond in data.get("conditions", []):
        cond  = _strip(cond)
        tech  = cond.get("technique",  "UNKNOWN")
        mov   = cond.get("movement",   "UNKNOWN")
        clust = cond.get("clustering", "UNKNOWN")
        ci    = cond.get("conditionIndex", -1)
        tb    = cond.get("techBlock", -1)

        for t in cond.get("trials", []):
            if not isinstance(t, dict):
                continue
            tr, tj = _parse_trial(t, pid, ci, tech, mov, clust, tb)
            trial_rows.append(tr)
            if tj is not None:
                traj_rows.append(tj)

    return trial_rows, traj_rows


def _parse_trajectories(data: dict) -> tuple[list, list]:
    """Parse a flat trajectories JSON (key: 'trials')."""
    data     = _strip(data)
    pid      = data.get("participant", 0)
    trial_rows, traj_rows = [], []

    for t in data.get("trials", []):
        if not isinstance(t, dict):
            continue
        t     = _strip(t)
        tech  = t.get("technique",  "UNKNOWN")
        mov   = t.get("movement",   "UNKNOWN")
        clust = t.get("clustering", "UNKNOWN")
        ci    = t.get("conditionIndex", -1)
        tb    = t.get("techBlock", -1)

        tr, tj = _parse_trial(t, pid, ci, tech, mov, clust, tb)
        trial_rows.append(tr)
        if tj is not None:
            traj_rows.append(tj)

    return trial_rows, traj_rows


# ── Deduplication ──────────────────────────────────────────────────────────────

def _trial_key(row: dict) -> tuple:
    """
    Unique identity for a trial.
    Uses conditionIndex when available (most reliable); falls back to
    technique+movement+clustering so it still works for trajectory-only files
    that lack conditionIndex.
    """
    ci = row.get("conditionIndex", -1)
    if ci == -1:
        return (row.get("participant"),
                row.get("technique"),
                row.get("movement"),
                row.get("clustering"),
                row.get("trialNumber"))
    return (row.get("participant"),
            ci,
            row.get("trialNumber"))


def _traj_key(row: dict) -> tuple:
    """Same identity logic for trajectory records."""
    ci = row.get("conditionIndex", -1)
    if ci == -1:
        return (row.get("participant"),
                row.get("technique"),
                row.get("movement"),
                row.get("clustering"),
                row.get("trialNumber"))
    return (row.get("participant"),
            ci,
            row.get("trialNumber"))


def _dedup_trials(rows: list[dict]) -> tuple[list[dict], int]:
    """
    Deduplicate trial rows, keeping the *richest* record when duplicates exist.
    'Richest' = the one with the most non-None values (trajectory-file records
    tend to have more fields populated than block-save records when both are
    loaded together).
    Returns (deduplicated_list, n_dropped).
    """
    best: dict[tuple, dict] = {}
    for row in rows:
        key = _trial_key(row)
        if key not in best:
            best[key] = row
        else:
            # Prefer whichever has more populated fields
            existing_score = sum(v is not None for v in best[key].values())
            new_score      = sum(v is not None for v in row.values())
            if new_score > existing_score:
                best[key] = row
    dropped = len(rows) - len(best)
    return list(best.values()), dropped


def _dedup_trajs(rows: list[dict]) -> tuple[list[dict], int]:
    """
    Deduplicate trajectory records.
    When duplicates exist, prefer the record that has normalizedTrajectory
    (XY data) over one that only has closureCurve.
    Returns (deduplicated_list, n_dropped).
    """
    best: dict[tuple, dict] = {}
    for row in rows:
        key = _traj_key(row)
        if key not in best:
            best[key] = row
        else:
            existing_has_xy = isinstance(best[key].get("normalizedTrajectory"), list)
            new_has_xy      = isinstance(row.get("normalizedTrajectory"), list)
            # Upgrade: closure-only → full XY
            if new_has_xy and not existing_has_xy:
                best[key] = row
            # Upgrade: more trajectory points
            elif new_has_xy and existing_has_xy:
                if len(row["normalizedTrajectory"]) > len(best[key]["normalizedTrajectory"]):
                    best[key] = row
    dropped = len(rows) - len(best)
    return list(best.values()), dropped


# ── Main loader ────────────────────────────────────────────────────────────────

def load_files(paths: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """
    Load ≥1 JSON files; auto-detect format.
    Deduplicates so that block saves + session save + trajectory file
    for the same participant can all be passed without double-counting.
    Returns (df, traj_records).
    """
    all_trials: list[dict] = []
    all_trajs:  list[dict] = []

    for p in paths:
        path = Path(p)
        print(f"  Loading {path.name} … ", end="", flush=True)
        with open(path) as f:
            data = json.load(f)

        fmt = _detect(data)
        print(f"[{fmt}]")

        if fmt == "block":
            tr, tj = _parse_block(data)
        elif fmt == "session":
            tr, tj = _parse_session(data)
        elif fmt == "trajectories":
            tr, tj = _parse_trajectories(data)
        else:
            # Last-ditch: if there's a 'conditions' key at all, try block parser
            if "conditionsInBlock" in data or "conditions" in data:
                print(f"    ⚠  Ambiguous format — attempting block/session parse …")
                key = "conditionsInBlock" if "conditionsInBlock" in data else "conditions"
                data_copy = dict(data)
                data_copy["conditionsInBlock"] = data_copy.pop(key)
                tr, tj = _parse_block(data_copy)
                if tr:
                    fmt = "recovered"
                else:
                    print(f"    ✗  Could not parse {path.name} — skipping.")
                    continue
            else:
                print(f"    ✗  Unknown format — skipping {path.name}.")
                continue

        print(f"    {len(tr):>4} trial rows,  {len(tj):>4} trajectory records")
        all_trials.extend(tr)
        all_trajs.extend(tj)

    if not all_trials:
        sys.exit("\n✗  No valid trial data found in any supplied file.")

    # ── Deduplication ──────────────────────────────────────────────────────────
    # Block saves + session/completion save + trajectory file all overlap.
    # We deduplicate on (participant, conditionIndex, trialNumber) so every
    # unique trial appears exactly once regardless of how many files encode it.
    raw_trial_count = len(all_trials)
    raw_traj_count  = len(all_trajs)

    all_trials, t_dropped = _dedup_trials(all_trials)
    all_trajs,  j_dropped = _dedup_trajs(all_trajs)

    if t_dropped or j_dropped:
        print(f"\n  ── Deduplication ──────────────────────────────────────────")
        print(f"     trials    : {raw_trial_count:>5} raw → {len(all_trials):>5} unique"
              f"  ({t_dropped} duplicates removed)")
        print(f"     trajectories: {raw_traj_count:>5} raw → {len(all_trajs):>5} unique"
              f"  ({j_dropped} duplicates removed)")
    else:
        print(f"\n  No duplicate records found across files.")

    # ── Build DataFrame ────────────────────────────────────────────────────────
    df = pd.DataFrame(all_trials)

    # Ordered categoricals for clean axis ordering
    for col, order in [("technique",  TECHNIQUE_ORDER),
                       ("movement",   MOVEMENT_ORDER),
                       ("clustering", CLUSTERING_ORDER)]:
        present = [x for x in order if x in df[col].unique()]
        extras  = [x for x in df[col].unique() if x not in order]
        df[col] = pd.Categorical(df[col],
                                 categories=present + extras,
                                 ordered=True)

    return df, all_trajs


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CSV / STATS EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_csvs(df: pd.DataFrame, out: Path) -> None:
    d = out / "data"
    d.mkdir(parents=True, exist_ok=True)

    df.to_csv(d / "trials.csv", index=False)
    print(f"    ✓ {d / 'trials.csv'}  ({len(df)} rows)")

    grp = ["participant", "technique", "movement", "clustering"]
    agg = (df.groupby(grp, observed=True)
             .agg(
                n_trials        = ("time_ms",            "count"),
                mean_mt_ms      = ("time_ms",            "mean"),
                median_mt_ms    = ("time_ms",            "median"),
                sd_mt_ms        = ("time_ms",            "std"),
                total_errors    = ("errorClickCount",    "sum"),
                mean_errors     = ("errorClickCount",    "mean"),
                mean_dist_px    = ("distanceToCenter_px","mean"),
                precision       = ("clickedInsideTarget","mean"),
                mean_amplitude  = ("amplitude_px",       "mean"),
                mean_shannon_id = ("shannon_id_bits",    "mean"),
             ).reset_index())

    elig = df[df["shannon_id_bits"].notna() & df["time_ms"].notna()].copy()
    if not elig.empty:
        elig["_tp"] = elig["shannon_id_bits"] / (elig["time_ms"] / 1000)
        tp = (elig.groupby(grp, observed=True)["_tp"]
                  .mean().reset_index()
                  .rename(columns={"_tp": "throughput_shannon_bps"}))
        agg = agg.merge(tp, on=grp, how="left")

    agg.to_csv(d / "conditions.csv", index=False)
    print(f"    ✓ {d / 'conditions.csv'}  ({len(agg)} rows)")

    summary = (df.groupby(["participant", "technique"], observed=True)
                 .agg(
                    n            = ("time_ms",            "count"),
                    mean_mt      = ("time_ms",            "mean"),
                    sd_mt        = ("time_ms",            "std"),
                    median_mt    = ("time_ms",            "median"),
                    total_errors = ("errorClickCount",    "sum"),
                    precision    = ("clickedInsideTarget","mean"),
                 ).reset_index())
    summary.to_csv(d / "summary.csv", index=False)
    print(f"    ✓ {d / 'summary.csv'}  ({len(summary)} rows)")


def export_stats(df: pd.DataFrame, out: Path) -> None:
    sd = out / "stats"
    sd.mkdir(parents=True, exist_ok=True)

    # Descriptives
    grp  = ["technique", "movement", "clustering"]
    desc = (df.groupby(grp, observed=True)["time_ms"]
              .agg(n="count", mean="mean", sd="std",
                   median="median",
                   q25=lambda x: x.quantile(0.25),
                   q75=lambda x: x.quantile(0.75))
              .reset_index())
    desc.columns = grp + ["n", "mt_mean", "mt_sd", "mt_median", "mt_q25", "mt_q75"]

    err_agg = (df.groupby(grp, observed=True)["errorClickCount"]
                 .agg(mean="mean", sd="std").reset_index())
    err_agg.columns = grp + ["err_mean", "err_sd"]
    desc = desc.merge(err_agg, on=grp)

    prec = (df.groupby(grp, observed=True)["clickedInsideTarget"]
              .mean().reset_index()
              .rename(columns={"clickedInsideTarget": "precision_rate"}))
    desc = desc.merge(prec, on=grp, how="left")
    desc.to_csv(sd / "descriptives.csv", index=False)
    print(f"    ✓ {sd / 'descriptives.csv'}  ({len(desc)} rows)")

    # Fitts Law
    df_f = df[df["shannon_id_bits"].notna() & df["time_ms"].notna()].copy()
    rows = []
    for tech, g in df_f.groupby("technique", observed=True):
        x = g["shannon_id_bits"].values
        y = g["time_ms"].values / 1000
        if len(x) < 5:
            continue
        slope, intercept, r, p_val, se = stats.linregress(x, y)
        rows.append({
            "technique":     tech,
            "n":             len(x),
            "a_s":           round(intercept, 4),
            "b_s_per_bit":   round(slope, 4),
            "r":             round(r, 4),
            "R2":            round(r ** 2, 4),
            "p":             round(p_val, 4),
            "se":            round(se, 4),
        })
    if rows:
        pd.DataFrame(rows).to_csv(sd / "fitts.csv", index=False)
        print(f"    ✓ {sd / 'fitts.csv'}")

    # Pairwise Welch t + Cohen's d
    techs = df["technique"].cat.categories.tolist()
    pairs = []
    for i in range(len(techs)):
        for j in range(i + 1, len(techs)):
            a = df[df["technique"] == techs[i]]["time_ms"].dropna()
            b = df[df["technique"] == techs[j]]["time_ms"].dropna()
            if len(a) < 2 or len(b) < 2:
                continue
            t_s, p_v = stats.ttest_ind(a, b, equal_var=False)
            pool_sd  = math.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
            d        = (a.mean() - b.mean()) / pool_sd if pool_sd > 0 else 0
            pairs.append({
                "pair":      f"{techs[i]} vs {techs[j]}",
                "mean_A_ms": round(a.mean(), 1),
                "mean_B_ms": round(b.mean(), 1),
                "diff_ms":   round(a.mean() - b.mean(), 1),
                "t":         round(t_s, 3),
                "p":         round(p_v, 4),
                "cohens_d":  round(d, 3),
                "nA":        len(a),
                "nB":        len(b),
            })
    if pairs:
        pd.DataFrame(pairs).to_csv(sd / "pairwise_mt.csv", index=False)
        print(f"    ✓ {sd / 'pairwise_mt.csv'}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  TRAJECTORY MATH
# ═══════════════════════════════════════════════════════════════════════════════

def _resample(traj: list[dict], bins: int) -> np.ndarray:
    """Time-resample to `bins` buckets. Returns (bins, 3): [x, y, dist_px]."""
    arr = np.array([[s.get("t_ms", 0), s.get("x", 0),
                     s.get("y", 0),    s.get("dist_px", 0)]
                    for s in traj], dtype=float)
    t_max = arr[-1, 0]
    if t_max <= 0:
        return np.zeros((bins, 3))

    acc  = np.zeros((bins, 3))
    cnt  = np.zeros(bins, dtype=int)
    idxs = np.clip((arr[:, 0] / t_max * bins).astype(int), 0, bins - 1)
    for i, row in zip(idxs, arr[:, 1:]):
        acc[i] += row
        cnt[i] += 1

    last = arr[0, 1:].copy()
    for i in range(bins):
        if cnt[i]:
            acc[i] /= cnt[i]
            last = acc[i].copy()
        else:
            acc[i] = last
    return acc


def _aggregate(trials: list[dict], bins: int) -> np.ndarray:
    stacked = np.stack([_resample(r["normalizedTrajectory"], bins)
                        for r in trials])
    return stacked.mean(axis=0)


def _smooth(arr: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    return gaussian_filter1d(arr, sigma=sigma, axis=0)


def _resample_closure(cc: list[dict], bins: int) -> np.ndarray:
    if not cc:
        return np.full(bins, np.nan)
    arr   = np.array([[s.get("t_ms", 0), s.get("dist_px", 0)] for s in cc])
    t_max = arr[-1, 0]
    if t_max <= 0:
        return np.full(bins, np.nan)
    acc, cnt = np.zeros(bins), np.zeros(bins, dtype=int)
    idxs = np.clip((arr[:, 0] / t_max * bins).astype(int), 0, bins - 1)
    for i, v in zip(idxs, arr[:, 1]):
        acc[i] += v
        cnt[i] += 1
    last = arr[0, 1]
    for i in range(bins):
        if cnt[i]:
            acc[i] /= cnt[i]
            last = acc[i]
        else:
            acc[i] = last
    return acc


def _group_trajs(trajs: list[dict], require_xy: bool = True) -> dict:
    """Group traj records by technique. require_xy=False includes closure-only records."""
    out: dict[str, list] = {}
    for r in trajs:
        if require_xy:
            if not isinstance(r.get("normalizedTrajectory"), list):
                continue
            if len(r["normalizedTrajectory"]) < 2:
                continue
        else:
            has_xy = (isinstance(r.get("normalizedTrajectory"), list)
                      and len(r["normalizedTrajectory"]) >= 2)
            has_cc = (isinstance(r.get("closureCurve"), list)
                      and len(r["closureCurve"]) >= 2)
            if not (has_xy or has_cc):
                continue
        tech = r.get("technique", "UNKNOWN")
        out.setdefault(tech, []).append(r)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  STATISTICAL FIGURES  (light theme)
# ═══════════════════════════════════════════════════════════════════════════════

def _violin_strip(ax, data, x, y, order, palette, title, xlabel, ylabel):
    present = [o for o in order if o in data[x].values]
    if not present:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        return
    pal = {k: v for k, v in palette.items() if k in present}
    sns.violinplot(data=data, x=x, y=y, order=present,
                   palette=pal, inner="quartile", cut=0, ax=ax)
    sns.stripplot(data=data, x=x, y=y, order=present,
                  palette=pal, size=2.5, alpha=0.22, jitter=True, ax=ax)
    ylim = ax.get_ylim()
    for i, cat in enumerate(present):
        med = data[data[x] == cat][y].median()
        ax.text(i, ylim[1] * 0.97, f"Md={med:.0f}",
                ha="center", fontsize=8,
                color=pal.get(cat, "#333"), fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def figure_performance(df: pd.DataFrame, out: Path, extra: dict) -> None:
    _light_style()
    tech_pal = {t: _tech_color(t, extra) for t in TECHNIQUE_ORDER}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Movement Time & Error Rate", fontsize=14, fontweight="bold")

    _violin_strip(axes[0, 0], df, "technique", "time_ms",
                  TECHNIQUE_ORDER, tech_pal,
                  "MT by Technique", "Technique", "MT (ms)")
    _violin_strip(axes[0, 1], df, "movement", "time_ms",
                  MOVEMENT_ORDER, MOVEMENT_PALETTE,
                  "MT by Movement Condition", "Movement", "MT (ms)")
    _violin_strip(axes[1, 0], df, "clustering", "time_ms",
                  CLUSTERING_ORDER, CLUSTERING_PALETTE,
                  "MT by Target Clustering", "Clustering", "MT (ms)")

    present = [t for t in TECHNIQUE_ORDER if t in df["technique"].values]
    pal     = {t: tech_pal[t] for t in present}
    sns.boxplot(data=df, x="technique", y="errorClickCount",
                order=present, palette=pal, width=0.5,
                flierprops={"marker": ".", "alpha": 0.3}, ax=axes[1, 1])
    axes[1, 1].set_title("Error Clicks by Technique", fontweight="bold")
    axes[1, 1].set_xlabel("Technique")
    axes[1, 1].set_ylabel("Error Clicks per Trial")

    plt.tight_layout()
    _save(fig, out / "figures" / "performance" / "mt_distributions.png")

    # Precision bar chart
    prec = df["clickedInsideTarget"].dropna()
    if len(prec) > 0:
        fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.5))
        fig2.suptitle("Precision Rate  (click inside target)",
                      fontsize=13, fontweight="bold")
        for ax, (col, order, pal) in zip(axes2, [
            ("technique",  TECHNIQUE_ORDER,  tech_pal),
            ("movement",   MOVEMENT_ORDER,   MOVEMENT_PALETTE),
            ("clustering", CLUSTERING_ORDER, CLUSTERING_PALETTE),
        ]):
            pres  = [o for o in order if o in df[col].values]
            means = [df[df[col] == c]["clickedInsideTarget"].mean() * 100
                     for c in pres]
            bars  = ax.bar(pres, means,
                           color=[pal.get(c, "#94a3b8") for c in pres],
                           edgecolor="white", linewidth=0.8)
            ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
            ax.set_ylim(0, 115)
            ax.set_ylabel("Precision Rate (%)")
            ax.set_xlabel(col.capitalize())
            ax.set_title(f"Precision by {col.capitalize()}", fontweight="bold")
        plt.tight_layout()
        _save(fig2, out / "figures" / "performance" / "precision.png")


def figure_throughput(df: pd.DataFrame, out: Path, extra: dict) -> None:
    _light_style()
    elig = df[df["shannon_id_bits"].notna() & df["time_ms"].notna()].copy()
    if elig.empty:
        print("    ⚠  No amplitude data — skipping throughput figure.")
        return
    elig["tp_bps"] = elig["shannon_id_bits"] / (elig["time_ms"] / 1000)
    tech_pal = {t: _tech_color(t, extra) for t in TECHNIQUE_ORDER}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Shannon Throughput  (ID / MT)", fontsize=13, fontweight="bold")
    _violin_strip(axes[0], elig, "technique", "tp_bps",
                  TECHNIQUE_ORDER, tech_pal,
                  "TP by Technique", "Technique", "Throughput (bits/s)")
    _violin_strip(axes[1], elig, "movement", "tp_bps",
                  MOVEMENT_ORDER, MOVEMENT_PALETTE,
                  "TP by Movement", "Movement", "Throughput (bits/s)")
    _violin_strip(axes[2], elig, "clustering", "tp_bps",
                  CLUSTERING_ORDER, CLUSTERING_PALETTE,
                  "TP by Clustering", "Clustering", "Throughput (bits/s)")
    plt.tight_layout()
    _save(fig, out / "figures" / "performance" / "throughput.png")


def figure_fitts(df: pd.DataFrame, out: Path, extra: dict) -> None:
    _light_style()
    df_f = df[df["shannon_id_bits"].notna() & df["time_ms"].notna()].copy()
    if df_f.empty:
        print("    ⚠  No Fitts data — skipping.")
        return
    techs   = [t for t in TECHNIQUE_ORDER if t in df_f["technique"].values]
    n_techs = max(len(techs), 1)
    fig, axes = plt.subplots(1, n_techs, figsize=(5 * n_techs, 5), sharey=True)
    if n_techs == 1:
        axes = [axes]
    fig.suptitle("Fitts' Law  —  ID vs Movement Time  (per technique)",
                 fontsize=13, fontweight="bold")

    for ax, tech in zip(axes, techs):
        g     = df_f[df_f["technique"] == tech]
        color = _tech_color(tech, extra)
        x     = g["shannon_id_bits"].values
        y     = g["time_ms"].values / 1000
        ax.scatter(x, y, color=color, alpha=0.28, s=12)
        if len(x) > 30:
            ax.hexbin(x, y, gridsize=18, cmap="Greys", alpha=0.35, linewidths=0.2)
        if len(x) >= 5:
            slope, intercept, r, p_val, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 120)
            ax.plot(x_line, intercept + slope * x_line, color=color, lw=2.2,
                    label=f"a={intercept:.3f}  b={slope:.3f}\n"
                          f"R²={r**2:.3f}  p={p_val:.3f}")
            ax.legend(fontsize=8, framealpha=0.7)
        ax.set_title(tech, fontweight="bold", color=color)
        ax.set_xlabel("Index of Difficulty  (bits)")
        if ax is axes[0]:
            ax.set_ylabel("Movement Time  (s)")
    plt.tight_layout()
    _save(fig, out / "figures" / "fitts" / "id_vs_mt.png")


def figure_interactions(df: pd.DataFrame, out: Path, extra: dict) -> None:
    _light_style()
    tech_pal = {t: _tech_color(t, extra) for t in TECHNIQUE_ORDER}
    techs    = [t for t in TECHNIQUE_ORDER if t in df["technique"].values]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Condition Interactions", fontsize=14, fontweight="bold")

    def _pp(ax, data, x, y, hue, x_order, hue_order, palette, title, ylabel):
        kw = dict(data=data, x=x, y=y, hue=hue,
                  order=[o for o in x_order if o in data[x].values],
                  hue_order=[h for h in hue_order if h in data[hue].values],
                  palette=palette, dodge=True, markers="o",
                  linestyles="-", ax=ax)
        try:
            sns.pointplot(errorbar="se", **kw)
        except TypeError:
            sns.pointplot(ci=68, **kw)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.legend(title=hue.capitalize(), fontsize=8)

    _pp(axes[0, 0], df, "movement", "time_ms", "technique",
        MOVEMENT_ORDER, techs, tech_pal,
        "MT: Technique × Movement", "Mean MT (ms)")
    _pp(axes[0, 1], df, "clustering", "time_ms", "technique",
        CLUSTERING_ORDER, techs, tech_pal,
        "MT: Technique × Clustering", "Mean MT (ms)")
    _pp(axes[1, 0], df, "movement", "errorClickCount", "technique",
        MOVEMENT_ORDER, techs, tech_pal,
        "Errors: Technique × Movement", "Mean Error Clicks")

    # Heatmap: movement × clustering
    pivot = (df.groupby(["movement", "clustering"], observed=True)["errorClickCount"]
               .mean().unstack("clustering")
               .reindex(index=[m for m in MOVEMENT_ORDER if m in df["movement"].values],
                        columns=[c for c in CLUSTERING_ORDER if c in df["clustering"].values]))
    if not pivot.empty:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                    linewidths=0.5, ax=axes[1, 1])
        axes[1, 1].set_title("Mean Errors: Movement × Clustering", fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "figures" / "interactions" / "interaction_plots.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  TRAJECTORY FIGURES  (dark theme)
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_xy(ax, by_tech, bins, extra, show_i, show_a):
    ax.set_title("XY Spatial  (normalised frame)")
    ax.set_xlabel("Approach  x  [px]")
    ax.set_ylabel("Lateral  y  [px]")
    ax.grid(True, alpha=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axhline(0, color="white", lw=0.5, ls="--", alpha=0.15)
    ax.axvline(0, color="white", lw=0.5, ls="--", alpha=0.15)

    for tech, recs in by_tech.items():
        xy_recs = [r for r in recs
                   if isinstance(r.get("normalizedTrajectory"), list)
                   and len(r["normalizedTrajectory"]) >= 2]
        if not xy_recs:
            continue
        color = _tech_color(tech, extra)
        if show_i:
            for r in xy_recs:
                pts = np.array([[s.get("x", 0), s.get("y", 0)]
                                for s in r["normalizedTrajectory"]])
                ax.plot(pts[:, 0], pts[:, 1], color=color,
                        lw=0.7, alpha=DARK["alpha_i"])
        if show_a:
            agg = _smooth(_aggregate(xy_recs, bins))
            ax.plot(agg[:, 0], agg[:, 1], color=color,
                    lw=DARK["agg_lw"], zorder=4,
                    label=f"{tech}  (n={len(xy_recs)})",
                    path_effects=[pe.Stroke(linewidth=DARK["agg_lw"] + 2,
                                            foreground="black", alpha=0.5),
                                  pe.Normal()])
            ax.scatter(agg[::DARK["dot_n"], 0], agg[::DARK["dot_n"], 1],
                       color=color, s=18, zorder=5)
            ax.annotate(tech, xy=(agg[-1, 0], agg[-1, 1]),
                        xytext=(5, 0), textcoords="offset points",
                        color=color, fontsize=8, fontweight="bold")

    ax.scatter([0], [0], color=DARK["target"], s=90, zorder=6,
               edgecolors="white", linewidths=1)
    ax.annotate("⊕ target", xy=(0, 0), xytext=(5, 5),
                textcoords="offset points", color=DARK["target"], fontsize=8)
    ax.legend(loc="upper left", fontsize=8)


def _draw_3d(ax, by_tech, bins, extra, show_i, show_a):
    ax.set_title("XY + Time  (3-D)")
    ax.set_xlabel("x  [px]")
    ax.set_ylabel("y  [px]")
    ax.set_zlabel("Norm. time")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(DARK["grid"])
    ax.grid(True, alpha=0.3)

    for tech, recs in by_tech.items():
        xy_recs = [r for r in recs
                   if isinstance(r.get("normalizedTrajectory"), list)
                   and len(r["normalizedTrajectory"]) >= 2]
        if not xy_recs:
            continue
        color = _tech_color(tech, extra)
        if show_i:
            for r in xy_recs:
                pts   = r["normalizedTrajectory"]
                t_max = max(pts[-1]["t_ms"], 1)
                ax.plot([s.get("x", 0) for s in pts],
                        [s.get("y", 0) for s in pts],
                        [s["t_ms"] / t_max for s in pts],
                        color=color, lw=0.5, alpha=DARK["alpha_i"] * 0.7)
        if show_a:
            agg    = _smooth(_aggregate(xy_recs, bins))
            t_norm = np.linspace(0, 1, bins)
            ax.plot(agg[:, 0], agg[:, 1], t_norm,
                    color=color, lw=DARK["agg_lw"], zorder=4,
                    label=f"{tech}  (n={len(xy_recs)})")
            ax.scatter(agg[::DARK["dot_n"], 0],
                       agg[::DARK["dot_n"], 1],
                       t_norm[::DARK["dot_n"]],
                       color=color, s=14, zorder=5, depthshade=True)
            ax.plot(agg[:, 0], agg[:, 1], np.zeros(bins),
                    color=color, lw=0.8, alpha=0.18, ls="--")

    ax.scatter([0], [0], [0], color=DARK["target"], s=70, zorder=6)
    ax.text(0, 0, 0.02, "⊕", color=DARK["target"], fontsize=7)
    ax.legend(loc="upper left", fontsize=8)


def _draw_closure(ax, by_tech, bins, extra, show_i, show_a):
    ax.set_title("Closure Profile  (dist to target vs norm. time)")
    ax.set_xlabel("Normalised time")
    ax.set_ylabel("Distance to target  [px]")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(True, alpha=0.5)
    ax.axhline(0, color=DARK["target"], lw=1, ls="--", alpha=0.25)
    t_norm = np.linspace(0, 1, bins)

    for tech, recs in by_tech.items():
        color = _tech_color(tech, extra)

        if show_i:
            for r in recs:
                if isinstance(r.get("normalizedTrajectory"), list) and \
                        len(r["normalizedTrajectory"]) >= 2:
                    pts   = r["normalizedTrajectory"]
                    t_max = max(pts[-1]["t_ms"], 1)
                    ax.plot([s["t_ms"] / t_max for s in pts],
                            [s.get("dist_px", 0) for s in pts],
                            color=color, lw=0.7, alpha=DARK["alpha_i"])
                elif isinstance(r.get("closureCurve"), list) and \
                        len(r["closureCurve"]) >= 2:
                    cc    = r["closureCurve"]
                    t_max = max(cc[-1]["t_ms"], 1)
                    ax.plot([s["t_ms"] / t_max for s in cc],
                            [s.get("dist_px", 0) for s in cc],
                            color=color, lw=0.7, alpha=DARK["alpha_i"])

        if show_a and recs:
            dist_arrays = []
            for r in recs:
                if isinstance(r.get("normalizedTrajectory"), list) and \
                        len(r["normalizedTrajectory"]) >= 2:
                    dist_arrays.append(_resample(r["normalizedTrajectory"], bins)[:, 2])
                elif isinstance(r.get("closureCurve"), list) and \
                        len(r["closureCurve"]) >= 2:
                    dist_arrays.append(_resample_closure(r["closureCurve"], bins))
            if not dist_arrays:
                continue
            agg_dist = gaussian_filter1d(
                np.stack(dist_arrays).mean(axis=0), 1.2)
            ax.plot(t_norm, agg_dist, color=color, lw=DARK["agg_lw"],
                    zorder=4, label=f"{tech}  (n={len(dist_arrays)})",
                    path_effects=[pe.Stroke(linewidth=DARK["agg_lw"] + 2,
                                            foreground="black", alpha=0.5),
                                  pe.Normal()])
            ax.scatter(t_norm[::DARK["dot_n"]], agg_dist[::DARK["dot_n"]],
                       color=color, s=14, zorder=5)
            li = bins // 10
            ax.annotate(tech, xy=(t_norm[li], agg_dist[li]),
                        xytext=(0, 6), textcoords="offset points",
                        color=color, fontsize=8, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)


def _draw_lateral(ax, by_tech, bins, extra):
    ax.set_title("Lateral Deviation  (mean ± 1 SD)")
    ax.set_xlabel("Normalised time")
    ax.set_ylabel("Lateral  y  [px]")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(0, color="white", lw=0.5, ls="--", alpha=0.18)
    ax.grid(True, alpha=0.5)
    t_norm = np.linspace(0, 1, bins)

    for tech, recs in by_tech.items():
        xy_recs = [r for r in recs
                   if isinstance(r.get("normalizedTrajectory"), list)
                   and len(r["normalizedTrajectory"]) >= 2]
        if not xy_recs:
            continue
        color   = _tech_color(tech, extra)
        stacked = np.stack([_resample(r["normalizedTrajectory"], bins)[:, 1]
                            for r in xy_recs])
        mu  = gaussian_filter1d(stacked.mean(axis=0), 1.5)
        sd  = stacked.std(axis=0)
        ax.plot(t_norm, mu, color=color, lw=DARK["agg_lw"],
                label=f"{tech}  (n={len(xy_recs)})",
                path_effects=[pe.Stroke(linewidth=DARK["agg_lw"] + 2,
                                        foreground="black", alpha=0.5),
                              pe.Normal()])
        ax.fill_between(t_norm, mu - sd, mu + sd, color=color, alpha=0.12)
    ax.legend(loc="upper right", fontsize=8)


def figure_trajectory_overview(trajs, out, bins, views, show_i, show_a):
    _dark_style()
    by_tech = _group_trajs(trajs, require_xy=False)
    extra   = _extra_colors(list(by_tech))
    n       = len(views)
    n_cols  = min(n, 2)
    n_rows  = math.ceil(n / n_cols)

    fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows))
    fig.patch.set_facecolor(DARK["bg"])
    n_trials = sum(len(v) for v in by_tech.values())
    fig.suptitle(f"Trajectory Overview  —  n={n_trials} records",
                 color=DARK["bright"], fontsize=11, y=0.998)

    for idx, view in enumerate(views):
        is_3d = (view == "3d")
        ax = fig.add_subplot(n_rows, n_cols, idx + 1,
                             projection="3d" if is_3d else None)
        if not is_3d:
            ax.set_facecolor(DARK["panel"])
        if view == "xy":
            _draw_xy(ax, by_tech, bins, extra, show_i, show_a)
        elif view == "3d":
            _draw_3d(ax, by_tech, bins, extra, show_i, show_a)
        elif view == "closure":
            _draw_closure(ax, by_tech, bins, extra, show_i, show_a)
        elif view == "lateral":
            _draw_lateral(ax, by_tech, bins, extra)

    plt.tight_layout(rect=[0, 0, 1, 0.997])
    _save(fig, out / "figures" / "trajectories" / "overview.png", fc=DARK["bg"])


def figure_closure_faceted(trajs, out, bins):
    _dark_style()
    t_norm = np.linspace(0, 1, bins)
    techs  = sorted({r.get("technique", "?") for r in trajs})
    extra  = _extra_colors(techs)

    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)
    fig.patch.set_facecolor(DARK["bg"])
    fig.suptitle("Closure Profiles  —  Movement × Clustering",
                 color=DARK["bright"], fontsize=12, y=0.998)

    for ri, mov in enumerate(MOVEMENT_ORDER):
        for ci, clust in enumerate(CLUSTERING_ORDER):
            ax = axes[ri, ci]
            ax.set_facecolor(DARK["panel"])
            ax.grid(True, alpha=0.4)
            ax.axhline(0, color=DARK["target"], lw=0.8, ls="--", alpha=0.25)

            subset = [r for r in trajs
                      if r.get("movement") == mov
                      and r.get("clustering") == clust]

            for tech in techs:
                recs = [r for r in subset if r.get("technique") == tech]
                if not recs:
                    continue
                color = _tech_color(tech, extra)

                dist_arrays = []
                for r in recs:
                    if isinstance(r.get("normalizedTrajectory"), list) \
                            and len(r["normalizedTrajectory"]) >= 2:
                        dist_arrays.append(
                            _resample(r["normalizedTrajectory"], bins)[:, 2])
                    elif isinstance(r.get("closureCurve"), list) \
                            and len(r["closureCurve"]) >= 2:
                        dist_arrays.append(
                            _resample_closure(r["closureCurve"], bins))

                for da in dist_arrays:
                    ax.plot(t_norm, da, color=color, lw=0.6,
                            alpha=DARK["alpha_i"] * 1.4)
                if dist_arrays:
                    agg = gaussian_filter1d(
                        np.stack(dist_arrays).mean(axis=0), 1.2)
                    ax.plot(t_norm, agg, color=color, lw=2.0, zorder=4,
                            label=f"{tech} n={len(dist_arrays)}")

            if ri == 0:
                ax.set_title(clust, color=DARK["bright"],
                             fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(mov, color=DARK["bright"],
                              fontsize=10, fontweight="bold")
            ax.xaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1, decimals=0))
            if ri == 2:
                ax.set_xlabel("Norm. time", color=DARK["text"], fontsize=8)
            if ri == 0 and ci == 2:
                ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.997])
    _save(fig, out / "figures" / "trajectories" / "closure_faceted.png",
          fc=DARK["bg"])


def figure_lateral_faceted(trajs, out, bins):
    _dark_style()
    t_norm = np.linspace(0, 1, bins)
    techs  = [t for t in TECHNIQUE_ORDER
              if any(r.get("technique") == t for r in trajs)]
    extra  = _extra_colors(techs)
    n_rows = max(len(techs), 1)

    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 4.2 * n_rows),
                              sharex=True, sharey="row")
    if n_rows == 1:
        axes = [axes]
    fig.patch.set_facecolor(DARK["bg"])
    fig.suptitle("Lateral Deviation  —  Technique × Movement  (mean ± 1 SD)",
                 color=DARK["bright"], fontsize=12, y=0.999)

    for ri, tech in enumerate(techs):
        color = _tech_color(tech, extra)
        for ci, mov in enumerate(MOVEMENT_ORDER):
            ax = axes[ri][ci]
            ax.set_facecolor(DARK["panel"])
            ax.grid(True, alpha=0.4)
            ax.axhline(0, color="white", lw=0.5, ls="--", alpha=0.18)

            recs = [r for r in trajs
                    if r.get("technique") == tech
                    and r.get("movement") == mov
                    and isinstance(r.get("normalizedTrajectory"), list)
                    and len(r["normalizedTrajectory"]) >= 2]

            if recs:
                stacked = np.stack([_resample(r["normalizedTrajectory"],
                                              bins)[:, 1] for r in recs])
                mu = gaussian_filter1d(stacked.mean(axis=0), 1.5)
                sd = stacked.std(axis=0)
                ax.plot(t_norm, mu, color=color, lw=2.0)
                ax.fill_between(t_norm, mu - sd, mu + sd,
                                color=color, alpha=0.15)
                ax.text(0.97, 0.95, f"n={len(recs)}",
                        transform=ax.transAxes, ha="right", va="top",
                        color=DARK["text"], fontsize=8)

            if ri == 0:
                ax.set_title(mov, color=DARK["bright"],
                             fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{tech}\ny [px]",
                              color=color, fontsize=9, fontweight="bold")
            ax.xaxis.set_major_formatter(
                mticker.PercentFormatter(xmax=1, decimals=0))
            if ri == n_rows - 1:
                ax.set_xlabel("Norm. time", color=DARK["text"], fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    _save(fig, out / "figures" / "trajectories" / "lateral_faceted.png",
          fc=DARK["bg"])


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  DEMO DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _make_demo() -> tuple[list, list]:
    rng = random.Random(42)
    tech_params = {
        "BUBBLE": dict(curv=0.10, over=0.04, mt_mu=650,  mt_sd=180, err=0.15),
        "POINT":  dict(curv=0.05, over=0.02, mt_mu=820,  mt_sd=220, err=0.06),
        "AREA":   dict(curv=0.16, over=0.14, mt_mu=590,  mt_sd=160, err=0.28),
    }
    trial_rows, traj_rows = [], []
    ci = 0
    for tech, p in tech_params.items():
        for mov in MOVEMENT_ORDER:
            for clust in CLUSTERING_ORDER:
                for trial_n in range(1, 21):
                    amp = rng.uniform(80, 380)
                    rad = rng.uniform(10, 30)
                    mt  = max(100, rng.gauss(p["mt_mu"], p["mt_sd"]) +
                              {"STATIC": 0, "SLOW": 80, "FAST": 160}[mov] +
                              {"LOW": -30, "MEDIUM": 0, "HIGH": 40}[clust])
                    n   = rng.randint(25, 60)
                    traj, cc = [], []
                    for k in range(n + 1):
                        pct  = k / n
                        ease = 1 - (1 - pct) ** 2.4
                        x    = amp * (1 - ease) + rng.gauss(0, 4)
                        y    = (amp * p["curv"] * math.sin(math.pi * pct)
                                + amp * p["over"] * math.sin(math.pi * pct * 2)
                                * (1 if rng.random() > 0.5 else -1)
                                + rng.gauss(0, 5))
                        d    = round(math.hypot(x, y), 2)
                        traj.append({"t_ms": round(pct * mt),
                                     "x": round(x, 2), "y": round(y, 2),
                                     "dist_px": d})
                        cc.append({"t_ms": round(pct * mt), "dist_px": d})

                    trial_rows.append({
                        "participant":          1,
                        "technique":            tech,
                        "movement":             mov,
                        "clustering":           clust,
                        "conditionIndex":       ci,
                        "techBlock":            TECHNIQUE_ORDER.index(tech) + 1,
                        "trialNumber":          trial_n,
                        "time_ms":              round(mt),
                        "errorClickCount":      rng.choices([0, 1, 2],
                                                    weights=[0.7, 0.2, 0.1])[0],
                        "clickX":               None,
                        "clickY":               None,
                        "targetX":              None,
                        "targetY":              None,
                        "distanceToCenter_px":  round(rng.uniform(0, rad), 2),
                        "normalizedDistance":   round(rng.uniform(0, 1), 3),
                        "clickedInsideTarget":  rng.random() > p["err"],
                        "amplitude_px":         round(amp, 2),
                        "targetRadius":         round(rad, 1),
                        "targetVx_px_per_s":    0,
                        "targetVy_px_per_s":    0,
                        "targetSpeed_px_per_s": 0,
                        "shannon_id_bits":      _shannon_id(amp, rad),
                    })
                    traj_rows.append({
                        "participant":          1,
                        "conditionIndex":       ci,
                        "technique":            tech,
                        "movement":             mov,
                        "clustering":           clust,
                        "trialNumber":          trial_n,
                        "normalizedTrajectory": traj,
                        "closureCurve":         cc,
                    })
                ci += 1
    return trial_rows, traj_rows


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    p.add_argument("input", nargs="?", default=None,
                   metavar="DIR_OR_FILE",
                   help="Directory of JSON files, a single JSON file, or omit "
                        "for synthetic demo.")
    p.add_argument("--output", "-o", default="traj_output", metavar="DIR",
                   help="Root output directory  (default: traj_output/).")
    p.add_argument("--bins",   type=int, default=80,
                   help="Time-resampling bins for trajectory math  (default: 80).")
    p.add_argument("--views", nargs="+",
                   choices=["xy", "3d", "closure", "lateral"],
                   default=["xy", "3d", "closure", "lateral"],
                   help="Trajectory panels to include in overview figure.")
    p.add_argument("--no-individual",   action="store_true",
                   help="Omit individual paths from trajectory figures.")
    p.add_argument("--no-aggregate",    action="store_true",
                   help="Omit aggregate lines from trajectory figures.")
    p.add_argument("--no-stats",        action="store_true",
                   help="Skip statistical figures (performance, Fitts, interactions).")
    p.add_argument("--no-trajectories", action="store_true",
                   help="Skip trajectory figures.")
    return p.parse_args()


def _resolve_json_paths(raw: str | None) -> list[str]:
    """
    Accept:
      - None              → return [] (demo mode)
      - path/to/file.json → return [that file]
      - path/to/dir/      → return all *.json files in the directory,
                            sorted by name, searched non-recursively by default
                            but one level of subdirectories is also scanned so
                            that a directory of per-participant sub-folders works.
    Exits with a clear message if nothing is found.
    """
    if raw is None:
        return []

    p = Path(raw)

    if not p.exists():
        sys.exit(f"\n✗  Path does not exist: {p}")

    if p.is_file():
        if p.suffix.lower() != ".json":
            print(f"  ⚠  {p.name} does not have a .json extension — trying anyway.")
        return [str(p)]

    # It's a directory — collect all JSON files
    direct    = sorted(p.glob("*.json"))
    one_deep  = sorted(p.glob("*/*.json"))
    all_paths = direct + [q for q in one_deep if q not in direct]

    if not all_paths:
        sys.exit(f"\n✗  No .json files found in {p}  (searched top-level and "
                 "one sub-directory level).")

    print(f"  Found {len(all_paths)} JSON file(s) in {p}")
    return [str(q) for q in all_paths]


def main():
    args  = parse_args()
    out   = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("\n── Loading data ──────────────────────────────────────────────────")
    paths = _resolve_json_paths(args.input)

    if paths:
        df, trajs = load_files(paths)
    else:
        print("  No input supplied — using synthetic demo data.")
        trial_rows, trajs = _make_demo()
        df = pd.DataFrame(trial_rows)
        for col, order in [("technique",  TECHNIQUE_ORDER),
                           ("movement",   MOVEMENT_ORDER),
                           ("clustering", CLUSTERING_ORDER)]:
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)

    techs = df["technique"].cat.categories.tolist()
    extra = _extra_colors(techs)

    print(f"\n  {len(df)} trials  ·  "
          f"{df['participant'].nunique()} participant(s)  ·  "
          f"{len(trajs)} trajectory records")
    print(f"  Techniques : {techs}")
    print(f"  Movements  : {df['movement'].cat.categories.tolist()}")
    print(f"  Clustering : {df['clustering'].cat.categories.tolist()}")

    # Sanity check: expected = participants × 27 conditions × 20 trials
    n_p   = df["participant"].nunique()
    n_exp = n_p * 27 * 20
    n_got = len(df)
    if n_got != n_exp:
        print(f"\n  ⚠  Expected {n_exp} trials ({n_p}p × 27 conditions × 20 trials) "
              f"but got {n_got}.  Check for missing/incomplete blocks.")
    else:
        print(f"\n  ✓  Trial count matches expected {n_p}p × 27 × 20 = {n_exp}.")

    print("\n── Exporting CSVs ────────────────────────────────────────────────")
    export_csvs(df, out)
    export_stats(df, out)

    if not args.no_stats:
        print("\n── Statistical figures ───────────────────────────────────────────")
        figure_performance(df, out, extra)
        figure_throughput(df, out, extra)
        figure_fitts(df, out, extra)
        figure_interactions(df, out, extra)

    if not args.no_trajectories and trajs:
        print("\n── Trajectory figures ────────────────────────────────────────────")
        show_i = not args.no_individual
        show_a = not args.no_aggregate
        figure_trajectory_overview(trajs, out, args.bins, args.views,
                                   show_i, show_a)
        figure_closure_faceted(trajs, out, args.bins)
        figure_lateral_faceted(trajs, out, args.bins)
    elif not trajs:
        print("  ⚠  No trajectory records — skipping trajectory figures.")

    print(f"\n✓  Done.  Output in  {out}/\n")


if __name__ == "__main__":
    main()