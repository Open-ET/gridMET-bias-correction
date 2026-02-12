"""
Croplands monthly OpenET ET vs. closed flux-tower ET, comparing corrected vs uncorrected (Figure 6).
2×2 panels (Ensemble, eeMETRIC, SIMS, SSEBop) with r-squared/MBE/MAE.

Author: John Volk (john.volk@dri.edu)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Data"))

PAIRED_DIR = os.path.join(DATA_DIR, "paired_flux_OpenET_data")
CORR_FILE = os.path.join(PAIRED_DIR, "merged_monthly_corrv3.csv")
UNCORR_FILE = os.path.join(PAIRED_DIR, "merged_monthly_uncorrv3.csv")

OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Plots", "OpenET_accuracy"))
OUT_NAME = "Figure6_croplands_monthly_openet_vs_flux.jpg"
OUT_PATH = os.path.join(OUT_DIR, OUT_NAME)

LAND_COVER = "Croplands"
MIN_MONTHS = 3

MODEL_COLS = {
    "ensemble_mean": "Ensemble",
    "EEMETRIC":      "eeMETRIC",
    "SIMS":          "SIMS",
    "SSEBOP":        "SSEBop",
}

# visual style
COLOR_MAP = {"corrected": "tab:blue", "uncorrected": "tab:red"}
MARKER_MAP = {"corrected": "o", "uncorrected": "s"}

AX_LIM = (0, 300)
TICK_VALS = np.arange(0, 301, 50)

# in-panel stats placement
STAT_X = 0.03
STAT_Y = {"corrected": 0.93, "uncorrected": 0.84}

# small white backing so text stays readable on darkgrid
STAT_BBOX = dict(
    facecolor="white",
    edgecolor="none",
    alpha=0.65,
    boxstyle="round,pad=0.12",
)


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------

def _harmonize_general_classification(df):
    if "General classification" not in df.columns:
        for cand in ("General classification_x", "General classification_y"):
            if cand in df.columns:
                df = df.rename(columns={cand: "General classification"})
                other = "General classification_y" if cand.endswith("_x") else "General classification_x"
                if other in df.columns:
                    df = df.drop(columns=[other])
                break
    if "General classification" not in df.columns:
        raise KeyError("Expected 'General classification' column in monthly OpenET–flux files.")
    return df


def _prep_monthly(df):
    df = df.copy()
    df = _harmonize_general_classification(df)

    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()
    df["General classification"] = df["General classification"].astype(str).str.strip()

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["DATE"] = df["DATE"].dt.to_period("M").dt.to_timestamp("M")
    return df


def _load_monthly_data():
    df_corr = _prep_monthly(pd.read_csv(CORR_FILE, low_memory=False))
    df_unc = _prep_monthly(pd.read_csv(UNCORR_FILE, low_memory=False))
    return df_corr, df_unc


def _filter_croplands_and_pair(df_corr, df_unc):
    # Croplands only + enforce shared SITE_ID/DATE sample between corr/uncorr
    c = df_corr[df_corr["General classification"] == LAND_COVER].copy()
    u = df_unc[df_unc["General classification"] == LAND_COVER].copy()

    keys = pd.merge(
        c[["SITE_ID", "DATE"]],
        u[["SITE_ID", "DATE"]],
        on=["SITE_ID", "DATE"],
    ).drop_duplicates()

    c = c.merge(keys, on=["SITE_ID", "DATE"], how="inner")
    u = u.merge(keys, on=["SITE_ID", "DATE"], how="inner")
    return c, u


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def _compute_site_weighted_metrics(df, x_col, y_col, min_months=3):
    # r² = pearson r squared on all paired points (no site filtering)
    df_pairs = df[[x_col, y_col]].dropna()
    if len(df_pairs) >= 2:
        r, _ = pearsonr(df_pairs[x_col].to_numpy(), df_pairs[y_col].to_numpy())
        r2 = float(r ** 2)
    else:
        r2 = np.nan

    # MBE/MAE = site metrics, then √n-weighted mean across sites (exclude n < min_months)
    site_rows = []
    for _, g in df.groupby("SITE_ID"):
        gv = g[[x_col, y_col]].dropna()
        n = len(gv)
        if n < min_months:
            continue

        diff = gv[y_col].to_numpy() - gv[x_col].to_numpy()
        mbe_site = float(np.mean(diff))
        mae_site = float(np.mean(np.abs(diff)))
        w = float(np.sqrt(n))
        site_rows.append((w, mbe_site, mae_site))

    if not site_rows:
        return r2, np.nan, np.nan

    w = np.array([t[0] for t in site_rows], dtype=float)
    mbe = np.array([t[1] for t in site_rows], dtype=float)
    mae = np.array([t[2] for t in site_rows], dtype=float)

    wsum = float(w.sum())
    if wsum == 0:
        return r2, np.nan, np.nan

    mbe_w = float((w * mbe).sum() / wsum)
    mae_w = float((w * mae).sum() / wsum)

    return r2, mbe_w, mae_w


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot():
    df_corr, df_unc = _load_monthly_data()
    df_corr, df_unc = _filter_croplands_and_pair(df_corr, df_unc)

    plt.style.use("seaborn-v0_8-darkgrid")

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.5), dpi=300)
    axes = axes.flatten()

    for i, (model_col, display_name) in enumerate(MODEL_COLS.items()):
        ax = axes[i]

        # corrected first, then uncorrected (matches your previous Figure 6)
        for mode, df_mode in (("corrected", df_corr), ("uncorrected", df_unc)):
            if model_col not in df_mode.columns:
                continue

            d = df_mode[["SITE_ID", "Closed", model_col]].dropna()
            if len(d) < 2:
                continue

            x = d["Closed"].to_numpy()
            y = d[model_col].to_numpy()

            r2, mbe, mae = _compute_site_weighted_metrics(d, "Closed", model_col, MIN_MONTHS)

            ax.scatter(
                x, y,
                s=12,
                marker=MARKER_MAP[mode],
                facecolors="none",
                edgecolors=COLOR_MAP[mode],
                linewidths=0.9,
                alpha=0.55,
                zorder=3,
            )

            sub = "c" if mode == "corrected" else "u"
            stat_text = rf"$r^2_{{{sub}}}={r2:.2f},\ MBE_{{{sub}}}={mbe:.2f},\ MAE_{{{sub}}}={mae:.2f}$"
            ax.text(
                STAT_X,
                STAT_Y[mode],
                stat_text,
                transform=ax.transAxes,
                fontsize=8.4,
                color=COLOR_MAP[mode],
                ha="left",
                va="top",
                bbox=STAT_BBOX,
                zorder=5,
            )

        # 1:1 line only
        ax.plot(
            [AX_LIM[0], AX_LIM[1]],
            [AX_LIM[0], AX_LIM[1]],
            linestyle="--",
            color="k",
            alpha=0.40,
            linewidth=1.0,
            zorder=1,
        )

        ax.set_title(display_name, fontsize=12)
        ax.set_xlim(*AX_LIM)
        ax.set_ylim(*AX_LIM)
        ax.set_xticks(TICK_VALS)
        ax.set_yticks(TICK_VALS)

        # keep panels clean; shared labels handle the rest
        if i % 2 == 1:
            ax.set_ylabel("")
        if i < 2:
            ax.set_xlabel("")

    fig.supxlabel("Closed Flux Tower ET [mm/month]", fontsize=12, y=0.06)
    fig.supylabel("OpenET ET [mm/month]", fontsize=12, x=0.04)

    # Legend markers MUST be visible hollow markers (outside on right)
    legend_handles = [
        Line2D([0], [0],
               marker=MARKER_MAP["corrected"],
               linestyle="None",
               markerfacecolor="none",
               markeredgecolor=COLOR_MAP["corrected"],
               markeredgewidth=1.6,
               markersize=5),
        Line2D([0], [0],
               marker=MARKER_MAP["uncorrected"],
               linestyle="None",
               markerfacecolor="none",
               markeredgecolor=COLOR_MAP["uncorrected"],
               markeredgewidth=1.6,
               markersize=5),
    ]

    fig.legend(
        legend_handles,
        ["Corrected", "Uncorrected"],
        loc="upper left",
        bbox_to_anchor=(0.84, 0.88),
        frameon=False,
        fontsize=10,
        handletextpad=0.6,
        labelspacing=0.7,
        borderaxespad=0.0,
    )

    # layout: reserve space at right for the legend (and keep labels close)
    fig.subplots_adjust(left=0.10, right=0.82, bottom=0.12, top=0.93, wspace=0.22, hspace=0.28)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot()

