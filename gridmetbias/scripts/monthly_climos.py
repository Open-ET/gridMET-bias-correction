"""
Croplands monthly ET climatology comparing flux-tower closed/unclosed with corrected and uncorrected OpenET models.

Reproduces the Figure 7, 2×2 panel plot (Ensemble, eeMETRIC, SIMS, SSEBop).

Author: John Volk (john.volk@dri.edu)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths + config
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Data"))

PAIRED_DIR = os.path.join(DATA_DIR, "paired_flux_OpenET_data")
CORR_FILE = os.path.join(PAIRED_DIR, "merged_monthly_corrv3.csv")
UNCORR_FILE = os.path.join(PAIRED_DIR, "merged_monthly_uncorrv3.csv")

OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Plots", "OpenET_accuracy"))
OUT_NAME = "Figure7_croplands_monthly_climatology.jpg"
OUT_PATH = os.path.join(OUT_DIR, OUT_NAME)

LAND_COVER = "Croplands"

MODEL_COLS = {
    "ensemble_mean": "Ensemble",
    "EEMETRIC":      "eeMETRIC",
    "SIMS":          "SIMS",
    "SSEBOP":        "SSEBop",
}

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS = np.arange(1, 13)

# Figure 7 look
C_TOWER = "0.15"
C_SHADE = "#D8C3A5"   # beige band (stands out on grey background)
C_CORR = "tab:blue"
C_UNCORR = "tab:red"

LS_TOWER = "-"
LS_UNC = "--"


# ---------------------------------------------------------------------
# IO helpers
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
    # normalize strings + month end timestamp (matches the other figure scripts)
    df = df.copy()
    df = _harmonize_general_classification(df)

    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()
    df["General classification"] = df["General classification"].astype(str).str.strip()

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["DATE"] = df["DATE"].dt.to_period("M").dt.to_timestamp("M")
    df["month"] = df["DATE"].dt.month
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
# Stats helpers
# ---------------------------------------------------------------------

def _align_months(s):
    return s.reindex(MONTHS)


def _tower_monthly_means(df):
    # tower series (always derived from the corrected-side paired sample)
    closed = df.groupby("month")["Closed"].mean()

    if "Unclosed" in df.columns:
        unclosed = df.groupby("month")["Unclosed"].mean()
    elif "Open" in df.columns:
        unclosed = df.groupby("month")["Open"].mean()
    else:
        raise KeyError("No raw tower ET column found ('Unclosed' or 'Open').")

    return _align_months(closed), _align_months(unclosed)


def _nice_ylim(max_val, step=50):
    if not np.isfinite(max_val):
        return 100.0
    return float(int(np.ceil(max_val / step) * step))


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_croplands_monthly_climatology():
    df_corr, df_unc = _load_monthly_data()
    c, u = _filter_croplands_and_pair(df_corr, df_unc)

    plt.style.use("seaborn-v0_8-darkgrid")

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), dpi=300)
    axes = axes.flatten()

    # tower climatology (shared across panels)
    closed_m, unclosed_m = _tower_monthly_means(c)

    legend_handles, legend_labels = None, None

    for i, (model_col, model_name) in enumerate(MODEL_COLS.items()):
        ax = axes[i]

        if model_col not in c.columns or model_col not in u.columns:
            ax.set_visible(False)
            continue

        corr_m = _align_months(c.groupby("month")[model_col].mean())
        uncorr_m = _align_months(u.groupby("month")[model_col].mean())

        # closure band first (so it stays behind lines)
        shade = ax.fill_between(
            MONTHS,
            closed_m.values,
            unclosed_m.values,
            color=C_SHADE,
            alpha=0.35,
            zorder=1,
        )

        # tower
        ln_closed, = ax.plot(
            MONTHS, closed_m.values,
            color=C_TOWER,
            linestyle=LS_TOWER,
            linewidth=1.8,
            zorder=3,
        )
        ln_unclosed, = ax.plot(
            MONTHS, unclosed_m.values,
            color=C_TOWER,
            linestyle=LS_UNC,
            linewidth=1.4,
            zorder=3,
        )

        # models
        ln_corr, = ax.plot(
            MONTHS, corr_m.values,
            color=C_CORR,
            linestyle=LS_TOWER,
            linewidth=1.6,
            zorder=4,
        )
        ln_uncorr, = ax.plot(
            MONTHS, uncorr_m.values,
            color=C_UNCORR,
            linestyle=LS_UNC,
            linewidth=1.4,
            zorder=4,
        )

        # lock legend ordering once
        if legend_handles is None:
            legend_handles = [ln_closed, ln_unclosed, ln_corr, ln_uncorr, shade]
            legend_labels = ["Flux Tower Closed", "Flux Tower Unclosed", "Corrected", "Uncorrected", "Closure range"]

        # axes cosmetics
        ax.set_title(model_name, fontsize=11)

        ax.set_xticks(MONTHS)
        ax.set_xticklabels(MONTH_LABELS, fontsize=8)

        ymax = np.nanmax([
            np.nanmax(closed_m.values),
            np.nanmax(unclosed_m.values),
            np.nanmax(corr_m.values),
            np.nanmax(uncorr_m.values),
        ])
        ytop = _nice_ylim(ymax, step=50)
        ax.set_ylim(0, ytop)
        ax.set_yticks(np.arange(0, ytop + 1e-9, 50))
        ax.tick_params(axis="y", labelsize=8)

        # no per-axis labels (we use shared)
        ax.set_xlabel("")
        if i % 2 == 1:
            ax.set_ylabel("")

    # legend (centered above panels, ordered as requested)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98), # lift legend above subplot titles
            ncol=3,
            frameon=False,
            fontsize=8.5,
            handlelength=2.2,
            columnspacing=1.2,
            handletextpad=0.6,
        )


    # tighten whitespace so sups are close
        fig.subplots_adjust(
        left=0.1,
        right=0.98,
        bottom=0.12,
        top=0.84,      # lower the top of the subplot area (for legend)
        wspace=0.18,
        hspace=0.30,
    )


    fig.supxlabel("Month", fontsize=11, y=0.06)
    fig.supylabel("ET [mm/month]", fontsize=11, x=0.04)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_croplands_monthly_climatology()

