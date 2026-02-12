"""
Monthly absolute error reduction heatmaps with bias sign overlay (Figure 8).

Visualizes how ETo bias correct affects absolute error in OpenET models across
land cover types, and shows the sign of bias after correction. 

Author: John Volk (john.volk@dri.edu)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Data"))
PAIRED_DIR = os.path.join(DATA_DIR, "paired_flux_OpenET_data")

CORR_FILE = os.path.join(PAIRED_DIR, "merged_monthly_corrv3.csv")
UNCORR_FILE = os.path.join(PAIRED_DIR, "merged_monthly_uncorrv3.csv")

OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Plots", "OpenET_accuracy"))
OUT_NAME = "Figure8_absolute_error_reduction_heatmaps.jpg"
OUT_PATH = os.path.join(OUT_DIR, OUT_NAME)

MODELS = {
    "ensemble_mean": "Ensemble",
    "EEMETRIC":      "eeMETRIC",
    "SIMS":          "SIMS",
    "SSEBOP":        "SSEBop",
}

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS = np.arange(1, 13)

LAND_ORDER = [
    "Wetlands",
    "Shrublands",
    "Mixed Forests",
    "Grasslands",
    "Evergreen Forests",
    "Croplands",
]

# Border styling 
SPINE_COLOR = "0.25"
SPINE_LW = 1.0


# ---------------------------------------------------------------------
# Helpers
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
    df["month"] = df["DATE"].dt.month
    return df


def load_monthly_data():
    df_corr = _prep_monthly(pd.read_csv(CORR_FILE, low_memory=False))
    df_unc = _prep_monthly(pd.read_csv(UNCORR_FILE, low_memory=False))
    return df_corr, df_unc


def keep_paired_rows(df_corr, df_unc):
    keys = pd.merge(
        df_corr[["SITE_ID", "DATE"]],
        df_unc[["SITE_ID", "DATE"]],
        on=["SITE_ID", "DATE"],
    ).drop_duplicates()
    return df_corr.merge(keys, on=["SITE_ID", "DATE"]), df_unc.merge(keys, on=["SITE_ID", "DATE"])


def compute_heat_and_sign(df_corr, df_unc):
    closed = (
        df_corr
        .groupby(["General classification", "month"])["Closed"]
        .mean()
        .reset_index(name="ET_closed")
    )

    bias_tables, sign_tables = {}, {}

    for col, label in MODELS.items():
        if col not in df_corr.columns or col not in df_unc.columns:
            continue

        corr_m = (
            df_corr
            .groupby(["General classification", "month"])[col]
            .mean()
            .reset_index(name="ET_corr")
        )
        uncorr_m = (
            df_unc
            .groupby(["General classification", "month"])[col]
            .mean()
            .reset_index(name="ET_uncorr")
        )

        dfb = (
            closed
            .merge(corr_m, on=["General classification", "month"], how="inner")
            .merge(uncorr_m, on=["General classification", "month"], how="inner")
        )

        dfb["imp"] = (
            (dfb["ET_uncorr"] - dfb["ET_closed"]).abs()
            - (dfb["ET_corr"]   - dfb["ET_closed"]).abs()
        )
        dfb["sign"] = np.where((dfb["ET_corr"] - dfb["ET_closed"]) > 0, "+", "-")

        heat = dfb.pivot(index="General classification", columns="month", values="imp")
        sign = dfb.pivot(index="General classification", columns="month", values="sign")

        heat = heat.reindex(LAND_ORDER).reindex(columns=MONTHS)
        sign = sign.reindex(LAND_ORDER).reindex(columns=MONTHS)

        bias_tables[label] = heat
        sign_tables[label] = sign

    # SIMS only Croplands 
    if "SIMS" in bias_tables:
        mask = bias_tables["SIMS"].index != "Croplands"
        bias_tables["SIMS"].loc[mask, :] = np.nan
        sign_tables["SIMS"].loc[mask, :] = np.nan

    return bias_tables, sign_tables


def compute_color_limits(bias_tables):
    all_vals = np.concatenate([h.to_numpy().ravel() for h in bias_tables.values()])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        raise ValueError("No finite improvement values found.")
    vmax = float(np.nanpercentile(np.abs(all_vals), 98))
    vmax = float(np.ceil(vmax / 5.0) * 5.0)
    vmax = max(vmax, 5.0)
    vmin = -vmax
    ticks = np.arange(vmin, vmax + 0.1, 5.0)
    return vmin, vmax, ticks


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_error_reduction_heatmaps():
    df_corr, df_unc = load_monthly_data()
    df_corr, df_unc = keep_paired_rows(df_corr, df_unc)
    bias_tables, sign_tables = compute_heat_and_sign(df_corr, df_unc)

    vmin, vmax, cbar_ticks = compute_color_limits(bias_tables)

    plt.style.use("seaborn-v0_8-white")

    fig = plt.figure(figsize=(8.1, 6.0), dpi=300)
    gs = fig.add_gridspec(
        nrows=2, ncols=3,
        width_ratios=[1, 1, 0.045],
        wspace=0.12,
        hspace=0.25
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cax = fig.add_subplot(gs[:, 2])

    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad(alpha=0.0)

    last_im = None

    for i, label in enumerate(MODELS.values()):
        ax = axes[i]
        if label not in bias_tables:
            ax.set_visible(False)
            continue

        heat = bias_tables[label]
        sign = sign_tables[label]
        H = np.ma.masked_invalid(heat.to_numpy())

        im = ax.imshow(
            H,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            origin="upper",
            interpolation="none",
        )
        last_im = im

        ax.set_title(label, fontsize=12)
        ax.grid(False)

        # X ticks: labels only on bottom row
        ax.set_xticks(np.arange(12))
        if i in (2, 3):
            ax.set_xticklabels(MONTH_LABELS, rotation=60, ha="right", fontsize=10)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)

        # Y ticks: labels only on left column
        ax.set_yticks(np.arange(len(LAND_ORDER)))
        if i in (0, 2):
            ax.set_yticklabels(LAND_ORDER, fontsize=11)
            ax.set_ylabel("")  # remove "Land-cover group"
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

        # Overlay +/- 
        n_row, n_col = heat.shape
        for r in range(n_row):
            for c in range(n_col):
                if not np.isfinite(heat.iat[r, c]): # avoid nulls (SIMS)
                    continue
                s = sign.iat[r, c]
                if not isinstance(s, str) or s not in ("+", "-"):
                    continue
                ax.text(c, r, s, ha="center", va="center", fontsize=10, color="black")

        # lighter borders
        for sp in ax.spines.values():
            sp.set_linewidth(SPINE_LW)
            sp.set_color(SPINE_COLOR)

    # Shared colorbar
    if last_im is not None:
        cbar = fig.colorbar(last_im, cax=cax, ticks=cbar_ticks)
        cbar.set_label("Absolute error reduction [mm/month]", fontsize=13, labelpad=6)
        cbar.ax.tick_params(labelsize=11)
        cbar.outline.set_linewidth(1.0)
        cbar.outline.set_edgecolor(SPINE_COLOR)
        for sp in cbar.ax.spines.values():
            sp.set_linewidth(1.0)
            sp.set_color(SPINE_COLOR)

    fig.subplots_adjust(left=0.20, right=0.93, top=0.95, bottom=0.10)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_error_reduction_heatmaps()

