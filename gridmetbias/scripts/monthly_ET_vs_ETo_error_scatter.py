"""
Absolute improvement in monthly OpenET ET (at EC sites) after applying 
an ETo bias correction versus improvement in ETo at the same flux stations.

Figure 9. 

Author: John Volk (john.volk@dri.edu)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Data"))

PAIRED_DIR = os.path.join(DATA_DIR, "paired_flux_OpenET_data")
ET_CORR_FILE = os.path.join(PAIRED_DIR, "merged_monthly_corrv3.csv")
ET_UNCORR_FILE = os.path.join(PAIRED_DIR, "merged_monthly_uncorrv3.csv")

FLUX_GRIDMET_DIR = os.path.join(DATA_DIR, "flux_gridmet")
ETO_MONTHLY_FILE = os.path.join(FLUX_GRIDMET_DIR, "flux_gridmet_monthly.csv")

OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "Plots", "OpenET_accuracy"))
OUT_NAME = "Figure9_error_reduction_scatter_by_landcover.jpg"
OUT_PATH = os.path.join(OUT_DIR, OUT_NAME)

MODELS = {
    "ensemble_mean": "Ensemble",
    "EEMETRIC":      "eeMETRIC",
    "SIMS":          "SIMS",
    "SSEBOP":        "SSEBop",
}

LAND_TYPES = [
    "Croplands",
    "Evergreen Forests",
    "Grasslands",
    "Mixed Forests",
    "Shrublands",
    "Wetlands",
]

COLOR_MAP = {
    "Croplands":         "#1f77b4",
    "Evergreen Forests": "#2ca02c",
    "Grasslands":        "#ff7f0e",
    "Mixed Forests":     "#9467bd",
    "Shrublands":        "#d62728",
    "Wetlands":          "#17becf",
}

MARKER_MAP = {
    "Croplands":         "o",
    "Evergreen Forests": "s",
    "Grasslands":        "D",
    "Mixed Forests":     "^",
    "Shrublands":        "X",
    "Wetlands":          "P",
}


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
        raise KeyError("Expected 'General classification' column.")
    return df


def _normalize_month_end(df, date_col="DATE"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[date_col] = df[date_col].dt.to_period("M").dt.to_timestamp("M")
    return df


def load_openet_monthly_et():
    df_corr = pd.read_csv(ET_CORR_FILE, low_memory=False)
    df_unc = pd.read_csv(ET_UNCORR_FILE, low_memory=False)

    df_corr = _harmonize_general_classification(df_corr)
    df_unc = _harmonize_general_classification(df_unc)

    for df in (df_corr, df_unc):
        df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()
        df["General classification"] = df["General classification"].astype(str).str.strip()
        df = _normalize_month_end(df, "DATE")

    # reassign after normalize (since we created copies)
    df_corr = _normalize_month_end(df_corr, "DATE")
    df_unc = _normalize_month_end(df_unc, "DATE")

    # filter to the land-cover groups used in the figure
    df_corr = df_corr[df_corr["General classification"].isin(LAND_TYPES)].copy()
    df_unc = df_unc[df_unc["General classification"].isin(LAND_TYPES)].copy()

    # keep only SITE_ID+DATE pairs present in BOTH corrected and uncorrected ET files
    keys = pd.merge(
        df_corr[["SITE_ID", "DATE"]],
        df_unc[["SITE_ID", "DATE"]],
        on=["SITE_ID", "DATE"],
    ).drop_duplicates()

    df_corr = df_corr.merge(keys, on=["SITE_ID", "DATE"], how="inner")
    df_unc = df_unc.merge(keys, on=["SITE_ID", "DATE"], how="inner")

    # Keep only required columns
    keep_corr = ["SITE_ID", "DATE", "Closed", "General classification"] + list(MODELS.keys())
    keep_unc = ["SITE_ID", "DATE"] + list(MODELS.keys())

    df_corr = df_corr[[c for c in keep_corr if c in df_corr.columns]].drop_duplicates()
    df_unc = df_unc[[c for c in keep_unc if c in df_unc.columns]].drop_duplicates()

    # Merge corrected + uncorrected model ET
    df = pd.merge(
        df_corr,
        df_unc,
        on=["SITE_ID", "DATE"],
        suffixes=("_corr", "_uncorr"),
        how="inner",
    )

    return df


def load_monthly_eto_pivot():
    df = pd.read_csv(ETO_MONTHLY_FILE, low_memory=False)
    df = _harmonize_general_classification(df)

    df["SITE_ID"] = df["SITE_ID"].astype(str).str.strip()
    df["General classification"] = df["General classification"].astype(str).str.strip()
    df = _normalize_month_end(df, "DATE")

    # Pivot corrected/uncorrected gridMET ETo into columns (mean handles duplicates)
    eto_pivot = (
        df.pivot_table(
            index=["SITE_ID", "DATE", "ASCE_ETo"],
            columns="gridMET Corr_Uncorr",
            values="GRIDMET_REFERENCE_ET",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"Corrected": "ETo_corr", "Uncorrected": "ETo_uncorr"})
    )

    # Require both columns
    if "ETo_corr" not in eto_pivot.columns or "ETo_uncorr" not in eto_pivot.columns:
        raise KeyError("ETo pivot did not produce both ETo_corr and ETo_uncorr columns.")

    return eto_pivot


def compute_panel_limits(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return -1, 1

    vals = np.concatenate([x[mask], y[mask]])
    lo, hi = np.nanpercentile(vals, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = -1, 1

    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


# ---------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------

def plot_error_reduction_scatter():
    df_et = load_openet_monthly_et()
    df_eto = load_monthly_eto_pivot()

    # Merge ET and ETo on SITE_ID+DATE (inner => only months with both)
    df = pd.merge(df_et, df_eto, on=["SITE_ID", "DATE"], how="inner")

    # Compute ETo improvement once (shared x across all panels)
    df["imp_ETo"] = (df["ETo_uncorr"] - df["ASCE_ETo"]).abs() - (df["ETo_corr"] - df["ASCE_ETo"]).abs()

    # Compute ET improvement per model (panel y)
    for col, label in MODELS.items():
        df[f"imp_ET_{label}"] = (
            (df[f"{col}_uncorr"] - df["Closed"]).abs()
            - (df[f"{col}_corr"] - df["Closed"]).abs()
        )

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.2), dpi=300)
    axes = axes.flatten()

    models_order = ["Ensemble", "eeMETRIC", "SIMS", "SSEBop"]

    # draw scatters
    for i, model_name in enumerate(models_order):
        ax = axes[i]

        x_all = df["imp_ETo"].to_numpy(dtype=float)
        y_all = df[f"imp_ET_{model_name}"].to_numpy(dtype=float)

        # per-land-cover series
        for lc in LAND_TYPES:
            sub = df[df["General classification"] == lc]
            if sub.empty:
                continue

            x = sub["imp_ETo"].to_numpy(dtype=float)
            y = sub[f"imp_ET_{model_name}"].to_numpy(dtype=float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() == 0:
                continue

            ax.scatter(
                x[m],
                y[m],
                s=16,
                c=COLOR_MAP[lc],
                marker=MARKER_MAP[lc],
                alpha=0.75,
                edgecolors="black",
                linewidths=0.3,
                label=lc if i == 0 else None,
                zorder=3,
            )

        # 1:1 dashed line (panel-specific safe limits)
        lo, hi = compute_panel_limits(x_all, y_all)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.35", linewidth=1.0, zorder=2)

        ax.set_title(model_name, fontsize=11)

        # r^2 annotation (Pearson r squared), like the PDF
        mxy = np.isfinite(x_all) & np.isfinite(y_all)
        if mxy.sum() >= 2:
            r, _ = pearsonr(x_all[mxy], y_all[mxy])
            r2 = r ** 2
        else:
            r2 = np.nan

        if np.isfinite(r2):
            ax.text(
                0.05, 0.90,
                rf"$r^2={r2:.2f}$",
                transform=ax.transAxes,
                fontsize=9,
                color="0.25",
                ha="left",
                va="top",
            )

        # ticks only on bottom row and left column (like PDF)
        ax.tick_params(labelsize=9)
        ax.tick_params(axis="x", labelbottom=(i in (2, 3)))
        ax.tick_params(axis="y", labelleft=(i in (0, 2)))

        # Lighten spines (PDF-style)
        for sp in ax.spines.values():
            sp.set_linewidth(1.0)
            sp.set_color("0.30")

    # shared labels
    fig.supxlabel(r"Absolute error reduction (ET$_\mathrm{o}$) [mm/month]", fontsize=11, y=0.05)
    fig.supylabel("Absolute error reduction (ET) [mm/month]", fontsize=11, x=0.06)

    # legend on the right, close to the panels (give it more room so labels aren't clipped)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.8, 0.92),  # move slightly left
            frameon=False,
            fontsize=9,
            borderaxespad=0.0,
            handletextpad=0.6,
            labelspacing=0.7,
            handlelength=1.8,             # a bit more room for marker+text
        )

    # spacing/margins (increase right margin area for legend text)
    fig.subplots_adjust(
        left=0.12,
        right=0.80,   # was 0.84; more space reserved for legend
        bottom=0.12,
        top=0.92,
        wspace=0.28,
        hspace=0.40,
    )


    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    plot_error_reduction_scatter()

