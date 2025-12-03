"""
Create scatter plots of OpenET versus flux tower ET before and after ETo bias correction
Author: Dr. John Volk (john.volk@dri.edu)
Modified by: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == "__main__":

    # === SETTINGS ===
    frequency = "monthly"  # "monthly" or "daily"
    output_dir = Path()/'..'/'..'/'Plots'/'OpenET_accuracy'

    output_dir.mkdir(parents=True, exist_ok=True)

    # File paths and axis settings
    if frequency == "monthly":
        df_corr_path = Path()/".."/".."/"Data"/"paired_flux_OpenET_data"/"merged_monthly_corrv2.csv"
        df_uncorr_path = Path()/".."/".."/"Data"/"paired_flux_OpenET_data"/"merged_monthly_uncorrv2.csv"
        x_title = "Closed Flux Tower ET [mm/month]"
        y_title_base = "OpenET {} ET [mm/month]"
        x_range = [0, 300]
        y_range = [0, 300]
        tick_vals = list(range(0, 301, 50))
    else:
        df_corr_path = Path()/".."/".."/"Data"/"paired_flux_OpenET_data"/"merged_daily_corrv2.csv"
        df_uncorr_path = Path()/".."/".."/"Data"/"paired_flux_OpenET_data"/"merged_daily_uncorrv2.csv"
        x_title = "Closed Flux Tower ET [mm/day]"
        y_title_base = "OpenET {} ET [mm/day]"
        x_range = [0, 14]
        y_range = [0, 14]
        tick_vals = list(range(0, 15, 2))

    # only plot models that are affected
    models = ['EEMETRIC', 'SSEBOP', 'SIMS', 'ensemble_mean']

    # load data files
    df_corr = pd.read_csv(df_corr_path)
    df_uncorr = pd.read_csv(df_uncorr_path)

    df_corr["DATE"] = pd.to_datetime(df_corr["DATE"])
    df_uncorr["DATE"] = pd.to_datetime(df_uncorr["DATE"])
    df_corr["SITE_ID"] = df_corr["SITE_ID"].astype(str)
    df_uncorr["SITE_ID"] = df_uncorr["SITE_ID"].astype(str)

    df_corr_east = df_corr[df_corr.Longitude >= -100].copy()
    df_corr_west = df_corr[df_corr.Longitude < -100].copy()
    df_uncorr_east = df_uncorr[df_uncorr.Longitude >= -100].copy()
    df_uncorr_west = df_uncorr[df_uncorr.Longitude < -100].copy()

    land_type_mapping = {
        "Croplands": "Croplands",
        "Evergreen Forests": "Evergreen Forests",
        "Grasslands": "Grasslands",
        "Mixed Forests": "Mixed Forests",
        "Shrublands": "Shrublands",
        "Wetlands": "Wetland/Riparian"  # label correction only
    }

    # plotting controls
    color_map = {'corrected': 'black', 'uncorrected': 'firebrick'}
    symbol_map = {'corrected': 'circle', 'uncorrected': 'square'}
    offset_y_r2 = {'corrected': 0.97, 'uncorrected': 0.84}
    offset_y_m = {'corrected': 0.91, 'uncorrected': 0.78}

    df_list = [
        (df_corr.copy(deep=True), df_uncorr.copy(deep=True)), 
        (df_corr_east, df_uncorr_east), 
        (df_corr_west, df_uncorr_west)
    ]
    df_list_names = ["All", "East", "West"]
    for df_tuple, df_name in zip(df_list, df_list_names):
        df_corr, df_uncorr = df_tuple
        plot_dir = output_dir / f"{frequency}_{df_name}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for m in models:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[land_type_mapping[k] for k in land_type_mapping],
                x_title=x_title,
                y_title=y_title_base.format(m),
                horizontal_spacing=0.07,
                vertical_spacing=0.15
            )

            for idx, land in enumerate(land_type_mapping):
                row = idx // 3 + 1
                col = idx % 3 + 1

                df_corr_land = df_corr[df_corr["General classification"] == land]
                df_uncorr_land = df_uncorr[df_uncorr["General classification"] == land]

                keys = pd.merge(
                    df_corr_land[["SITE_ID", "DATE"]],
                    df_uncorr_land[["SITE_ID", "DATE"]],
                    on=["SITE_ID", "DATE"]
                )
                df_corr_land = df_corr_land.merge(keys, on=["SITE_ID", "DATE"])
                df_uncorr_land = df_uncorr_land.merge(keys, on=["SITE_ID", "DATE"])

                for mode_label, df in [('corrected', df_corr_land), ('uncorrected', df_uncorr_land)]:
                    d = df[["Closed", m]].dropna()
                    if d.empty or len(d) < 2 or np.all(d["Closed"] == 0) or np.std(d["Closed"]) == 0:
                        continue

                    x = d["Closed"].values
                    y = d[m].values

                    try:
                        slope, *_ = np.linalg.lstsq(x[:, np.newaxis], y, rcond=None)
                        slope = slope[0]
                    except np.linalg.LinAlgError:
                        slope = np.nan

                    r2 = pearsonr(x, y)[0] ** 2

                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode='markers',
                        marker=dict(color=color_map[mode_label], 
                            symbol=symbol_map[mode_label], size=5, line=dict(width=0.5),
                            opacity=0.5),
                        name=mode_label.capitalize(),
                        legendgroup=mode_label,
                        showlegend=(idx == 0)
                    ), row=row, col=col)

                    # LSLR line
                    x_line = np.linspace(min(x), max(x), 100)
                    fig.add_trace(go.Scatter(
                        x=x_line, y=slope * x_line,
                        mode='lines',
                        line=dict(color=color_map[mode_label], dash='dot'),
                        showlegend=False
                    ), row=row, col=col)

                    label_prefix = 'c' if mode_label == 'corrected' else 'u'
                    fig.add_annotation(
                        text=f"$r^2_{{{label_prefix}}} = {r2:.2f}$",
                        xref='x domain', yref='y domain',
                        x=0.03, y=offset_y_r2[mode_label],
                        showarrow=False,
                        font=dict(size=12, color=color_map[mode_label]),
                        row=row, col=col
                    )
                    fig.add_annotation(
                        text=f"$m_{{{label_prefix}}} = {slope:.2f}$",
                        xref='x domain', yref='y domain',
                        x=0.03, y=offset_y_m[mode_label],
                        showarrow=False,
                        font=dict(size=12, color=color_map[mode_label]),
                        row=row, col=col
                    )

                # 1:1 Line
                fig.add_trace(go.Scatter(
                    x=x_range, y=y_range,
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    name='1:1 Line',
                    showlegend=(idx == 0)
                ), row=row, col=col)

                fig.update_xaxes(range=x_range, tickvals=tick_vals, row=row, col=col)
                fig.update_yaxes(range=y_range, tickvals=tick_vals, row=row, col=col)

            fig.update_layout(
                height=600, width=1000,
                showlegend=True,
                legend=dict(orientation="v", x=1.03, y=0.95),
                margin=dict(l=60, r=60, t=60, b=60)
            )

            out_path = plot_dir / f"{frequency}_{m}_vs_flux_ET_all_land_types.jpg"
            fig.write_image(str(out_path), scale=4)

