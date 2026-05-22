"""
Script to generate maps of weather observation coverage for each variable in
the CONUS-AgWeather_v1 dataset. For every variable in the standardized data,
produces a 3-panel figure showing:
  (a) total days of valid observations,
  (b) years of valid observations,
  (c) average annual record completeness (%).

Stats are computed once across all station files and cached to
``Plots/Variable_Maps/variable_stats.csv`` so re-plotting is fast.

Author: Christian Dunkerly (christian.dunkerly@dri.edu)
Modified by: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)
"""

import os
from multiprocessing import Pool

import pandas as pd
import numpy as np
import geopandas
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from matplotlib.lines import Line2D


# (short_name_for_plot_filename, exact_column_name_in_excel)
VARIABLES = [
    ('ETo', 'ETo (mm/day)'),
    ('ETr', 'ETr (mm/day)'),
    ('TMax', 'TMax (C)'),
    ('TAvg', 'TAvg (C)'),
    ('TMin', 'TMin (C)'),
    ('Ea', 'Ea (kPa)'),
    ('TDew', 'TDew (C)'),
    ('RHMax', 'RHMax (%)'),
    ('RHAvg', 'RHAvg (%)'),
    ('RHMin', 'RHMin (%)'),
    ('Compiled Ea', 'Compiled Ea (kPa)'),
    ('Rs', 'Rs (w/m2)'),
    ('Optimized TR Rs', 'Optimized TR Rs (w/m2)'),
    ('Rso', 'Rso (w/m2)'),
    ('Measured Uz', 'Measured Uz (m/s)'),
    ('Anemometer Height', 'Anemometer Height (m)'),
    ('Uz at 2m', 'Uz at 2m (m/s)'),
    ('Precipitation', 'Precipitation (mm)'),
]


def _compute_station_stats(args):
    """Worker: returns a dict of per-variable stats for one station.

    "Average annual record completeness" follows the convention used by the
    original station_map: for each calendar year present in the station's
    record, the per-year ratio is (valid days for this variable) / (days the
    station was active that year). The average across all such years is the
    station's completeness for that variable. Years where the station was
    active but the variable was not yet collected contribute 0% — this is
    intentional and is what makes the per-variable completeness sensitive to
    late-added variables (e.g. ETr added years into a station's life).
    """
    station_filepath, station_id = args
    try:
        df = pd.read_parquet(station_filepath)
    except Exception as exc:
        print(f'  ! failed to read {station_filepath}: {exc}')
        return None

    df['Date'] = pd.to_datetime(df['Date'])
    df['_year'] = df['Date'].dt.year
    # Days the station was active in each calendar year (rows in the parquet
    # represent the station's full operating record).
    yearly_active = df.groupby('_year').size()

    row = {'Station': station_id}
    for short, col in VARIABLES:
        if col not in df.columns:
            row[f'{short}__obs_count'] = 0
            row[f'{short}__years'] = 0.0
            row[f'{short}__completeness'] = 0.0
            continue

        mask = df[col].notna()
        obs_count = int(mask.sum())
        years = obs_count / 365.0  # matches metadata definition (obs / 365)

        if obs_count == 0:
            completeness = 0.0
        else:
            yearly_obs = (df.loc[mask].groupby('_year').size()
                          .reindex(yearly_active.index, fill_value=0))
            yearly_ratio = yearly_obs / yearly_active * 100.0
            completeness = float(yearly_ratio.mean())

        row[f'{short}__obs_count'] = obs_count
        row[f'{short}__years'] = years
        row[f'{short}__completeness'] = completeness

    return row


def build_stats_dataframe(data_dir, n_workers=8):
    metadata = pd.read_csv(os.path.join(data_dir, 'metadata_for_publication.csv'))
    parquet_dir = os.path.join(data_dir, 'standardized_data_parquet')

    tasks = []
    for _, m in metadata.iterrows():
        base = os.path.splitext(m['Filename'])[0]
        fp = os.path.join(parquet_dir, f'{base}_corrected.parquet')
        if os.path.isfile(fp):
            tasks.append((fp, m['Station']))

    print(f'Computing variable stats for {len(tasks)} stations using {n_workers} workers...')
    with Pool(n_workers) as pool:
        results = pool.map(_compute_station_stats, tasks)

    results = [r for r in results if r is not None]
    stats_df = pd.DataFrame(results)
    merged = metadata[['Station', 'Latitude', 'Longitude']].merge(stats_df, on='Station')
    return merged


# Fixed bins shared across all variables so legends are directly comparable.
# Out-of-range values are clipped into the lowest/highest bin at plot time.
FIXED_BINS = {
    'count': [400, 3000, 6000, 9000, 12000, 15000],
    'years': [1, 5, 10, 15, 20, 25],
    # First edge is replaced at runtime with the global completeness minimum
    # observed across all variables, so the lowest bin honestly reflects the
    # data floor rather than implying values can reach 0%.
    'completeness': [None, 40, 70, 80, 90, 95, 100],
}


def _compute_bins(kind):
    return list(FIXED_BINS[kind])


def _set_completeness_floor(merged):
    """Set the lowest completeness bin edge to the global minimum across all
    variables, considering only stations that actually have data for that
    variable. Floored to the nearest integer for a clean legend label."""
    comp_cols = [f'{short}__completeness' for short, _ in VARIABLES]
    obs_cols = [f'{short}__obs_count' for short, _ in VARIABLES]
    mins = []
    for comp_col, obs_col in zip(comp_cols, obs_cols):
        if comp_col in merged.columns and obs_col in merged.columns:
            sub = merged.loc[merged[obs_col] > 0, comp_col]
            if len(sub):
                mins.append(sub.min())
    if mins:
        FIXED_BINS['completeness'][0] = int(np.floor(min(mins)))


def _format_bin_label(start_val, end_val, kind):
    if kind == 'count':
        return f'{int(start_val):,} – {int(end_val):,}'
    return f'{int(round(start_val))} – {int(round(end_val))}'


def plot_variable_map(merged, contiguous_states, short_name, out_dir):
    """Generate one 3-panel figure for a single variable."""
    obs_col = f'{short_name}__obs_count'
    yr_col = f'{short_name}__years'
    comp_col = f'{short_name}__completeness'

    panels = [
        (obs_col, f'Days of {short_name} observations', '(a)', 'count'),
        (yr_col, f'Years of {short_name} observations', '(b)', 'years'),
        (comp_col, f'Average annual {short_name} record completeness (%)', '(c)', 'completeness'),
    ]

    # Only stations with at least one valid observation of this variable
    # should appear on the map. Without this filter, stations with zero obs
    # get clipped up to the lowest bin edge and falsely plotted as if they
    # had data (notably affects RHAvg/RHMax/RHMin).
    with_data = merged.loc[merged[obs_col] > 0].copy()
    if len(with_data) == 0:
        print(f'  no stations have data for {short_name}, skipping')
        return

    bins_per_panel = [_compute_bins(kind) for _, _, _, kind in panels]

    plt.rcParams.update({'font.size': 22})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    fig, axes = plt.subplots(3, 1, figsize=(25, 36))

    geometry = [Point(xy) for xy in zip(with_data['Longitude'], with_data['Latitude'])]
    gdf = geopandas.GeoDataFrame(with_data, geometry=geometry, crs='EPSG:4326').to_crs('ESRI:102004')

    for i, ((col, title, label, kind), bins) in enumerate(zip(panels, bins_per_panel)):
        ax = axes[i]
        contiguous_states.plot(ax=ax, color='#e3e3e3', edgecolor='black')

        if bins is None or (gdf[col] > 0).sum() == 0:
            ax.set_title(f'{label}  (insufficient data)',
                         fontdict={'fontsize': '30', 'fontweight': 'bold'})
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue

        # Keep the original 5-color prism_r ramp for the meaningful range
        # (>= 40% for completeness, full range for count/years). For
        # completeness, prepend black so the sub-40% outlier bin stands out
        # rather than blending with a similar prism_r hue.
        base_cmap = matplotlib.colormaps.get_cmap('prism_r')
        ramp_colors = [base_cmap(j / 4) for j in range(5)]
        if kind == 'completeness':
            colors = [(0.0, 0.0, 0.0, 1.0)] + ramp_colors
        else:
            colors = ramp_colors
        discrete_cmap = mcolors.ListedColormap(colors)

        data = gdf[col].clip(lower=bins[0], upper=bins[-1])
        gdf['_cat_'] = pd.cut(data, bins=bins, labels=False, include_lowest=True)
        plot_gdf = gdf.dropna(subset=['_cat_'])

        plot_gdf.plot(
            ax=ax,
            column='_cat_',
            cmap=discrete_cmap,
            marker='o',
            markersize=80,
            alpha=0.9,
            edgecolors='black',
            linewidth=0.5,
            legend=False,
            categorical=True,
        )

        new_handles, new_labels = [], []
        for idx in range(len(bins) - 1):
            new_labels.append(_format_bin_label(bins[idx], bins[idx + 1], kind))
            new_handles.append(Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=colors[idx], markersize=15, alpha=0.9))

        ax.legend(new_handles, new_labels,
                  title=title,
                  loc='center left',
                  bbox_to_anchor=(0.9, 0.5),
                  fontsize=30,
                  frameon=False,
                  title_fontsize=30)

        ax.set_title(label, fontdict={'fontsize': '30', 'fontweight': 'bold'})
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    plt.tight_layout()
    safe = short_name.replace(' ', '_')
    fig.savefig(os.path.join(out_dir, f'{safe}_map.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    data_dir = '../../Data/CONUS-AgWeather_v1'
    out_dir = '../../Plots/Variable_Maps'
    os.makedirs(out_dir, exist_ok=True)

    stats_csv = os.path.join(out_dir, 'variable_stats.csv')
    if os.path.exists(stats_csv):
        print(f'Loading cached variable stats from {stats_csv}')
        merged = pd.read_csv(stats_csv)
    else:
        merged = build_stats_dataframe(data_dir, n_workers=8)
        merged.to_csv(stats_csv, index=False)
        print(f'Cached variable stats to {stats_csv}')

    _set_completeness_floor(merged)
    print(f"Completeness bin edges: {FIXED_BINS['completeness']}")

    states = geopandas.read_file('../../Data/states/states.shp')
    contiguous_states = states[~states['STATE_ABBR'].isin(['AK', 'HI'])].to_crs('ESRI:102004')

    for short, _ in VARIABLES:
        print(f'Plotting {short}...')
        plot_variable_map(merged, contiguous_states, short, out_dir)

    print(f'\nDone. Figures written to {out_dir}')
