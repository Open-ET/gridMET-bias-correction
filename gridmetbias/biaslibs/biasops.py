"""
Contains functons related to bias ratio analysis and correlation plots.
Author: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)
"""

import pandas as pd
import re
import geopandas as gpd
import seaborn as sns
import calendar
import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from matplotlib.ticker import FuncFormatter
from xarray import corr
from .geeops import get_irr_crop_data
from glob import glob
from sklearn.metrics import mean_absolute_error


def correlation_matrix_with_pvalues(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    show_all_pvalues: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the correlation matrix with p-values for two DataFrames.

    Args:
        df1: The first DataFrame.
        df2: The second DataFrame.
        show_all_pvalues: A boolean indicating whether to show all
        p-values or only those less than 0.05.

    Returns:
        A tuple containing the correlation matrix and the p-value matrix 
        diagonal values as DataFrames.
    """
    corr_matrix = pd.DataFrame(
        np.full((df1.shape[1], df2.shape[1]), np.nan), 
        columns=df2.columns, 
        index=df1.columns
    )
    pvalue_matrix = pd.DataFrame(
        np.full((df1.shape[1], df2.shape[1]), np.nan), 
        columns=df2.columns, 
        index=df1.columns
    )

    for col1 in df1.columns:
        for col2 in df2.columns:
            corr, pvalue = pearsonr(df1[col1], df2[col2])
            if show_all_pvalues:
                corr_matrix.loc[col1, col2] = corr
                pvalue_matrix.loc[col1, col2] = pvalue
            elif pvalue < 0.05:
                    corr_matrix.loc[col1, col2] = corr
                    pvalue_matrix.loc[col1, col2] = pvalue
    corr_diagonal = np.diag(corr_matrix.values)
    pvalue_diagonal = np.diag(pvalue_matrix.values)

    # Convert the diagonal values to DataFrames
    corr_diagonal_df = pd.DataFrame(
        corr_diagonal, 
        index=corr_matrix.index
    )
    pvalue_diagonal_df = pd.DataFrame(
        pvalue_diagonal, 
        index=pvalue_matrix.index
    )
    return corr_diagonal_df, pvalue_diagonal_df


def plot_bias_corr_matrix_lon(
    et_files: list[str], 
    other_files: list[str], 
    month_col_pattern: str, 
    plot_dir: str,
    show_all_pvalues: bool = False,
    annot_pvalues: bool = True
) -> None:
    """
    Plot the Pearson correlation matrices between the monthly means of the 
    reference ET bias ratio and other variables' 
    bias ratio based on longitude.
    
    Args:
        et_files: A list of file paths to the reference ET data.
        other_files: A list of file paths to the other variables data.
        month_col_pattern: A regular expression pattern to match the columns 
        containing the monthly means.
        plot_dir: A string representing the directory to save the plots.
        show_all_pvalues (bool): A boolean indicating whether to show all
        p-values or only those less than 0.05.
        annot_pvalues (bool): A boolean indicating whether to annotate the
        heatmaps with p-values.

    Returns:
        None
    """
    # Initialize variables to store global min and max correlation values
    global_min = float('inf')
    global_max = float('-inf')
    lon_col = 'STATION_LON'
    station_col = 'STATION_ID'
    et_df_names = ['Western', 'Eastern']
    plot_dir = f'{plot_dir}East_vs_West/All_pval_{show_all_pvalues}/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    data_dict = {}
    for et_file in et_files:
        et_df_all = pd.read_csv(et_file)
        et_df_west = et_df_all[et_df_all[lon_col] < -100]
        et_df_east = et_df_all[et_df_all[lon_col] >= -100]
        et_df_list = [et_df_west, et_df_east]
        for et_df, et_df_name in zip(et_df_list, et_df_names):
            for other_file in other_files:
                other_df = pd.read_csv(other_file)
                if et_df_name == 'Western':
                    other_df = other_df[other_df[lon_col] < -100]
                else:
                    other_df = other_df[other_df[lon_col] >= -100]
                common_stations = set(et_df[station_col]).intersection(
                    set(other_df[station_col])
                )
                monthly_means = other_df[
                    other_df[station_col].isin(common_stations)
                ].sort_values(by=station_col) \
                .filter(regex=month_col_pattern)
                et_monthly_means = et_df[
                    et_df[station_col].isin(common_stations)
                ].sort_values(by=station_col) \
                .filter(regex=month_col_pattern)
                monthly_means.columns = [
                    calendar.month_name[m] for m in range(1, 13)
                ]
                et_monthly_means.columns = monthly_means.columns
                corr_matrix, pvalue_matrix = correlation_matrix_with_pvalues(
                    et_monthly_means, monthly_means, show_all_pvalues
                )
                annotations = corr_matrix.round(2).astype(str)
                if annot_pvalues:
                    annotations += "  (p=" + \
                        pvalue_matrix.map(lambda x: f"{x:.2e}").astype(str) + ")"
                et_name = et_file.split('/')[-1].split('_')[0].upper()
                var_name = other_file.split('/')[-1].split('_')[0].upper()
                data_dict[f'{et_name}_{var_name}_{et_df_name}'] = (
                    corr_matrix,
                    annotations
                )                
                min_corr = corr_matrix.min().min()
                max_corr = corr_matrix.max().max()
                global_min = min(global_min, min_corr)
                global_max = max(global_max, max_corr)

    var_name_dict = {
        'ETO': 'ETo',
        'ETR': 'ETr',
        'EA': 'ea',
        'TMIN': 'tmin',
        'TMAX': 'tmax',
        'U2': 'u2',
        'SRAD': 'srad'
    }
    for key, value in data_dict.items():
        et_name, var_name, et_df_name = key.split('_')
        corr_matrix, annotations = value
        figsize = (14, 8) if annot_pvalues else (10, 8)
        plt.figure(figsize=figsize)
        plt.rcParams.update({'font.size': 16})
        sns.heatmap(
            corr_matrix, annot=annotations, 
            cmap='coolwarm', fmt='',
            vmin=global_min, vmax=global_max
        )
        if var_name in ['TMIN', 'TMAX']:
            vname = f'{var_name_dict[var_name]} bias difference'
        else:
            vname = f'{var_name_dict[var_name]} bias ratio'
        plt.ylabel(f'{vname} ({et_df_name} CONUS)')
        plt.xlabel(f'{vname} ({et_df_name} CONUS)')
        plt.xticks([])
        plt.tight_layout()
        plt.savefig(
            f'{plot_dir}{key}_corr.png', 
            dpi=300
        )
        plt.clf()
        plt.close()


def plot_bias_corr_matrix_climate(
    et_files: list[str], 
    other_files: list[str], 
    climate_shp_file: str,
    month_col_pattern: str, 
    plot_dir: str,
    show_all_pvalues: bool = False,
    annot_pvalues: bool = True
) -> None:
    """
    Plot the Pearson correlation matrices between the monthly means of the 
    reference ET bias ratio and other variables' 
    bias ratio based on Koppen climate classifications.
    
    Args:
        et_files: A list of file paths to the reference ET data.
        other_files: A list of file paths to the other variables data.
        climate_shp_file: A string representing the file path to the shapefile 
        containing the Koppen climate classifications.
        month_col_pattern: A regular expression pattern to match the columns 
        containing the monthly means.
        plot_dir: A string representing the directory to save the plots.
        show_all_pvalues (bool): A boolean indicating whether to show all
        p-values or only those less than 0.05.
        annot_pvalues (bool): A boolean indicating whether to annotate the
        heatmaps with p-values.

    Returns:
        None
    """
     # Initialize variables to store global min and max correlation values
    global_min = float('inf')
    global_max = float('-inf')
    lat_col = 'STATION_LAT'
    lon_col = 'STATION_LON'
    station_col = 'STATION_ID'
    plot_dir = f'{plot_dir}Climate/All_pval_{show_all_pvalues}/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Read the climate shapefile and assign all the data points to the climate
    climate_gdf = gpd.read_file(climate_shp_file).to_crs('EPSG:4326')
    climate_col = 'gridcode'
    climate_gdf[climate_col] = climate_gdf[climate_col].astype(int)
    csv_climate_dict = {}
    group_climate_dict = {
        6: 5,
        8: 7,
        13: 12,
        19: 18,
    }
    climate_dict_names = {
        5: ('BSh + BSk', 'Hot and cold semi-arid (steppe) climates'),
        7: ('BWh + BWk', 'Hot and cold desert climates'),
        9: ('Cfa', 'Humid subtropical climate'),
        12: ('Csa + Csb', 'Hot- and warm-summer Mediterranean climates'),
        18: ('Dfa + Dfb', 'Hot- and warm-summer humid continental climates'),
    }
    for csv_file in et_files + other_files:
        climate_dir = os.path.dirname(csv_file) + '/Climate/'
        if not os.path.exists(climate_dir):
            os.makedirs(climate_dir)
        csv_name = csv_file.split(os.sep)[-1].split('.')[0]
        climate_file = f'{climate_dir}{csv_name}_climate.geojson'
        if not os.path.exists(climate_file):
            df = pd.read_csv(csv_file)
            data_gdf = gpd.GeoDataFrame(
                df, 
                geometry=gpd.points_from_xy(
                    df[lon_col], 
                    df[lat_col]
                ),
                crs=climate_gdf.crs
            )
            data_gdf = gpd.sjoin(
                data_gdf, 
                climate_gdf,
                how='inner', 
                predicate='within'
            ).clip(data_gdf.total_bounds) \
            .drop(columns='index_right')
            data_gdf[climate_col] = data_gdf[climate_col].replace(
                group_climate_dict
            )        
            data_gdf.to_file(climate_file)
        else:
            data_gdf = gpd.read_file(climate_file)
        df = data_gdf.drop(columns='geometry')
        climate_file_name = climate_file[
            climate_file.rfind('/') + 1 : climate_file.rfind('.')
        ].split('_')[0].upper()
        csv_climate_dict[climate_file_name] = df
    
    # Get the ET and other variables' keys
    et_keys = [k for k in csv_climate_dict.keys() if 'ET' in k]
    other_keys = [k for k in csv_climate_dict.keys() if k not in et_keys]
    data_dict = {}
    for et_key in et_keys:
        et_df = csv_climate_dict[et_key]
        et_df = et_df[et_df[climate_col].isin(climate_dict_names.keys())]
        for climate in et_df[climate_col].unique():
            for other_key in other_keys:
                other_df = csv_climate_dict[other_key].copy(deep=True)
                common_stations = set(et_df[station_col]).intersection(
                    set(other_df[station_col])
                )
                monthly_means = other_df[
                    (other_df[station_col].isin(common_stations)) &
                    (other_df[climate_col] == climate)
                ].sort_values(by=station_col) \
                .filter(regex=month_col_pattern)
                et_monthly_means = et_df[
                    (et_df[station_col].isin(common_stations)) &
                    (et_df[climate_col] == climate)
                ].sort_values(by=station_col) \
                .filter(regex=month_col_pattern)
                monthly_means.columns = [
                    calendar.month_name[m] for m in range(1, 13)
                ]
                et_monthly_means.columns = monthly_means.columns
                corr_matrix, pvalue_matrix = correlation_matrix_with_pvalues(
                    et_monthly_means, monthly_means, show_all_pvalues
                )
                annotations = corr_matrix.round(2).astype(str)
                if annot_pvalues:
                    annotations += "  (p=" + \
                        pvalue_matrix.map(lambda x: f"{x:.2e}").astype(str) + ")"
                data_dict[f'{et_key}_{other_key}_C{climate}'] = (
                    corr_matrix,
                    annotations.astype(str)
                )
                min_corr = corr_matrix.min().min()
                max_corr = corr_matrix.max().max()
                global_min = min(global_min, min_corr)
                global_max = max(global_max, max_corr)

    var_name_dict = {
        'ETO': 'ETo',
        'ETR': 'ETr',
        'EA': 'ea',
        'TMIN': 'tmin',
        'TMAX': 'tmax',
        'U2': 'u2',
        'SRAD': 'srad'
    }
    # Second pass to create the plots with the same colorbar scale
    for key, value in data_dict.items():
        et_key, other_key, climate = key.split('_')
        corr_matrix, annotations = value
        plot_file = f'{plot_dir}{et_key}_{other_key}_{climate}_corr.png'
        climate_code, climate_name = climate_dict_names[int(climate[1:])]
        # Plot the correlation matrix
        figsize = (14, 8) if annot_pvalues else (10, 8)
        plt.figure(figsize=figsize)
        plt.rcParams.update({'font.size': 16})
        sns.heatmap(
            corr_matrix, annot=annotations, 
            cmap='coolwarm', fmt='',
            vmin=global_min, vmax=global_max
        )
        if other_key in ['TMIN', 'TMAX']:
            vname = f'{var_name_dict[other_key]} bias difference'
        else:
            vname = f'{var_name_dict[other_key]} bias ratio'
        plt.ylabel(f'{var_name_dict[et_key]} bias ratio')
        plt.xlabel(f'{vname}')
        plt.xticks([])
        plt.title(f'{climate_code}: {climate_name}', fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.clf()
        plt.close()


def plot_bias_corr_matrix_all(
    et_files: list[str],
    other_files: list[str],
    month_col_pattern: str,
    plot_dir: str,
    show_all_pvalues: bool = False,
    annot_pvalues: bool = True
) -> None:
    """
    Plot the Pearson correlation matrices between the monthly means of the
    reference ET bias ratio and other variables' bias ratio without any 
    spatial or climate classification.

    Args:
        et_files: A list of file paths to the reference ET data.
        other_files: A list of file paths to the other variables data.
        month_col_pattern: A regular expression pattern to match the columns
        containing the monthly means.
        plot_dir: A string representing the directory to save the plots.
        show_all_pvalues (bool): A boolean indicating whether to show all
        p-values or only those less than 0.05.
        annot_pvalues (bool): A boolean indicating whether to annotate the
        heatmaps with p-values.

    Returns:
        None
    """
    plot_dir = f'{plot_dir}Correlation_Plots_All/All_pval_{show_all_pvalues}/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Initialize variables to store global min and max correlation values
    global_min = float('inf')
    global_max = float('-inf')
    station_col = 'STATION_ID'

    data_dict = {}
    for et_file in et_files:
        et_df = pd.read_csv(et_file)
        for other_file in other_files:
            other_df = pd.read_csv(other_file)
            common_stations = set(et_df[station_col]).intersection(
                    set(other_df[station_col])
            )
            monthly_means = other_df[
                other_df[station_col].isin(common_stations)
            ].sort_values(by=station_col) \
            .filter(regex=month_col_pattern)
            et_monthly_means = et_df[
                et_df[station_col].isin(common_stations)
            ].sort_values(by=station_col) \
            .filter(regex=month_col_pattern)
            monthly_means.columns = [
                calendar.month_name[m] for m in range(1, 13)
            ]
            et_monthly_means.columns = monthly_means.columns
            corr_matrix, pvalue_matrix = correlation_matrix_with_pvalues(
                et_monthly_means, monthly_means, show_all_pvalues
            )
            annotations = corr_matrix.round(2).astype(str)
            if annot_pvalues:
                annotations += "  (p=" + \
                    pvalue_matrix.map(lambda x: f"{x:.2e}").astype(str) + ")"
            et_name = et_file.split('/')[-1].split('_')[0].upper()
            var_name = other_file.split('/')[-1].split('_')[0].upper()
            data_dict[f'{et_name}_{var_name}'] = (
                corr_matrix,
                annotations
            )
            min_corr = corr_matrix.min().min()
            max_corr = corr_matrix.max().max()
            global_min = min(global_min, min_corr)
            global_max = max(global_max, max_corr)

    var_name_dict = {
        'ETO': 'ETo',
        'ETR': 'ETr',
        'EA': 'ea',
        'TMIN': 'tmin',
        'TMAX': 'tmax',
        'U2': 'u2',
        'SRAD': 'srad'
    }
    # Second pass to create the plots with the same colorbar scale
    for et_file in et_files:
        et_name = et_file.split('/')[-1].split('_')[0].upper()
        
        # Create subplots
        plt.rcParams.update({'font.size': 16})
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(25, 15))
        axes = axes.flatten()
        
        for ax_idx, other_file in enumerate(other_files):
            var_name = other_file.split('/')[-1].split('_')[0].upper()
            corr_matrix, annotations = data_dict[f'{et_name}_{var_name}']
            sns.heatmap(
                corr_matrix, 
                annot=annotations, fmt='', 
                cmap='coolwarm', 
                vmin=global_min, vmax=global_max, 
                ax=axes[ax_idx], 
                cbar=False
            )
            if var_name in ['TMIN', 'TMAX']:
                vname = f'{var_name_dict[var_name]} bias difference'
            else:
                vname = f'{var_name_dict[var_name]} bias ratio'
            axes[ax_idx].set_ylabel(f'{var_name_dict[et_name]} bias ratio', fontsize=16)
            axes[ax_idx].set_xlabel(f'{vname}', fontsize=16)
            axes[ax_idx].set_xticks([])

        # Add a colorbar to the last axis
        x0, y0, width, height = axes[-1].get_position().bounds
        fig.delaxes(axes[-1])
        cbar_ax = fig.add_axes([x0, y0 * 2, width, height * 0.1])
        norm = plt.Normalize(vmin=global_min, vmax=global_max)
        sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, cax=cbar_ax, orientation='horizontal', 
            label='Pearson correlation coefficient'
        )
        cbar.ax.xaxis.set_label_position('top')
        plt.savefig(f'{plot_dir}{et_name}_corr_subplots.png', dpi=300)
        plt.clf()
        plt.close()


def aggregate_flux_et_data(
        flux_et_file: str,
        output_dir: str,
        corr_type_col: str = 'gridMET Corr_Uncorr',
        date_col: str = 'DATE',
        lat_col: str = 'Latitude',
        lon_col: str = 'Longitude',
        station_col: str = 'SITE_ID',
        station_eto_col: str = 'ASCE_ETo',
        gridmet_eto_col: str = 'GRIDMET_REFERENCE_ET',
        season_col: str = 'summer_mean',
        lulc_col: str = 'General classification'
    ) -> list[str]:
    """
    Aggregate the flux ET data into corrected and uncorrected DataFrames.

    Args:
        flux_et_file: A string representing the file path to the flux ET data.
        corr_type_col: The column name for the correction type in the flux ET
        date_col: The column name for the date in the flux ET data.
        lat_col: The column name for the latitude in the flux ET data.
        lon_col: The column name for the longitude in the flux ET data.
        station_col: The column name for the station ID in the flux ET data.
        station_eto_col: The column name for the station ET in the flux ET data.
        gridmet_eto_col: The column name for the gridMET ET in the flux ET data.
        season_col: The name of the season column. Options are annual_mean, growseason_mean, and summer_mean.

    Returns:
        A list containing the file paths to the corrected and uncorrected csv files.
    """
    flux_df = pd.read_csv(flux_et_file)
    flux_df = flux_df[flux_df[lulc_col] == 'Croplands'] # we only want cropland stations
    flux_df = flux_df[[
        station_col, date_col,
        lat_col, lon_col, 
        station_eto_col, gridmet_eto_col,
        corr_type_col
    ]]
    eto_ratio_col = 'ETo_bias_ratio'
    flux_df[eto_ratio_col] = flux_df[gridmet_eto_col] / flux_df[station_eto_col]
    corrected_df = flux_df[flux_df[corr_type_col] == "Corrected"].copy()
    uncorrected_df = flux_df[flux_df[corr_type_col] == "Uncorrected"].copy()
    flux_df_list = [corrected_df, uncorrected_df]
    month_col = 'Month'
    year_col = 'Year'
    os.makedirs(output_dir, exist_ok=True)
    corrected_csv = f'{output_dir}Corrected ETo_Flux_Aggregated_{season_col}.csv'
    uncorrected_csv = f'{output_dir}Uncorrected ETo_Flux_Aggregated_{season_col}.csv'
    output_files = [corrected_csv, uncorrected_csv]
    for flux_df, out_file in zip(flux_df_list, output_files):
        flux_df[date_col] = pd.to_datetime(flux_df[date_col])
        flux_df[month_col] = flux_df[date_col].dt.month
        flux_df[year_col] = flux_df[date_col].dt.year
        if season_col == 'growseason_mean':
            time_df = flux_df[flux_df[month_col].isin(range(4, 11))]            
        elif season_col == 'summer_mean':
            time_df = flux_df[flux_df[month_col].isin([6, 7, 8])]
        else:
            time_df = flux_df.copy()
        agg_df = time_df.groupby([station_col]).agg({
            eto_ratio_col: 'mean',
            lat_col: 'first',
            lon_col: 'first'
        }).reset_index()
        agg_df = agg_df.rename(columns={
            eto_ratio_col: season_col,
            station_col: 'STATION_ID',
            lon_col: 'STATION_LON',
            lat_col: 'STATION_LAT'
        })
        agg_df['start_year'] = time_df.groupby([station_col])[year_col].min().values
        agg_df['end_year'] = time_df.groupby([station_col])[year_col].max().values
        agg_df.to_csv(out_file, index=False)
        agg_gdf = gpd.GeoDataFrame(
            agg_df,
            geometry=gpd.points_from_xy(
                agg_df['STATION_LON'],
                agg_df['STATION_LAT']
            ),
            crs='EPSG:4326'
        )
        geojson_file = out_file.replace('.csv', '.geojson')
        agg_gdf.to_file(geojson_file, driver='GeoJSON')
    return output_files


def plot_irr_crop_bias_distributions(
    et_files: list[str],
    other_files: list[str],
    plot_dir: str,
    gcloud_project: str = 'ee-grid-obs-comp',
    start_year_col: str = 'start_year',
    end_year_col: str = 'end_year',
    lat_col: str = 'STATION_LAT',
    lon_col: str = 'STATION_LON',
    station_col: str = 'STATION_ID',
    season_col: str = 'summer_mean',
    climate_col: str = 'gridcode',
    verbose: bool = False,
    flux_et: bool = False
) -> None:
    """
    Plot the distribution of the bias ratios for the reference ET and other
    variables based on the IrrMapper, LANID, and CDL data.

    Args:
        et_files: A list of file paths to the reference ET data.
        other_files: A list of file paths to the other variables data.
        plot_dir: A string representing the directory to save the plots.
        gcloud_project: The Google Cloud project ID.
        start_year_col: The column name for the start year in the bias file.
        end_year_col: The column name for the end year in the bias file.
        lat_col: The column name for the latitude in the bias file.
        lon_col: The column name for the longitude in the bias file.
        station_col: The column name for the station ID in the bias file.
        growing_season_col: The column name for the growing season in the bias 
        file.
        season_col: The name of the season column. Options are annual_mean, growseason_mean, and summer_mean.
        climate_col: The column name for the climate classification in the bias
        file.
        verbose: A boolean indicating whether to print the progress messages.
        flux_et: A boolean indicating whether the data is from flux ET dataset. If True additional aggregations are performed.

    Returns:
        None.
    """

    if flux_et:
        csv_files = aggregate_flux_et_data(
            et_files[0],
            output_dir=f'{os.path.dirname(et_files[0])}/Aggregated_Flux_ET/',
            season_col=season_col
        )            
    else:
        csv_files = et_files + other_files
    crop_irr_type = ['All', 'IrrComp', 'NoIrr', 'Irr']
    common_station_dict = {}
    bias_ratio_dict = {}
    for ctype in crop_irr_type:
        common_station_dict[ctype] = set()
        bias_ratio_dict[ctype] = {}
    plot_dir = f'{plot_dir}Crop_Bias_Distributions/'
    os.makedirs(plot_dir, exist_ok=True)
    var_name_dict = {
        'ETO': 'ETo',
        'CORRECTED ETO': 'Corrected ETo',
        'UNCORRECTED ETO': 'Uncorrected ETo',
        'ETR': 'ETr',
        'EA': 'ea',
        'TMIN': 'tmin',
        'TMAX': 'tmax',
        'U2': 'u2',
        'SRAD': 'srad'
    }
    season_name = season_col.split('_')[0]
    if season_name == 'growseason':
        season_name = 'growingseason'
    irr_crop_cols = ['Irrigation', 'Crop Type']
    irr_frac_col = 'Irrigation Fraction'
    is_irrigated_col = 'Irrigated'
    subset_cols = [
        station_col, 
        lat_col, 
        lon_col,
        start_year_col, 
        end_year_col,
        irr_frac_col,
        is_irrigated_col
    ] + irr_crop_cols
    if not flux_et:
        subset_cols.append(climate_col)
    for csv_file in csv_files:
        csv_name = csv_file.split(os.sep)[-1].split('.')[0] 
        if not flux_et:
            climate_dir = os.path.dirname(csv_file) + '/Climate/'
            bias_crop_name = f'{climate_dir}{csv_name}_climate'
        else:
            climate_dir = os.path.dirname(csv_file) + '/'
            bias_crop_name = f'{climate_dir}{csv_name}'       
        vector_file = f'{bias_crop_name}.geojson'
        bias_all_crop_csv =  f'{bias_crop_name}_all_crop_{season_name}.csv'
        bias_no_irr_crop_csv =  f'{bias_crop_name}_noirr_crop_{season_name}.csv'
        bias_irr_crop_csv =  f'{bias_crop_name}_irr_crop_{season_name}.csv'
        file_check = os.path.exists(bias_all_crop_csv) and \
            os.path.exists(bias_no_irr_crop_csv) and \
            os.path.exists(bias_irr_crop_csv)
        file_check = False  # always recompute for now
        var_name = csv_file.split('/')[-1].split('_')[0].upper()
        temp_flag = False
        if var_name in ['TMIN', 'TMAX']:
            var_name = f'{var_name_dict[var_name]} bias (°C)'
            temp_flag = True
        else:
            var_name = f'{var_name_dict[var_name]} bias ratio'
        if not file_check:
            output_file = vector_file[: vector_file.rfind('.')] +  \
                '_irr_crop.geojson'
            bias_df = get_irr_crop_data(
                vector_file, 
                output_file, 
                gcloud_project=gcloud_project,
                start_year_col=start_year_col,
                end_year_col=end_year_col,
                lat_col=lat_col,
                lon_col=lon_col,
                station_col=station_col,
                verbose=verbose
            )
            bias_val = bias_df[season_col]
            if temp_flag:
                bias_df[var_name] = -bias_val
            else:
                bias_df[var_name] = 1 / bias_val
            bias_df[is_irrigated_col] = bias_df.apply(
                lambda x: 'Irrigated' if x[irr_frac_col] > 0 else 'Non-Irrigated', axis=1
            )
            bias_all_crop_df = bias_df[subset_cols + [var_name]]            
            bias_all_crop_df.to_csv(bias_all_crop_csv, index=False)
            bias_no_irr_crop_df = bias_all_crop_df[
                (bias_all_crop_df[irr_frac_col]) == 0
            ].drop(columns=[irr_frac_col, irr_crop_cols[0]])
            bias_no_irr_crop_df.to_csv(bias_no_irr_crop_csv, index=False)
            bias_irr_crop_df = bias_all_crop_df[
                (bias_all_crop_df[irr_frac_col]) > 0
            ]
            bias_irr_crop_df.to_csv(bias_irr_crop_csv, index=False)
        else:
            bias_all_crop_df = pd.read_csv(bias_all_crop_csv)
            bias_no_irr_crop_df = pd.read_csv(bias_no_irr_crop_csv)
            bias_irr_crop_df = pd.read_csv(bias_irr_crop_csv)
        bias_df_list = [
            bias_all_crop_df, bias_all_crop_df, 
            bias_no_irr_crop_df, bias_irr_crop_df
        ]
        for bias_df, ctype in zip(bias_df_list, crop_irr_type):
            common_station_dict[ctype].update(bias_df[station_col].unique())
            bias_ratio_dict[ctype][var_name] = bias_df
    if flux_et:
        return # No crop-based plots for flux ET data as only 30 cropland stations are available
    crop_color_dict = {}
    for ctype in crop_irr_type:
        common_stations = common_station_dict[ctype]
        bias_dict = bias_ratio_dict[ctype]
        if ctype == 'All' or ctype == 'NoIrr':
            hue_cols = [irr_crop_cols[1]]
        elif ctype == 'IrrComp':
            hue_cols = [is_irrigated_col]
        else:
            hue_cols = irr_crop_cols
        for hue in hue_cols:
            plt.rcParams.update({'font.size': 16})
            fig, axes = plt.subplots(4, 2, figsize=(10, 15))
            axes = axes.flatten()
            for ax_idx, (var_name, bias_df) in enumerate(bias_dict.items()):            
                bias_df_common = bias_df[
                    bias_df[station_col].isin(common_stations)
                ]
                ax = axes[ax_idx]        
                hue_order =  sorted(bias_df_common[hue].unique()) 
                # swap last two elements
                if len(hue_order) > 1:
                    hue_order[-1], hue_order[-2] = hue_order[-2], hue_order[-1]
                if ctype == 'All':
                    for idx, crop in enumerate(hue_order):
                        crop_color_dict[crop] = sns.color_palette()[idx]
                crop_colors = crop_color_dict             
                if hue == 'Irrigation':
                    crop_colors = sns.color_palette("Set2", len(hue_order))
                if hue == is_irrigated_col:
                    crop_colors = sns.color_palette("vlag")
                    crop_colors =[crop_colors[-1], crop_colors[0]]
                if ctype == 'IrrComp' and hue == is_irrigated_col:
                    # show irrigated first and non-irrigated second
                    hue_order = ['Irrigated', 'Non-Irrigated']
                    crop_colors = crop_colors[::-1]
                sns.boxplot(
                    data=bias_df_common, 
                    y=var_name, 
                    hue=hue,
                    hue_order=hue_order,
                    palette=crop_colors,
                    ax=ax
                )
                var_name = var_name.split(' ')[0]
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda y, _: f'{y:.1f}')
                )
                ax.set_xticks([]) 
                handles, labels = ax.get_legend_handles_labels()
                ax.legend_.remove()      
            x0, y0, width, height = axes[-1].get_position().bounds
            fig.delaxes(axes[-1])
            legend_ax = fig.add_axes([x0, y0, width, height]) 
            ncol = 1 if hue in ['Irrigation', is_irrigated_col] else 2
            legend_title = '' if hue == is_irrigated_col else hue
            legend = legend_ax.legend(
                handles, labels, 
                loc='upper center', 
                ncol=ncol,
                frameon=False
            )
            legend.set_title(legend_title)     
            legend_ax.axis('off')       
            plt.subplots_adjust(
                left=0.12, right=0.98, 
                top=0.98, bottom=0.1, 
                wspace=0.3, hspace=0.1
            )
            if hue == 'Irrigation':
                plot_file = (f'{plot_dir}{ctype}_bias_{season_name}_'
                            f'distributions.png')
            else:
                plot_file = (f'{plot_dir}{ctype}_crop_bias_{season_name}_'
                            f'distributions.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

    climate_dict_names = {
        5: ('BSh + BSk', 'Hot and cold semi-arid (steppe)'),
        7: ('BWh + BWk', 'Hot and cold desert'),
        9: ('Cfa', 'Humid subtropical'),
        12: ('Csa + Csb', 'Hot- and warm-summer Mediterranean'),
        18: ('Dfa + Dfb', 'Hot- and warm-summer humid continental'),
    }
    plot_dir_climate = f'{plot_dir}Climate_IrrBias/'
    os.makedirs(plot_dir_climate, exist_ok=True)
    csv_df = pd.DataFrame()
    for ctype in crop_irr_type:
        common_stations = common_station_dict[ctype]
        bias_dict = bias_ratio_dict[ctype]
        if ctype == 'All' or ctype == 'NoIrr':
            hue_cols = [irr_crop_cols[1]]
        elif ctype == 'IrrComp':
            hue_cols = [is_irrigated_col]
        else:
            hue_cols = irr_crop_cols
        for hue in hue_cols:
            for var_name, bias_df in bias_dict.items():          
                bias_df_common = bias_df[
                    bias_df[station_col].isin(common_stations)
                ]
                min_var = bias_df_common[var_name].min()
                max_var = bias_df_common[var_name].max()
                plt.rcParams.update({'font.size': 16})
                fig, axes = plt.subplots(3, 2, figsize=(10, 15))
                axes = axes.flatten()
                for ax_idx, cl in enumerate(climate_dict_names.keys()):
                    bias_df_climate = bias_df_common[
                        bias_df_common[climate_col] == cl
                    ]
                    ax = axes[ax_idx]        
                    hue_order =  sorted(bias_df_climate[hue].unique())
                    # swap last two elements
                    if len(hue_order) > 1:
                        hue_order[-1], hue_order[-2] = hue_order[-2], hue_order[-1]
                    if ctype == 'All':
                        for idx, crop in enumerate(hue_order):
                            crop_color_dict[crop] = sns.color_palette()[idx]
                    crop_colors = crop_color_dict             
                    if hue == 'Irrigation':
                        crop_colors = sns.color_palette("Set2", len(hue_order))
                        if len(hue_order) == 2:
                            crop_colors = [crop_colors[1], crop_colors[0]]
                    if hue == is_irrigated_col:
                        if cl == 7: # everything is irrigated in desert climate
                            hue_order = ['Irrigated']
                            crop_colors = [sns.color_palette("vlag")[-1]]
                        else:
                            crop_colors = sns.color_palette("vlag")
                            crop_colors =[crop_colors[-1], crop_colors[0]]   
                    ax_title = (f'{climate_dict_names[cl][0]}: '
                                f'{climate_dict_names[cl][1]}') 
                    if 'Corrected' in var_name:
                        vname = 'Corr_ETo'
                    elif 'Uncorrected' in var_name:
                        vname = 'Uncorr_ETo'
                    else:
                        vname = var_name.split(' ')[0]
                    for h in hue_order:
                        sub_df = bias_df_climate[bias_df_climate[hue] == h]                        
                        sub_df_info = sub_df[[var_name]].describe().T
                        sub_df_info['Season'] = season_name
                        sub_df_info['Hue'] = h
                        sub_df_info['Climate'] = ax_title
                        sub_df_info['Variable'] = vname
                        sub_df_info = sub_df_info.iloc[:, ::-1]
                        csv_df = pd.concat([csv_df, sub_df_info])
                    sns.boxplot(
                        data=bias_df_climate, 
                        y=var_name, 
                        hue=hue,
                        hue_order=hue_order,
                        palette=crop_colors,
                        ax=ax
                    )
                    ax.set_ylim(min_var - 0.1, max_var + 0.1)
                    ax.yaxis.set_major_formatter(FuncFormatter(
                        lambda y, _: f'{y:.1f}')
                    )
                    ax.set_xticks([]) 
                    ax.set_title(ax_title, fontsize=12, pad=1.5)
                    handles, labels = ax.get_legend_handles_labels()
                    if ax.legend_:
                        ax.legend_.remove()      
                x0, y0, width, height = axes[-1].get_position().bounds
                fig.delaxes(axes[-1])
                legend_ax = fig.add_axes([x0, y0, width, height]) 
                legend = legend_ax.legend(
                    handles, labels, 
                    loc='upper center', 
                    ncol=1,
                    frameon=False
                )
                legend.set_title(hue if hue != is_irrigated_col else '')     
                legend_ax.axis('off')       
                plt.subplots_adjust(
                    left=0.12, right=0.98, 
                    top=0.98, bottom=0.1, 
                    wspace=0.25, hspace=0.2
                )
                plot_file = (f'{plot_dir_climate}{ctype}_{vname}_{hue.split()[0]}'
                                f'_irr_bias_{season_name}_distributions.png')
                plt.savefig(plot_file, dpi=300)
                plt.close()
    csv_df.to_csv(f'{plot_dir_climate}Climate_Irr_Bias_Summary.csv', index=True)    


def plot_ag_bias_distributions(
    et_files: list[str],
    other_files: list[str],
    plot_dir: str,
    gcloud_project: str = 'ee-grid-obs-comp',
    start_year_col: str = 'start_year',
    end_year_col: str = 'end_year',
    lat_col: str = 'STATION_LAT',
    lon_col: str = 'STATION_LON',
    station_col: str = 'STATION_ID',
    season_col: str = 'summer_mean',
    climate_col: str = 'gridcode',
    num_ag_cats: int = 3,
    flux_et: bool = False
) -> None:
    """
    Plot the distribution of the bias ratios for the reference ET and other
    variables based on the NLCD agriculture fractions and climate 
    classifications.

    Args:
        et_files: A list of file paths to the reference ET data.
        other_files: A list of file paths to the other variables data.
        plot_dir: A string representing the directory to save the plots.
        gcloud_project: The Google Cloud project ID.
        start_year_col: The column name for the start year in the bias file.
        end_year_col: The column name for the end year in the bias file.
        lat_col: The column name for the latitude in the bias file.
        lon_col: The column name for the longitude in the bias file.
        station_col: The column name for the station ID in the bias file.
        season_col: The name of the season column. Options are annual_mean, growseason_mean, and summer_mean.
        climate_col: The column name for the climate classification in the bias file.
        num_ag_cats: The number of agriculture categories to consider. Default is 3.
        flux_et: A boolean indicating whether the data is from flux ET dataset. If True additional aggregations are performed.

    Returns:
        None.
    """

    if flux_et:
        csv_files = aggregate_flux_et_data(
            et_files[0],
            output_dir=f'{os.path.dirname(et_files[0])}/Aggregated_Flux_ET/',
            season_col=season_col
        )
    else:
        csv_files = et_files + other_files
    crop_irr_type = ['All']
    common_station_dict = {}
    bias_ratio_dict = {}
    for ctype in crop_irr_type:
        common_station_dict[ctype] = set()
        bias_ratio_dict[ctype] = {}
    plot_dir = f'{plot_dir}Crop_Bias_Distributions/'
    os.makedirs(plot_dir, exist_ok=True)
    offset_dict = {
        'All': {
            'ETo': (0.1, 0.2),
            'Corrected ETo': (0.1, 0.2),
            'Uncorrected ETo': (0.1, 0.2),
            'ETr': (0.1, 0.2),
            'ea': (0.15, 0.01),
            'tmin': (2, 2),
            'tmax': (2, 2),
            'u2': (0, 0.8),
            'srad': (0.02, 0.1)
        }
    }
    var_name_dict = {
        'ETO': 'ETo',
        'CORRECTED ETO': 'Corrected ETo',
        'UNCORRECTED ETO': 'Uncorrected ETo',
        'ETR': 'ETr',
        'EA': 'ea',
        'TMIN': 'tmin',
        'TMAX': 'tmax',
        'U2': 'u2',
        'SRAD': 'srad'
    }
    season_name = season_col.split('_')[0]
    if season_name == 'growseason':
        season_name = 'growingseason'
    nlcd_ag_frac_col = 'NLCD_Ag Fraction'
    nlcd_ag_col = 'NLCD Agriculture'
    cdl_ag_frac_col = 'CDL_Ag Fraction'
    cdl_ag_col = 'CDL Agriculture'
    ag_cols = [nlcd_ag_col, cdl_ag_col]
    ag_col_names = ['NLCD Agriculture', 'CDL Agriculture']
    subset_cols = [
        station_col, 
        lat_col, 
        lon_col,
        start_year_col, 
        end_year_col,
        nlcd_ag_frac_col,
        cdl_ag_frac_col
    ] + ag_cols
    if not flux_et:
        subset_cols.append(climate_col)
    for csv_file in csv_files:
        csv_name = csv_file.split(os.sep)[-1].split('.')[0]
        if not flux_et:
            climate_dir = os.path.dirname(csv_file) + '/Climate/'
            bias_crop_name = f'{climate_dir}{csv_name}_climate'
        else:
            climate_dir = os.path.dirname(csv_file) + '/'
            bias_crop_name = f'{climate_dir}{csv_name}'
        vector_file = f'{bias_crop_name}.geojson'
        bias_all_crop_csv =  f'{bias_crop_name}_all_ag_{season_name}.csv'
        file_check = os.path.exists(bias_all_crop_csv)
        var_name = csv_file.split(os.sep)[-1].split('_')[0].upper()
        temp_flag = False
        if var_name in ['TMIN', 'TMAX']:
            var_name = f'{var_name_dict[var_name]} bias (°C)'
            temp_flag = True
        else:
            var_name = f'{var_name_dict[var_name]} bias ratio'
        if file_check:
            output_file = vector_file[: vector_file.rfind('.')] +  \
                '_irr_crop.geojson'
            bias_df = get_irr_crop_data(
                vector_file, 
                output_file, 
                gcloud_project=gcloud_project,
                start_year_col=start_year_col,
                end_year_col=end_year_col,
                lat_col=lat_col,
                lon_col=lon_col,
                station_col=station_col,
                num_ag_cats=num_ag_cats
            ).dropna()
            bias_val = bias_df[season_col]
            if temp_flag:
                bias_df[var_name] = -bias_val
            else:
                bias_df[var_name] = 1 / bias_val
            bias_all_crop_df = bias_df[subset_cols + [var_name]]            
            bias_all_crop_df.to_csv(bias_all_crop_csv, index=False)
        else:
            bias_all_crop_df = pd.read_csv(bias_all_crop_csv)
        bias_df_list = [
            bias_all_crop_df
        ]
        for bias_df, ctype in zip(bias_df_list, crop_irr_type):
            common_station_dict[ctype].update(bias_df[station_col].unique())
            bias_ratio_dict[ctype][var_name] = bias_df
    if flux_et:
        return  # No crop-based plots for flux ET data as only 30 cropland stations are available
    crop_color_dict = {}
    csv_dir = f'{plot_dir}Metrics_CSV/AgBias/'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_df = pd.DataFrame()
    for ctype in crop_irr_type:
        common_stations = common_station_dict[ctype]
        bias_dict = bias_ratio_dict[ctype]
        for hue, ag_col_name in zip(ag_cols, ag_col_names):
            plt.rcParams.update({'font.size': 16})
            fig, axes = plt.subplots(4, 2, figsize=(10, 15))
            axes = axes.flatten()
            for ax_idx, (var_name, bias_df) in enumerate(bias_dict.items()):            
                bias_df_common = bias_df[
                    bias_df[station_col].isin(common_stations)
                ]
                q1 = bias_df_common[var_name].quantile(0.25)
                q3 = bias_df_common[var_name].quantile(0.75)
                iqr = q3 - q1
                ll = q1 - 1.5 * iqr
                ul = q3 + 1.5 * iqr
                ax = axes[ax_idx]        
                hue_order =  sorted(bias_df_common[hue].unique())
                # swap last two elements for plotting by high, medium, and low
                if len(hue_order) > 1:
                    hue_order[-1], hue_order[-2] = hue_order[-2], hue_order[-1]
                if ctype == 'All':
                    for idx, crop in enumerate(hue_order):
                        crop_color_dict[crop] = sns.color_palette()[idx]
                crop_colors = crop_color_dict
                if 'Corrected' in var_name:
                    vname = 'Corr_ETo'
                elif 'Uncorrected' in var_name:
                    vname = 'Uncorr_ETo'
                else:   
                    vname = var_name.split(' ')[0]
                for h in hue_order:
                    sub_df = bias_df_common[bias_df_common[hue] == h]                        
                    sub_df_info = sub_df[[var_name]].describe().T \
                        .reset_index(drop=True)
                    sub_df_info['Season'] = season_name
                    sub_df_info['Ag Class'] = h
                    sub_df_info['Ag Data'] = ag_col_name
                    sub_df_info['Variable'] = vname
                    sub_df_info = sub_df_info.iloc[:, ::-1]
                    csv_df = pd.concat([csv_df, sub_df_info])        
                sns.boxplot(
                    data=bias_df_common, 
                    y=var_name, 
                    hue=hue,
                    hue_order=hue_order,
                    palette=crop_colors,
                    ax=ax
                )                
                offset1, offset2 = offset_dict[ctype][vname]
                ax.set_ylim(ll - offset1, ul + offset2) 
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda y, _: f'{y:.1f}')
                )
                ax.set_xticks([]) 
                handles, labels = ax.get_legend_handles_labels()
                ax.legend_.remove()      
            x0, y0, width, height = axes[-1].get_position().bounds
            fig.delaxes(axes[-1])
            legend_ax = fig.add_axes([x0, y0, width, height]) 
            legend = legend_ax.legend(
                handles, labels, 
                loc='upper center', 
                ncol=1,
                frameon=False
            )
            legend.set_title(hue)     
            legend_ax.axis('off')       
            plt.subplots_adjust(
                left=0.12, right=0.98, 
                top=0.98, bottom=0.1, 
                wspace=0.3, hspace=0.1
            )
            plot_file = (f'{plot_dir}{ctype}_{hue.split()[0]}_ag_bias_'
                            f'{season_name}_distributions.png')
            plt.savefig(plot_file, dpi=300)
            plt.close()
    climate_dict_names = {
        5: ('BSh + BSk', 'Hot and cold semi-arid (steppe)'),
        7: ('BWh + BWk', 'Hot and cold desert'),
        9: ('Cfa', 'Humid subtropical'),
        12: ('Csa + Csb', 'Hot- and warm-summer Mediterranean'),
        18: ('Dfa + Dfb', 'Hot- and warm-summer humid continental'),
    }
    csv_df.to_csv(
        f'{csv_dir}AgBiasDistributions_{season_name}.csv', 
        index=False
    )
    csv_df = pd.DataFrame()
    for ctype in crop_irr_type:
        common_stations = common_station_dict[ctype]
        bias_dict = bias_ratio_dict[ctype]
        for hue, ag_col_name in zip(ag_cols, ag_col_names):
            for var_name, bias_df in bias_dict.items():          
                bias_df_common = bias_df[
                    bias_df[station_col].isin(common_stations)
                ]
                min_var = bias_df_common[var_name].min()
                max_var = bias_df_common[var_name].max()
                plt.rcParams.update({'font.size': 16})
                fig, axes = plt.subplots(3, 2, figsize=(10, 15))
                axes = axes.flatten()
                for ax_idx, cl in enumerate(climate_dict_names.keys()):
                    bias_df_climate = bias_df_common[
                        bias_df_common[climate_col] == cl
                    ]
                    ax = axes[ax_idx]        
                    hue_order =  sorted(bias_df_climate[hue].unique())
                    # swap last two elements for plotting by high, medium, and low
                    if len(hue_order) > 1:
                        hue_order[-1], hue_order[-2] = hue_order[-2], hue_order[-1]
                    if ctype == 'All':
                        for idx, crop in enumerate(hue_order):
                            crop_color_dict[crop] = sns.color_palette()[idx]
                    crop_colors = crop_color_dict   
                    ax_title = (f'{climate_dict_names[cl][0]}: '
                                f'{climate_dict_names[cl][1]}')  
                    if 'Corrected' in var_name:
                        vname = 'Corr_ETo'
                    elif 'Uncorrected' in var_name:
                        vname = 'Uncorr_ETo'
                    else:
                        vname = var_name.split(' ')[0]
                    for h in hue_order:
                        sub_df = bias_df_climate[bias_df_climate[hue] == h]                        
                        sub_df_info = sub_df[[var_name]].describe().T
                        sub_df_info['Season'] = season_name
                        sub_df_info['Ag Class'] = h
                        sub_df_info['Ag Data'] = ag_col_name
                        sub_df_info['Climate'] = ax_title
                        sub_df_info['Variable'] = vname
                        sub_df_info = sub_df_info.iloc[:, ::-1]
                        csv_df = pd.concat([csv_df, sub_df_info])
                    sns.boxplot(
                        data=bias_df_climate, 
                        y=var_name, 
                        hue=hue,
                        hue_order=hue_order,
                        palette=crop_colors,
                        ax=ax
                    )
                    ax.set_ylim(min_var - 0.1, max_var + 0.1)
                    ax.yaxis.set_major_formatter(FuncFormatter(
                        lambda y, _: f'{y:.1f}')
                    )
                    ax.set_xticks([]) 
                    ax.set_title(ax_title, fontsize=12, pad=1.5)
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend_.remove()      
                x0, y0, width, height = axes[-1].get_position().bounds
                fig.delaxes(axes[-1])
                legend_ax = fig.add_axes([x0, y0, width, height]) 
                legend = legend_ax.legend(
                    handles, labels, 
                    loc='upper center', 
                    ncol=1,
                    frameon=False
                )
                legend.set_title(hue)     
                legend_ax.axis('off')       
                plt.subplots_adjust(
                    left=0.12, right=0.98, 
                    top=0.98, bottom=0.1, 
                    wspace=0.25, hspace=0.2
                )
                plot_file = (f'{plot_dir}{ctype}_{vname}_{hue.split()[0]}'
                                f'_ag_bias_{season_name}_distributions.png')
                plt.savefig(plot_file, dpi=300)
                plt.close()
    csv_df.to_csv(
        f'{csv_dir}ClimateAgBiasDistributions_{season_name}.csv', 
        index=False
    )


def gridmet_bias_comp_analysis(
        gridmet_csv: str,
        metadata_xlsx: str,
        station_files_daily_dir: str,
        station_files_monthly_dir: str,
        output_dir: str,
        site_col: str = 'SITE_ID',
        lulc_col: str = 'General classification',
        gridmet_corr_col: str = 'GRIDMET_REFERENCE_ET_BIAS_CORR',
        gridmet_uncorr_col: str = 'GRIDMET_REFERENCE_ET',   
        date_col: str = 'DATE',
        station_et_col: str = 'ASCE_ETo',
        station_lon_col: str = 'Longitude',
) -> None:
    """
    Perform the bias comparison analysis for the GridMET data.

    Args:
        gridmet_csv: A string representing the file path to the GridMET data.
        metadata_xlsx: A string representing the file path to the metadata
        Excel file.
        station_files_daily_dir: A string representing the directory containing
        the daily station files.
        station_files_monthly_dir: A string representing the directory containing
        the monthly station files.
        output_dir: A string representing the directory to save the plots.
        site_col: A string representing the column name for the site ID.
        lulc_col: A string representing the column name for the land cover
        classification.
        gridmet_corr_col: A string representing the column name for the corrected
        GridMET data.
        gridmet_uncorr_col: A string representing the column name for the uncorrected
        GridMET data.
        date_col: A string representing the column name for the date.
        station_et_col: A string representing the column name for the station ET.
        station_lon_col: A string representing the column name for the station longitude.

    """


    # Read the GridMET monthly data
    gridmet_df = pd.read_csv(gridmet_csv)

    print(f'Num stations in {gridmet_csv}:', len(gridmet_df[site_col].unique()))
    # Read the metadata Excel file
    metadata_df = pd.read_excel(metadata_xlsx, skiprows=1)
    metadata_df = metadata_df.rename(columns={'Site ID': site_col})
    metadata_df[lulc_col] = metadata_df[lulc_col].replace({'Wetland/Riparian': 'Wetlands'})

    # Join the GridMET data with the metadata
    gridmet_df = gridmet_df.merge(
        metadata_df, on=site_col
    )

    print('Num stations in merged site info and gridMET csv data:', len(gridmet_df[site_col].unique()))
    print('Num cropland stations in merged site info and gridMET csv data:', 
          len(gridmet_df[gridmet_df[lulc_col] == 'Croplands'][site_col].unique()))

    gridmet_df_west = gridmet_df[gridmet_df[station_lon_col] < -100].copy(deep=True)
    gridmet_df_east = gridmet_df[gridmet_df[station_lon_col] >= -100].copy(deep=True)
    gridmet_df_list = [gridmet_df, gridmet_df_west, gridmet_df_east]
    region_names = ['All', 'West', 'East']
    unit_dict = {
        'Daily': '(mm/day)',
        'Monthly': '(mm/month)'
    }
    stat_dir = f'{output_dir}GridMET_ETo_Stats/'
    os.makedirs(stat_dir, exist_ok=True)
    region_dict = {
        'East': 'Eastern U.S.',
        'West': 'Western U.S.',
        'All': 'CONUS'
    }
    for gridmet_df, region_name in zip(gridmet_df_list, region_names):
        print(f'\n\nWorking on the GridMET data for {region_name} sites...')
        plot_dir = f'{output_dir}GridMET_Plots/{region_name}/'
        os.makedirs(plot_dir, exist_ok=True)
        gridmet_col = gridmet_uncorr_col
        gridmet_corr_data = gridmet_df.drop(columns=gridmet_uncorr_col).rename(columns={
            gridmet_corr_col: gridmet_col
        })
        hue_col = 'gridMET ET$_o$'
        gridmet_corr_data[hue_col] = 'Corrected'
        gridmet_uncorr_data = gridmet_df.drop(columns=gridmet_corr_col)
        gridmet_uncorr_data[hue_col] = 'Uncorrected'
        gridmet_df = pd.concat([gridmet_corr_data, gridmet_uncorr_data])
    
        # Plot monthly bias boxplot distributions for corrected and uncorrected data across different land cover types    
        
        gridmet_df_monthly = gridmet_df[gridmet_df.TIMESTEP == 'monthly']
        plt.rcParams.update({'font.size': 16})
        _, axes = plt.subplots(3, 2, figsize=(10, 15))
        axes = axes.flatten()
        for ax_idx, lulc in enumerate(gridmet_df_monthly[lulc_col].unique()):
            ax = axes[ax_idx]
            gridmet_lulc_df = gridmet_df_monthly[gridmet_df_monthly[lulc_col] == lulc]
            sns.boxplot(
                data=gridmet_lulc_df, 
                y=gridmet_col, 
                ax=ax,
                hue=hue_col,
                palette='Set2',
                legend=ax_idx == 0
            )
            if ax_idx == 0:
                sns.move_legend(ax, loc='upper left')
            ax.set_ylabel('ET$_o$ (mm/month)')
            ax.set_title(lulc)
            ax.set_xlabel('')
            ax.set_ylim(0, 350)
        plt.subplots_adjust(
            left=0.1, right=0.98, 
            top=0.98, bottom=0.1, 
            wspace=0.3, hspace=0.3
        )
        plt.savefig(f'{plot_dir}GridMET_ETo_Bias_Boxplots_{region_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        new_hue_col = 'gridMET Corr_Uncorr'
        gridmet_df_monthly = gridmet_df_monthly.rename(columns={hue_col: new_hue_col}).reset_index(drop=True)
        if 'index' in gridmet_df_monthly.columns:
            gridmet_df_monthly = gridmet_df_monthly.drop(columns=['index'])

        num_stations = len(gridmet_df_monthly[site_col].unique())
        print(f'Total number of stations in merged site info and monthly gridMET csv data: {num_stations}')

        num_cropland_stations = len(gridmet_df_monthly[
            gridmet_df_monthly[lulc_col] == 'Croplands'
        ][site_col].unique())
        print(f'Total number of cropland stations in merged site info and monthly gridMET csv data: {num_cropland_stations}')
        station_files_dir = [station_files_daily_dir, station_files_monthly_dir]
        gridmet_df_daily = gridmet_df[gridmet_df.TIMESTEP == 'daily']
        print('Total number of stations in merged site info and daily gridMET csv data:', len(gridmet_df_daily[site_col].unique()))
        num_cropland_stations_daily = len(gridmet_df_daily[gridmet_df_daily[lulc_col] == 'Croplands'][site_col].unique())
        print('Total number of cropland stations in merged site info and daily gridMET csv data:', num_cropland_stations_daily, '\n\n')

        gridmet_time_df_list = [gridmet_df_daily, gridmet_df_monthly]
        time_list = unit_dict.keys()
        for gridmet_time_df, station_file_dir, time_val in zip(gridmet_time_df_list, station_files_dir, time_list):   
            time_station_df = pd.DataFrame()
            for station_csv in glob(f'{station_file_dir}*.csv'):
                station_df = pd.read_csv(station_csv)
                if station_et_col in station_df.columns:
                    station_name = station_csv.split('/')[-1].split(f'_{time_val.lower()}')[0]
                    station_df[site_col] = station_name
                    time_station_df = pd.concat([time_station_df, station_df])

            time_station_df = time_station_df.rename(columns={'date': date_col})
            if time_val == 'Monthly':
                # set date to Year-Month
                time_station_df[date_col] = pd.to_datetime(time_station_df[date_col]).dt.to_period('M')
                gridmet_time_df[date_col] = pd.to_datetime(gridmet_time_df[date_col]).dt.to_period('M')
            time_station_df = time_station_df.merge(
                gridmet_time_df, on=[site_col, date_col]
            )
            time_station_df = time_station_df.dropna(subset=[gridmet_col, station_et_col]).drop_duplicates()
            time_station_df = time_station_df.rename(columns={hue_col: new_hue_col})
            time_station_csv = f'{plot_dir}GridMET_{time_val}_{region_name}_Station_Data.csv'
            time_station_df.to_csv(time_station_csv, index=False)
            unique_stations = pd.DataFrame({site_col: time_station_df[site_col].unique()})
            unique_stations.to_csv(f'{plot_dir}Unique_Stations_GridMET_{time_val}_{region_name}.csv', index=False)
            if time_val == 'Daily':
                time_station_zip = f'{plot_dir}GridMET_{time_val}_{region_name}_Station_Data.zip'
                with zipfile.ZipFile(time_station_zip, 'w', zipfile.ZIP_LZMA) as zipf:
                    zipf.write(time_station_csv, os.path.basename(time_station_csv))

            num_stations = len(time_station_df[site_col].unique())
            print(f'Total number of stations in the merged {time_val} gridMET and station data: {num_stations}')

            for lulc in sorted(time_station_df[lulc_col].unique()):
                lulc_df = time_station_df[time_station_df[lulc_col] == lulc]
                num_lulc_stations = len(lulc_df[site_col].unique())
                print(f'Number of {lulc} stations in the merged {time_val} gridMET and station data: {num_lulc_stations}')
                unique_stations = lulc_df[metadata_df.columns].drop_duplicates()
                # remove special characters from the land cover type
                lulc_name = lulc.replace('/', '_')
                unique_stations.to_csv(f'{plot_dir}Unique_{lulc_name}_Stations_{time_val}_{region_name}.csv', index=False)
            print('\n')

            # calculate error metrics across all stations for the uncorrected data
            uncorr_data = time_station_df[time_station_df[new_hue_col] == 'Uncorrected']
            r2_uncorr_all = np.corrcoef(uncorr_data[station_et_col], uncorr_data[gridmet_col])[0, 1] ** 2
            # rmse_uncorr_all = root_mean_squared_error(uncorr_data[station_et_col], uncorr_data[gridmet_col])
            mae_uncorr_all = mean_absolute_error(uncorr_data[station_et_col], uncorr_data[gridmet_col])
            mbe_uncorr_all = np.mean(uncorr_data[station_et_col] - uncorr_data[gridmet_col])

            # calculate error metrics across all stations for the corrected data
            corr_data = time_station_df[time_station_df[new_hue_col] == 'Corrected']
            r2_corr_all = np.corrcoef(corr_data[station_et_col], corr_data[gridmet_col])[0, 1] ** 2
            # rmse_corr_all = root_mean_squared_error(corr_data[station_et_col], corr_data[gridmet_col])
            mae_corr_all = mean_absolute_error(corr_data[station_et_col], corr_data[gridmet_col])
            mbe_corr_all = np.mean(corr_data[station_et_col] - corr_data[gridmet_col])

            # show subplots for the uncorrected data and corrected data
            plt.rcParams.update({'font.size': 16})
            _, axes = plt.subplots(1, 2, figsize=(20, 10))
            hue_order = sorted(time_station_df[lulc_col].unique())
            for ax_idx, data in enumerate([uncorr_data, corr_data]): 
                # show scatter plot of uncorrected data across all stations
                ax = axes[ax_idx]
                sns.scatterplot(
                    data=data, 
                    x=station_et_col, 
                    y=gridmet_col,
                    hue=lulc_col,
                    style=lulc_col,
                    hue_order=hue_order,
                    style_order=hue_order,
                    markers=['o', 's', 'D', 'P', 'v', '^'],
                    palette='deep',
                    alpha=0.2,
                    ax=ax
                )
                ax.legend(loc='upper left', title='LULC')
                # Get the legend object
                legend = ax.get_legend()
                # Iterate through the legend handles and set the alpha
                for handle in legend.legend_handles:
                    handle.set_alpha(1)  # Set alpha to 0.5 (adjust as needed)
                max_x = max(uncorr_data[station_et_col].max(), uncorr_data[gridmet_col].max())
                max_y = max(corr_data[station_et_col].max(), corr_data[gridmet_col].max())
                max_x = max(max_x, max_y) + 5
                ax.plot([0, max_x], [0, max_x], color='k', linestyle='--')
                ax.set_xlabel(f'Station ETo {unit_dict[time_val]}')
                ax.set_ylabel(f'gridMET ETo {unit_dict[time_val]}')
                ax.set_xlim(-0.5, max_x)
                ax.set_ylim(-0.5, max_x)
                if ax_idx == 0:
                    ax.set_title('Uncorrected')
                    ax.text(
                        max_x * 0.3, max_y * 0.1,
                        f'$r^2_u = {r2_uncorr_all:.2f}, MBE_u = {mbe_uncorr_all:.2f}, MAE_u = {mae_uncorr_all:.2f}$', 
                        fontsize=16,
                        color='red',
                        bbox=dict(facecolor=(1, 0, 0, 0.05), edgecolor='none')

                    )
                else:
                    ax.set_title('Corrected')
                    ax.text(
                        max_x * 0.3, max_y * 0.1,
                        f'$r^2_c = {r2_corr_all:.2f}, MBE_c = {mbe_corr_all:.2f}, MAE_c = {mae_corr_all:.2f}$',
                        fontsize=16,
                        color='blue',
                        bbox=dict(facecolor=(0, 0, 1, 0.05), edgecolor='none')
                    )
            plt.savefig(
                f'{plot_dir}GridMET_ETo_{time_val}_{region_name}_Scatter_Corr_Uncorr.png', 
                dpi=300, bbox_inches='tight'
            )
            plt.close()            

            # plot the scatterplots for each land cover type
            for lulc in time_station_df[lulc_col].unique():
                # plot subplots for the uncorrected data and corrected data

                uncorr_lulc_data = uncorr_data[uncorr_data[lulc_col] == lulc]
                corr_lulc_data = corr_data[corr_data[lulc_col] == lulc]

                # calculate error metrics for the uncorrected data
                r2_uncorr = np.corrcoef(uncorr_lulc_data[station_et_col], uncorr_lulc_data[gridmet_col])[0, 1] ** 2
                # rmse_uncorr = root_mean_squared_error(uncorr_lulc_data[station_et_col], uncorr_lulc_data[gridmet_col])
                mae_uncorr = mean_absolute_error(uncorr_lulc_data[station_et_col], uncorr_lulc_data[gridmet_col])
                mbe_uncorr = np.mean(uncorr_lulc_data[station_et_col] - uncorr_lulc_data[gridmet_col])

                # calculate error metrics for the corrected data
                r2_corr = np.corrcoef(corr_lulc_data[station_et_col], corr_lulc_data[gridmet_col])[0, 1] ** 2
                # rmse_corr = root_mean_squared_error(corr_lulc_data[station_et_col], corr_lulc_data[gridmet_col])
                mae_corr = mean_absolute_error(corr_lulc_data[station_et_col], corr_lulc_data[gridmet_col])
                mbe_corr = np.mean(corr_lulc_data[station_et_col] - corr_lulc_data[gridmet_col])

                # show subplots for the uncorrected data and corrected data
                plt.rcParams.update({'font.size': 16})
                _, axes = plt.subplots(1, 2, figsize=(20, 10))

                for ax_idx, data in enumerate([uncorr_lulc_data, corr_lulc_data]):
                    ax = axes[ax_idx]
                    sns.scatterplot(
                        data=data, 
                        x=station_et_col, 
                        y=gridmet_col,
                        color='red' if ax_idx == 0 else 'blue',
                        marker='o',
                        ax=ax,
                        alpha=0.2
                    )
                    ax.plot([0, max_x], [0, max_x], color='k', linestyle='--')
                    ax.set_xlabel(f'Station ETo {unit_dict[time_val]}')
                    ax.set_ylabel(f'gridMET ETo {unit_dict[time_val]}')
                    ax.set_xlim(-0.5, max_x)
                    ax.set_ylim(-0.5, max_x)
                    if ax_idx == 0:
                        ax.set_title(f'{lulc} Uncorrected: {region_dict[region_name]}')
                        ax.text(
                            max_x * 0.3, max_y * 0.1, 
                            f'$r^2_u = {r2_uncorr:.2f}, MBE_u = {mbe_uncorr:.2f}, MAE_u = {mae_uncorr:.2f}$', 
                            fontsize=16,
                            color='red',
                            bbox=dict(facecolor=(1, 0, 0, 0.05), edgecolor='none')
                        )
                    else:
                        ax.set_title(f'{lulc} Corrected: {region_dict[region_name]}')
                        ax.text(
                            max_x * 0.3, max_y * 0.1, 
                            f'$r^2_c = {r2_corr:.2f}, MBE_c = {mbe_corr:.2f}, MAE_c = {mae_corr:.2f}$', 
                            fontsize=16,
                            color='blue',
                            bbox=dict(facecolor=(0, 0, 1, 0.05), edgecolor='none')
                        )
                # remove special characters from the land cover type
                lulc_name = lulc.replace('/', '_')
                plt.savefig(
                    f'{plot_dir}GridMET_ETo_{time_val}_{region_name}_Scatter_{lulc_name}.png', 
                    dpi=300, bbox_inches='tight'
                )
                plt.close()
            # here we'll make monthly subplots for the same
            if time_val == 'Monthly':
                time_station_df_monthly = time_station_df.copy(deep=True)
                time_station_df_monthly = time_station_df_monthly.rename(columns={new_hue_col: 'gridMET ETo'})
                new_hue_col = 'gridMET ETo'
                time_station_df_monthly['Month'] = time_station_df_monthly[date_col].dt.month
                max_x = max(
                    time_station_df_monthly[station_et_col].max(), 
                    time_station_df_monthly[gridmet_col].max()
                ) + 5
                lulc_vals = ['All'] + time_station_df_monthly[lulc_col].unique().tolist()
                for lulc_val in lulc_vals:
                    if lulc_val == 'All':
                            data_df = time_station_df_monthly.copy(deep=True)
                    else:
                        data_df = time_station_df_monthly[time_station_df_monthly[lulc_col] == lulc_val]
                    months = sorted(data_df['Month'].unique())
                    _, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20), sharey=True)
                    for ax, month in zip(axes.flatten(), months):
                        month_data = data_df[data_df['Month'] == month]
                        month_corr = month_data[month_data[new_hue_col] == 'Corrected']
                        month_uncorr = month_data[month_data[new_hue_col] == 'Uncorrected']
                        r2_corr = np.corrcoef(month_corr[station_et_col], month_corr[gridmet_col])[0, 1] ** 2
                        r2_uncorr = np.corrcoef(month_uncorr[station_et_col], month_uncorr[gridmet_col])[0, 1] ** 2
                        mbe_corr = np.mean(month_corr[station_et_col] - month_corr[gridmet_col])
                        mbe_uncorr = np.mean(month_uncorr[station_et_col] - month_uncorr[gridmet_col])
                        mae_corr = np.mean(np.abs(month_corr[station_et_col] - month_corr[gridmet_col]))
                        mae_uncorr = np.mean(np.abs(month_uncorr[station_et_col] - month_uncorr[gridmet_col]))
                        sns.scatterplot(
                            data=month_data,
                            x=station_et_col,
                            y=gridmet_col,
                            hue=new_hue_col,
                            ax=ax,
                            alpha=0.2,
                            palette={'Uncorrected': 'red', 'Corrected': 'blue'},
                            legend=True if month == 1 else False,
                        )
                        text_posx = 5
                        text_posy_uncorr = max_x * 0.9
                        text_posy_corr = max_x * 0.8
                        if month == 1:
                            # Get the legend object
                            legend = ax.get_legend()
                            # Iterate through the legend handles and set the alpha
                            for handle in legend.legend_handles:
                                handle.set_alpha(1)
                        elif month in [6, 7, 8]: 
                            text_posy_uncorr = max_x * 0.18
                            text_posy_corr = max_x * 0.08
                        ax.set_title(calendar.month_name[month])
                        ax.set_xlabel('Station ETo (mm/month)')
                        ax.set_ylabel('gridMET ETo (mm/month)')
                        ax.set_xlim(-0.5, max_x)
                        ax.set_ylim(-0.5, max_x)
                        ax.plot([0, max_x], [0, max_x], color='k', linestyle='--')
                        ax.text(
                            text_posx, text_posy_uncorr,
                            f'$r^2_u = {r2_uncorr:.2f}, MBE_u = {mbe_uncorr:.2f}, MAE_u = {mae_uncorr:.2f}$',
                            fontsize=12,
                            color='red',
                            bbox=dict(facecolor=(1, 0, 0, 0.05), edgecolor='none')
                        )
                        ax.text(
                            text_posx, text_posy_corr,
                            f'$r^2_c = {r2_corr:.2f}, MBE_c = {mbe_corr:.2f}, MAE_c = {mae_corr:.2f}$',
                            fontsize=12,
                            color='blue',
                            bbox=dict(facecolor=(0, 0, 1, 0.05), edgecolor='none')
                        )
                    plt.tight_layout()
                    plt.savefig(
                        f'{plot_dir}GridMET_ETo_{region_name}_{lulc_val}_Scatter_Monthly.png',
                        dpi=300, bbox_inches='tight'
                    )
                    plt.close()
