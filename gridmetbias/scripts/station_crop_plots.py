# Plot station variables grouped by CDL crop type
# author: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

import pandas as pd
import ee
import os
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from dask import delayed, compute
from dask.distributed import Client, LocalCluster
from http.client import RemoteDisconnected


def fix_cdl_classes(
    cdl_df: pd.DataFrame,
    cdl_col: str = 'cdl_crop'
) -> pd.DataFrame:
    """
    Fix the CDL classes in the CDL DataFrame.

    Args:
        cdl_df: The DataFrame containing the CDL data.

    Returns:
        The DataFrame with the fixed CDL classes.
    """
    cropland_classes = {
        1: "Corn",
        2: "Cotton",
        5: "Soybeans",
        22: "Wheat",
        23: "Wheat",
        24: "Wheat",
        26: "Soybeans",
        36: "Alfalfa",
        225: "Corn",
        226: "Corn",
        230: "Wheat",
        232: "Cotton",
        234: "Wheat",
        236: "Wheat",
        237: "Corn",
        238: "Cotton",
        239: "Cotton",
        240: "Soybeans",
        241: "Corn",
        254: "Soybeans"
    }

    crop_type_col = 'Crop Type'
    cdl_df[crop_type_col] = cdl_df[cdl_col].apply(
        lambda x: cropland_classes.get(x, 'Other')
    )
    cdl_df = cdl_df.drop(columns=[cdl_col])
    return cdl_df


def generate_chunks(input_list: list, num_chunks: int):
    """
    Partition a list into equally sized chunks.

    Args:
        input_list (List): List of objects.
        num_chunks (int): Number of chunks.

    Returns:
        Generator object
    """

    num_chunks = max(1, num_chunks)
    for idx in range(0, len(input_list), num_chunks):
        yield input_list[idx: idx + num_chunks]


def download_station_cdl(
    year_list: list[int],
    output_dir: str,
    station_id: str,
    lon_val: float,
    lat_val: float,
    cdl_target_scale: int = 100,
    station_col: str = "Station",
    year_col: str = "year",
    crop_col: str = "cdl_crop",
    gcloud_project: str = "ee-grid-obs-comp"
) -> None:
    """
    Download CDL data for a specific station and year range.

    Args:
        year_list (list[int]): The list of years to download data for.
        output_dir (str): The directory to save the output CSV files.
        station_id (str): The ID of the station.
        lon_val (float): The longitude of the station.
        lat_val (float): The latitude of the station.
        cdl_target_scale (int, optional): The target scale for the CDL data. Defaults to 100.
        station_col (str, optional): The column name for the station ID. Defaults to "Station".
        year_col (str, optional): The column name for the year. Defaults to "year". 
        This will be added to the output CSV file.
        crop_col (str, optional): The column name for the crop type. Defaults to "cdl_crop". 
        This will be added to the output CSV file.
        gcloud_project (str, optional): The Google Cloud project ID. Defaults to "ee-grid-obs-comp".

    Returns:
        None
    """

    station_csv = f"{output_dir}/{station_id}_cdl_data.csv"
    if os.path.exists(station_csv) and pd.read_csv(station_csv).shape[0] > 0:
        return
    # Initialize Earth Engine
    ee.Initialize(
        project=gcloud_project,
        opt_url='https://earthengine-highvolume.googleapis.com'
    )
    # Get the station's location
    location = ee.Geometry.Point(lon_val, lat_val)
    cdl_ic = ee.ImageCollection("USDA/NASS/CDL")
    max_pixels = np.ceil((cdl_target_scale / 30) ** 2)
    cdl_proj = cdl_ic.first().projection()
    cdl_list = []
    for year in year_list:
        cdl_start_year = f'{year}-01-01'
        cdl_end_year = f'{year}-12-31'
        retry_download = True
        while retry_download:
            try:
                cdl_year = cdl_ic.filterDate(cdl_start_year, cdl_end_year) \
                            .select('cropland') \
                            .first() \
                            .setDefaultProjection(
                                crs=cdl_proj, 
                                scale=30
                            )
                cdl_year_target = cdl_year.reduceResolution(
                        reducer=ee.Reducer.mode(maxRaw=max_pixels),
                        bestEffort=True,
                        maxPixels=max_pixels
                    ).reproject(cdl_proj, scale=cdl_target_scale)
                cdl_year_val = cdl_year_target.sample(
                        region=location,
                        scale=cdl_target_scale,
                        numPixels=1
                    ).first().getInfo()
                if cdl_year_val is not None:
                    cdl_year_val = int(cdl_year_val['properties']['cropland'])
                else:
                    cdl_year_val = np.nan
                    print('Missing CDL for station:', station_id, 'year:', year)
                cdl_list.append(cdl_year_val)
                retry_download = False
            except (
                ee.EEException, requests.exceptions.RequestException,
                requests.exceptions.ConnectionError, RemoteDisconnected
            ) as e:
                print('Error:', e, '.Retrying station,', station_id, '...')
                retry_download = True
                time.sleep(5)

    station_cdl_df = pd.DataFrame({
        station_col: station_id,
        year_col: year_list,
        crop_col: cdl_list
    })
    station_cdl_df.to_csv(station_csv, index=False)

def create_station_cdl_csv(
        station_df: pd.DataFrame,
        output_dir: str,
        station_xls_dir: str,
        station_id_col: str = "Station",
        station_id_col_xls: str = "Station ID",
        lon_col: str = "Longitude",
        lat_col: str = "Latitude",
        date_col: str = "Date",
        gcloud_project: str = "ee-grid-obs-comp",
        num_workers: int = 40,
        worker_memory: str = "1G"
) -> pd.DataFrame:
    """
    Create a CSV file with CDL data for each station for each year from 2008 to 2020.

    Args:
        station_df (pd.DataFrame): DataFrame containing station information.
        output_dir (str): Directory to save output files.
        station_xls_dir (str): Directory containing station Excel files.
        station_id_col (str): Name of the column containing station IDs.
        station_id_col_xls (str): Name of the column containing station IDs in the Excel files.
        lon_col (str): Name of the column containing longitude values.
        lat_col (str): Name of the column containing latitude values.
        gcloud_project (str): The Google Cloud project ID. Defaults to "ee-grid-obs-comp".
        num_workers (int): The number of worker processes to use. Defaults to 40.
        worker_memory (str): The amount of memory to allocate for each worker. Defaults to "1GB".

    Returns:
        pd.DataFrame: DataFrame containing CDL data for each station.
    """
      

    final_csv = f"{output_dir}/station_cdl_data.csv"
    if not os.path.exists(final_csv):
        # Get CDL data for each station
        cdl_target_scale = 1000
        year_list = list(range(2008, 2021))
        station_ids = station_df[station_id_col].tolist()
        lon_vals = station_df[lon_col].tolist()
        lat_vals = station_df[lat_col].tolist()
        year_col = 'year'
        crop_col = 'cdl_crop'
        station_info = list(zip(station_ids, lon_vals, lat_vals))
        station_chunks = generate_chunks(station_info, num_workers)
        itr = 1
        temp_dir = f"{output_dir}/temp/"
        os.makedirs(temp_dir, exist_ok=True)
        num_chunks = int(np.ceil(len(station_info) / num_workers))
        dask_cluster = LocalCluster(n_workers=num_workers, memory_limit=worker_memory)
        dask_cluster.scale(num_workers)
        dask_client = Client(dask_cluster)
        dask_client.wait_for_workers(1)
        print(f'Using {num_workers} local workers...')
        for station_chunk in station_chunks:
            print(f'Working on station chunk {itr} / {num_chunks} ...')
            compute(
                delayed(download_station_cdl)(
                    year_list,
                    temp_dir,
                    station_vals[0],
                    station_vals[1],
                    station_vals[2],
                    cdl_target_scale,
                    station_id_col,
                    year_col,
                    crop_col,
                    gcloud_project
                )
                for station_vals in station_chunk
            )
            itr += 1
        dask_client.shutdown()
        station_cdl_df = pd.DataFrame()
        print('Merging station files...')
        for temp_file in glob(f"{temp_dir}*.csv"):
            df = pd.read_csv(temp_file)
            station_cdl_df = pd.concat([station_cdl_df, df])
        station_cdl_df = station_cdl_df.dropna()
        station_cdl_df = station_cdl_df.rename(columns={station_id_col: station_id_col_xls})
        station_cdl_df = fix_cdl_classes(station_cdl_df, crop_col)
        station_xls_df = pd.concat([pd.read_excel(f) for f in glob(f"{station_xls_dir}*.xlsx")])
        station_xls_df[date_col] = pd.to_datetime(station_xls_df[date_col])
        station_xls_df[year_col] = station_xls_df[date_col].dt.year
        station_cdl_df = station_cdl_df.merge(station_xls_df, on=[station_id_col_xls, year_col])
        station_cdl_df.to_csv(final_csv, index=False)
    else:
        station_cdl_df = pd.read_csv(final_csv)
    return station_cdl_df


def make_station_cdl_plots(
    station_cdl_df: pd.DataFrame, 
    output_dir: str,
    crop_col: str = 'Crop Type'
) -> None:
    """
    Create plots for the CDL data of each station with both boxplots and KDE plots.

    Args:
        station_cdl_df (pd.DataFrame): DataFrame containing CDL data for each station.
        output_dir (str): Directory to save the plots.
        crop_col (str): Column name for crop types.
    """
    os.makedirs(output_dir, exist_ok=True)
    station_vars = station_cdl_df.columns[7:]
    crops = sorted(station_cdl_df[crop_col].unique())

    # Use Paired color palette as requested
    colors = plt.cm.Paired(np.linspace(0, 1, len(crops)))
    crop_colors = dict(zip(crops, colors))
    
    # Special handling for ETo and ETr - put them together
    et_vars = [var for var in station_vars if 'ETo' in var or 'ETr' in var]
    other_vars = [var for var in station_vars if var not in et_vars]
    
    # Group other variables into pairs
    var_groups = []
    if et_vars:
        var_groups.append(et_vars[:2])  # ETo and ETr together
    
    # Add pairs of other variables
    for i in range(0, len(other_vars), 2):
        var_groups.append(other_vars[i:i+2])
    
    for group_idx, var_group in enumerate(var_groups):
        # Create subplots with custom gridspec for better control
        fig = plt.figure(figsize=(18, 18))
        
        # Special handling for ETo and ETr - calculate shared y-axis limits
        shared_y_limits = None
        if len(var_group) == 2 and all('ET' in var for var in var_group):
            # Calculate combined y-limits for ETo and ETr
            all_et_data = []
            for var in var_group:
                var_data = station_cdl_df[var].dropna()
                if len(var_data) > 0:
                    all_et_data.extend(var_data.values)
            
            if all_et_data:
                y_min = min(all_et_data)
                y_max = max(all_et_data)
                y_range = y_max - y_min
                y_padding = y_range * 0.05  # 5% padding
                shared_y_limits = (y_min - y_padding, y_max + y_padding)
        
        for var_idx, var in enumerate(var_group):
            # Create custom gridspec - reduced colspan for 2 figures per plot
            # Left side for KDE (2 columns), right side for boxplot (2 columns) - no gap
            ax_kde = plt.subplot2grid((1, len(var_group)*5), (0, var_idx*5), colspan=2)
            ax_box = plt.subplot2grid((1, len(var_group)*5), (0, var_idx*5+2), colspan=2)
            
            # Plot overall KDE on left axis with histogram bars
            overall_data = station_cdl_df[var].dropna()
            if len(overall_data) > 1:
                # Plot histogram bars in light gray with counts (not density)
                n, bins, patches = ax_kde.hist(
                    overall_data,
                    bins=30,
                    orientation='horizontal',  # Horizontal orientation to match vertical KDE
                    alpha=0.4,
                    color='#D3D3D3',  # Light gray (light shade of black)
                    density=False,  # Changed to False to show counts instead of density
                    edgecolor='#808080',  # Medium gray for edges
                    linewidth=0.5
                )
                
                # Plot KDE line in dark color over histogram
                # Create a second y-axis for KDE to match the count scale
                ax_kde_twin = ax_kde.twiny()
                sns.kdeplot(
                    y=overall_data,
                    ax=ax_kde_twin,
                    color="#000406",  # Dark color for KDE line
                    alpha=0.9,
                    linewidth=4,
                    fill=False,  # Don't fill, just show the line
                    label=f'All sites (n={len(overall_data):,})'
                )
                
                # Hide the twin axis ticks and labels
                ax_kde_twin.set_xlabel('')
                ax_kde_twin.tick_params(top=False, labeltop=False)
                ax_kde_twin.spines['top'].set_visible(False)   
                ax_kde_twin.spines['right'].set_visible(False)           
                
            
            # Create very narrow boxplot on right axis
            sns.boxplot(
                data=station_cdl_df, 
                y=var, 
                hue=crop_col,
                ax=ax_box,
                palette=crop_colors,
                hue_order=crops,
                width=0.5,  # Make boxplots much narrower
                dodge=True
            )
            
            # Set variable label with bigger font
            var_label = f"({chr(97 + group_idx*2 + var_idx)})"
            ax_box.text(-0.1, 1.05, var_label, transform=ax_box.transAxes, 
                       fontsize=20, fontweight='bold', ha='left')
            
            # Style KDE axis (left) - only keep left and bottom spines
            if var in ['TMax (C)', 'TMin (C)', 'TAvg (C)']:
                var_unit_label = var.replace(' (C)', ' (°C)')
            else:
                var_unit_label = var
            ax_kde.set_ylabel(f'{var_unit_label}', fontsize=18)
            # Set x-axis label for counts
            ax_kde.set_xlabel('site-days', fontsize=18)
            ax_kde.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))  # Integer ticks only
            ax_kde.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))  # Comma formatting
            ax_kde.tick_params(axis='both', labelsize=14)
            
            # Remove top and right spines for KDE
            ax_kde.spines['top'].set_visible(False)
            ax_kde.spines['right'].set_visible(False)
            ax_kde.spines['left'].set_linewidth(2)
            ax_kde.spines['bottom'].set_linewidth(2)
            ax_kde.grid(False)
            
            # Style boxplot axis (right) - remove all spines
            ax_box.set_ylabel('')  # Remove y-label since KDE axis has it
            ax_box.set_xlabel('')
            ax_box.tick_params(axis='both', labelsize=14)
            ax_box.set_xticklabels([])  # Remove x-axis labels
            
            # Remove all spines for boxplot
            for spine in ax_box.spines.values():
                spine.set_visible(False)
            ax_box.grid(False)
            ax_box.tick_params(left=False, bottom=False)  # Remove tick marks
            
            # Apply shared y-axis limits for ETo and ETr
            if shared_y_limits is not None:
                ax_kde.set_ylim(shared_y_limits)
                ax_box.set_ylim(shared_y_limits)
                if 'ax_kde_twin' in locals():
                    ax_kde_twin.set_ylim(shared_y_limits)
            else:
                # Match y-axis limits between KDE and boxplot for non-ET variables
                y_min = min(ax_kde.get_ylim()[0], ax_box.get_ylim()[0])
                y_max = max(ax_kde.get_ylim()[1], ax_box.get_ylim()[1])
                ax_kde.set_ylim(y_min, y_max)
                ax_box.set_ylim(y_min, y_max)
            
            # Remove y-axis labels and ticks from boxplot
            ax_box.set_yticklabels([])
            ax_box.tick_params(left=False)
            
            # Remove individual legends from each subplot
            if ax_box.get_legend():
                ax_box.get_legend().remove()
            if ax_kde.get_legend():
                ax_kde.get_legend().remove()
        
        # Add single legend for crops at the top of the figure with bigger font
        handles = []
        labels = []
        
        # Add overall sites legend first
        handles.append(plt.Rectangle((0,0),1,1, facecolor='#D3D3D3', alpha=0.4))
        n_total = len(station_cdl_df)
        labels.append(f'All sites ({n_total:,})')
        
        # Get crop colors for legend
        for crop in crops:
            n_crop = len(station_cdl_df[station_cdl_df[crop_col] == crop])
            handles.append(plt.Rectangle((0,0),1,1, facecolor=crop_colors[crop], alpha=0.8))
            labels.append(f'{crop} ({n_crop:,})')
        # Create legend at the top with bigger font
        var1_name = var_group[0].split('(')[0].strip() if len(var_group) > 0 else "variable"
        var2_name = var_group[1].split('(')[0].strip() if len(var_group) > 1 else "variable"
        legend_title = f"Site-days of {var1_name} and {var2_name}"

        # Create legend at the top with bigger font
        fig.legend(
            handles, labels, 
            title=legend_title,
            loc='upper center', 
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(len(crops)+1, 4), 
            fontsize=16,
            title_fontsize=16,
            frameon=False # Remove legend border
        )  
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.85, wspace=0.01)  # Make room for legend and reduce spacing

        # Save plot
        plt.savefig(f"{output_dir}/station_variables_group_{group_idx+1}.png", 
                   dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        
    print(f"Created {len(var_groups)} plots with 2 variables each in {output_dir}")

if __name__ == "__main__":
    station_metadata = pd.read_csv("../../Data/supporting_files/metadata_for_publication.csv")
    station_xls_dir = "../../Data/supporting_files/standardized_data/"
    output_directory = "../../Data/supporting_files/Station_CDL/"
    station_cdl_df = create_station_cdl_csv(
        station_metadata, 
        output_directory, 
        station_xls_dir
    )
    plot_dir = "../../Plots/Station_CDL/"
    make_station_cdl_plots(station_cdl_df, plot_dir)
