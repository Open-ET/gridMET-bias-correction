"""
This script performs a site analysis for gridMET reference ET data, generating scatter plots
of gridMET ET against flux tower ET for different models and versions.
It calculates error metrics such as R2, MAE, MBE, and Pearson correlation,
and saves the results in a CSV file.

Author: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import seaborn as sns
import site_analysis_openet as openet
import matplotlib.pyplot as plt
from shutil import copytree
from pathlib import Path
from sklearn.metrics import mean_absolute_error


def flux_gridmet_plots(
        df_site: pd.DataFrame,
        site_id: str,
        output_dir: str,
        gridmet_col: str = 'GRIDMET_REFERENCE_ET',
        flux_col: str = 'ASCE_ETo',
) -> pd.DataFrame:
    """
    Generate scatter plots of gridMET reference ET against flux tower reference ET.
    This function calculates error metrics such as R2, RMSE, MAE, and MBE, and saves the results in a CSV file.

    Args:
        df_site (pd.DataFrame): DataFrame containing the site data.
        site_id (str): Site identifier.
        output_dir (str): Directory to save the output plots and metrics. 

    Returns:
        pd.DataFrame: DataFrame containing the calculated metrics for each model.
    """

    metrics_df = pd.DataFrame()
    # Define the output file name based on version and type
    output_file = Path(output_dir) / f"{site_id}_scatter.png"
    dt_type = 'day' if 'daily' in str(output_dir) else 'month'

    # Plot corrected and uncorrected gridMET ET against flux tower ET as subplots
    _, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    xlim_max = 250 if dt_type == 'month' else 15  # Adjust max limit based on dt_type
    ylim_max = 250 if dt_type == 'month' else 15  # Adjust max limit based on dt_type
    for corr_uncorr in ['Uncorrected', 'Corrected']:
        df_corr = df_site[df_site['gridMET Corr_Uncorr'] == corr_uncorr]
        if df_corr.empty:
            continue
        
        ax = axes[0] if corr_uncorr == 'Corrected' else axes[1]
        plt.rcParams.update({'font.size': 14})
        sns.scatterplot(
            data=df_corr,
            x=flux_col,
            y=gridmet_col,
            ax=ax,
            s=100,
            alpha=0.7
        )        
        ax.grid()
        # plot 1:1 line
        max_val = max(df_corr[flux_col].max(), df_corr[gridmet_col].max())
        min_val = min(df_corr[flux_col].min(), df_corr[gridmet_col].min())
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='1:1 Line')
        ax.legend()
        ax.set_xlim(df_corr[flux_col].min() * 0.8, xlim_max)  # Adjust x-limits to allow for better visibility
        ax.set_ylim(df_corr[gridmet_col].min() * 0.8, ylim_max)  # Adjust y-limits to allow for better visibility
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_title(f"{corr_uncorr}", fontsize=16)

        # Calculate metrics
        r2 = np.corrcoef(df_corr[flux_col], df_corr[gridmet_col])[0, 1] ** 2
        mae = mean_absolute_error(df_corr[flux_col], df_corr[gridmet_col])
        mbe = np.mean(df_corr[flux_col] - df_corr[gridmet_col])

        # Add metrics to plot
        metrics_text = (
            f"r$^2$: {r2:.2f}\n"
            f"MAE: {mae:.2f} mm/{dt_type}\n"
            f"MBE: {mbe:.2f} mm/{dt_type}\n"
        )
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
        ax.set_xlabel(f'Station ET$_o$ (mm/{dt_type})')
        ax.set_ylabel(f'gridMET ET$_o$ (mm/{dt_type})')
        
        # Add metrics to DataFrame
        metrics_df = pd.concat([
            metrics_df,
            pd.DataFrame({
                'SITE_ID': [site_id],
                'Latitude': [df_corr['Latitude'].iloc[0]],
                'Longitude': [df_corr['Longitude'].iloc[0]],
                'corr_uncorr_type': [corr_uncorr],
                'r2': [r2],
                'MAE': [mae],
                'MBE': [mbe]
            })
        ], ignore_index=True)       

    # save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    return metrics_df


def cropland_site_analysis_gridmet(
        gridmet_flux_csv: str,
        output_dir: str,
        gridmet_col: str = 'GRIDMET_REFERENCE_ET',
        flux_col: str = 'ASCE_ETo',
) -> None:
    """
    Perform site analysis for gridMET data over croplands.

    This function generates scatter plots of gridMET reference ET against flux tower ET for croplands.
    It calculates error metrics such as R2, RMSE, MAE, and MBE, and saves the results in a CSV file.

    Args:
        merged_gridmet_flux_csv (str): Path to the merged CSV file containing gridMET and flux tower data. 
        This file is generated from corr_analysis_gridmet.py.
        output_dir (str): Directory to save the output plots and metrics.

    Returns:
        None
    """
    
    metrics_csv_path = Path(output_dir) / "All_cropland_sites_gridmet_metrics.csv"
    if not metrics_csv_path.exists():
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        metrics_df = pd.DataFrame()
        print("Starting site analysis and plotting for gridMET data...")
        gridmet_flux_df = pd.read_csv(gridmet_flux_csv)[[
            "SITE_ID", "Latitude", "Longitude", "DATE",
            "General classification",
            gridmet_col, flux_col,
            "gridMET Corr_Uncorr"

        ]]
        gridmet_flux_df = gridmet_flux_df[gridmet_flux_df["General classification"] == "Croplands"]
        for sid in gridmet_flux_df["SITE_ID"].unique():
            df_site = gridmet_flux_df[gridmet_flux_df["SITE_ID"] == sid]
            outdir = Path(output_dir) / f"{sid}/"
            os.makedirs(outdir, exist_ok=True)
            m_df = flux_gridmet_plots(
                df_site,
                site_id=sid,
                output_dir=outdir
            )
            metrics_df = pd.concat([metrics_df, m_df], ignore_index=True)
        outdir = Path(output_dir) / "All/"
        os.makedirs(outdir, exist_ok=True)
        _ = flux_gridmet_plots(
            gridmet_flux_df,
            site_id='All',
            output_dir=outdir
        )
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Metrics saved to {metrics_csv_path}")
    else:  
        metrics_df = pd.read_csv(metrics_csv_path)
        print(f"Metrics loaded from {metrics_csv_path}")
    return metrics_df    


def query_metrics(
        metrics_df: pd.DataFrame, 
        plot_dir: str,
        error_metric: str | list [str] = "MBE"
) -> pd.DataFrame:
    """
    Query the metrics dataframe to find the sites where the bias correction improves the gridMET reference ET.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame containing the metrics for each site and model.
        plot_dir (str): Directory to copy the plots for the improved sites. The plots created by the `flux_et_plots` 
        function should be in this directory.
        error_metric (str | list[str]): The error metric to use for filtering. Default is "MBE". 
        Combinations of metrics can be provided as a list. For example, ["r2", "MAE", "MBE"].

    Returns:
        A tuple of (pd.DataFrame, str): DataFrame containing the site IDs where the bias correction improves the 
        gridMET reference ET. And the directory where the plots for these sites are saved.
    """

    # Filter the metrics DataFrame for corr and uncorr types
    corr_df = metrics_df[metrics_df["corr_uncorr_type"] == "Corrected"]
    uncorr_df = metrics_df[metrics_df["corr_uncorr_type"] == "Uncorrected"]

    # Merge the two DataFrames on site_id
    merged_df = pd.merge(
        corr_df, uncorr_df, 
        on="SITE_ID", 
        suffixes=('_corr', '_uncorr')
    )

    merged_df = merged_df.drop(columns=['Latitude_uncorr', 'Longitude_uncorr'])
    merged_df = merged_df.rename(columns={
        'Latitude_corr': 'Latitude',
        'Longitude_corr': 'Longitude'
    })

    # Calculate the differences in metrics
    merged_df['r2_diff'] = merged_df['r2_corr'] - merged_df['r2_uncorr']
    # merged_df['RMSE_diff'] = merged_df['RMSE_corr'] - merged_df['RMSE_uncorr']
    merged_df['MAE_diff'] = merged_df['MAE_corr'] - merged_df['MAE_uncorr']
    merged_df['MBE_diff'] = merged_df['MBE_corr'] - merged_df['MBE_uncorr']

    # Filter for sites where the bias correction improves the performance
    improved_sites = merged_df.copy()
    if isinstance(error_metric, list) and len(error_metric) == 1:
        error_metric = error_metric[0]
    if isinstance(error_metric, list) and len(error_metric) > 1:
        for metric in error_metric:
            print(f"Filtering for metric: {metric}")
            if metric == "r2":
                improved_sites = improved_sites[improved_sites['r2_diff'] > 0]
            elif metric in ["RMSE", "MAE", "MBE"]:
                improved_sites = improved_sites[improved_sites[f'{metric}_diff'] < 0]
            else:
                raise ValueError(f"Unsupported error metric: {metric}")
        error_metric_type = '_'.join(error_metric)
    else:
        if error_metric == "r2":
            improved_sites = merged_df[merged_df['r2_diff'] > 0]
        elif error_metric in ["RMSE", "MAE", "MBE"]:
            improved_sites = merged_df[merged_df[f'{error_metric}_diff'] < 0]
        else:
            raise ValueError(f"Unsupported error metric: {error_metric}")
        error_metric_type = error_metric
    
    improved_site_ids = improved_sites['SITE_ID'].tolist()
    if not improved_site_ids:
        print(f"No sites found with improved performance using metrics: {error_metric}")
        return []
    print(f"Sites with improved performance using metrics: {error_metric}:")
    print(improved_site_ids)
    improved_dir = Path(plot_dir) / f"Improved_Sites/{error_metric_type}"
    improved_dir.mkdir(parents=True, exist_ok=True)
    # Copy the plots for the improved sites to the output directory
    for site in improved_site_ids:
        site_plots_dir = Path(plot_dir) / site
        if site_plots_dir.exists():
            # move the site directory to the improved directory
            improved_site_dir = improved_dir / site
            copytree(site_plots_dir, improved_site_dir, dirs_exist_ok=True)
            print(f"Copied plots for site {site} to {improved_site_dir}")
        else:
            print(f"No plots found for site {site} in {plot_dir}. Skipping.")
    improved_sites.to_csv(improved_dir / f"gridmet_improved_metrics.csv", index=False)
    return improved_sites, improved_dir


def create_us_map_improved_sites(
        conus_shp: str,
        output_dir: str,
        metrics_df: pd.DataFrame
) -> None:
    """
    Create a US map showing improved sites with different color codes based on improvement metrics.
    Sites are prioritized as: All metrics (R2, MAE, MBE) > MBE > MAE > R2
    
    Args:
        conus_shp (str): Path to the CONUS shapefile.
        output_dir (str): Directory to save the map.
        metrics_df (pd.DataFrame): DataFrame containing the metrics for each site and model.
    
    Returns:
        None
    """
    
    # Load CONUS boundaries from shapefile
    try:
        conus_gdf = gpd.read_file(conus_shp)
        # only keep the lower 48 states
        conus_gdf = conus_gdf[~conus_gdf['STATE_ABBR'].isin(['AK', 'HI'])]
        conus_gdf = conus_gdf.to_crs("ESRI:102004")  # Albers Equal Area for CONUS
    except Exception as e:
        print(f"Failed to load CONUS boundaries from {conus_shp}: {e}")
        print("Creating map without state boundaries.")
        conus_gdf = None
    else:
        conus_gdf = conus_gdf.to_crs("ESRI:102004")  # Albers Equal Area for CONUS
    
    # Define metric combinations in priority order
    metric_combinations = [
        (["r2", "MAE", "MBE"], "All Metrics", "#2E8B57", "s", 100),  # Sea Green, square, large
        (["MBE"], "MBE", "#FF4500", "^", 80),                    # Orange Red, triangle, medium-large
        (["MAE"], "MAE", "#4169E1", "o", 60),                    # Royal Blue, circle, medium
        (["r2"], "R²", "#DC143C", "D", 40),                     # Crimson, diamond, small
    ]
    
    # Get improved sites for each metric combination
    improved_sites_dict = {}
    for metrics, label, color, marker, size in metric_combinations:
        try:
            improved_sites, improved_dir = query_metrics(
                metrics_df=metrics_df,
                plot_dir=output_dir,
                error_metric=metrics
            )
            
            if isinstance(improved_sites, pd.DataFrame) and not improved_sites.empty:
                improved_sites_dict[label] = {
                    'sites': improved_sites['SITE_ID'].tolist(),
                    'latitudes': improved_sites['Latitude'].tolist(),
                    'longitudes': improved_sites['Longitude'].tolist(),
                    'color': color,
                    'marker': marker,
                    'size': size
                }
                improved_sites_gdf = save_improved_sites_as_geojson(
                    improved_sites=improved_sites,
                    output_dir=improved_dir
                )
                # convert improved_sites_gdf to dictionary
                improved_sites_dict[label]['gdf'] = improved_sites_gdf
        except Exception as e:
            print(f"Error processing {label}: {e}")
            continue
    
    # Create the map
    _, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot CONUS state boundaries if available
    if conus_gdf is not None:
        conus_gdf.plot(ax=ax, facecolor='lightgray', edgecolor='black', linewidth=0.5, alpha=0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Set map extent for lower 48 states
    # ax.set_xlim(-125, -66)  # Longitude limits
    # ax.set_ylim(20, 50)     # Latitude limits
    
    # Plot improved sites with hierarchy (reverse order so higher priority appears on top)
    legend_elements = []
    
    for label, data in reversed(list(improved_sites_dict.items())):
        # Convert lat/lon to projected coordinates to match conus_gdf
        gdf = data['gdf']
        if conus_gdf is not None:
            # Reproject points to match the map projection
            gdf_projected = gdf.to_crs(conus_gdf.crs)
            x_coords = gdf_projected.geometry.x
            y_coords = gdf_projected.geometry.y
        else:
            # Use original coordinates if no projection
            x_coords = gdf['Longitude']
            y_coords = gdf['Latitude']
        
        site_ids = gdf['SITE_ID'].tolist()
        color = data['color']
        marker = data['marker']
        size = data['size']
        site_str = 'sites' if len(site_ids) > 1 else 'site'
        
        plt.rcParams.update({'font.size': 20})
        scatter = ax.scatter(
            x_coords,
            y_coords,
            c=color,
            marker=marker,
            s=size,
            alpha=0.9,
            edgecolors='black',
            linewidth=1,
            label=f"{label} ({len(site_ids)} {site_str})",
            zorder=10
        )
        legend_elements.append(scatter)
    
    # Styling
    # ax.set_xlabel('Longitude (°)', fontsize=20, fontweight='bold')
    # ax.set_ylabel('Latitude (°)', fontsize=20, fontweight='bold')
    
    # Add legend
    if legend_elements:
        legend = ax.legend(
            handles=legend_elements,
            loc='lower right',
            bbox_to_anchor=(0.98, 0.02),
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=20,
            title='Improvement Categories',
            title_fontsize=20
        )
        legend.get_frame().set_facecolor('#F8F8FF')
        legend.get_frame().set_alpha(0.2)
        legend.get_title().set_fontweight('bold')
    
    # Add text box with summary
    total_sites = len(metrics_df['SITE_ID'].unique())
    # Count unique improved SITE_IDs across all metric categories
    all_improved_sites = set()
    for data in improved_sites_dict.values():
        all_improved_sites.update(data['sites'])
    improved_count = len(all_improved_sites)
    
    textstr = f'Total Sites: {total_sites}\nImproved Sites: {improved_count}\nImprovement Rate: {improved_count/total_sites*100:.1f}%'
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.9, edgecolor='black')
    ax.text(0.02, 0.08, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='bottom', bbox=props, fontweight='bold')
    
    # Add grid
    # ax.grid(True, alpha=0.3, linestyle='--')
    # ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.grid(False)
    
    # Save the map
    plt.title('gridMET ETo', fontsize=24, fontweight='bold')
    plt.tight_layout()
    map_path = Path(output_dir) / "gridmet_improved_sites_map.png"
    plt.savefig(map_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"US map saved to {map_path}")
    
    # Create a summary table
    summary_data = []
    for label, data in improved_sites_dict.items():
        summary_data.append({
            'Metric_Combination': label,
            'Number_of_Sites': len(data['sites']),
            'Site_IDs': ', '.join(data['sites'])
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_path = Path(output_dir) / f"gridmet_improved_sites_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary table saved to {summary_path}")


def save_improved_sites_as_geojson(
        improved_sites: pd.DataFrame, 
        output_dir: str
) -> gpd.GeoDataFrame:
    """
    Save the improved sites as a GeoJSON file.

    Args:
        improved_sites (pd.DataFrame): DataFrame containing the improved sites.
        output_dir (str): Directory to save the GeoJSON files.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the improved sites.
    """

    gdf = gpd.GeoDataFrame(
        improved_sites, 
        geometry=gpd.points_from_xy(
            improved_sites["Longitude"], 
            improved_sites["Latitude"]
        ),
        crs="EPSG:4326"  # WGS 84
    )

    geojson_path = Path(output_dir) / "improved_sites.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"Saved improved sites to {geojson_path}")
    return gdf

if __name__ == "__main__":
    
    gridmet_flux_csv_dict = {
        'daily': "../../Plots/GridMET_Plots/All/GridMET_Daily_All_Station_Data.csv",
        'monthly': "../../Plots/GridMET_Plots/All/GridMET_Monthly_All_Station_Data.csv"
    }
    conus_shp = '../../Data/states/states.shp'

    # Create US map showing all improved sites with different color codes
    for dt in gridmet_flux_csv_dict.keys():
        output_directory = f"../../Plots/Site_Analysis_GridMET/{dt}/"
        metrics_df = cropland_site_analysis_gridmet(
            gridmet_flux_csv=gridmet_flux_csv_dict[dt],
            output_dir=output_directory
        )
        create_us_map_improved_sites(
            conus_shp=conus_shp,
            output_dir=output_directory,
            metrics_df=metrics_df
        )

        openet_improved_sites = f"../../Plots/Site_Analysis_OpenET/{dt}/All_metrics.csv"
        openet_metrics_df = pd.read_csv(openet_improved_sites)
        common_sites = set(metrics_df['SITE_ID']).intersection(set(openet_metrics_df['SITE_ID']))
        common_output_dir = Path(output_directory) / "Common_Sites"
        common_output_dir.mkdir(parents=True, exist_ok=True)
        if common_sites:
            print(f"Common sites between gridMET and OpenET for {dt}: {common_sites}")
            openet_metrics_df = openet_metrics_df[openet_metrics_df['SITE_ID'].isin(common_sites)]
            openet_metrics_df.to_csv(common_output_dir / "OpenET_Common_Sites_Metrics.csv", index=False)
            for openet_model in openet_metrics_df['openet_model'].unique():
                openet.create_us_map_improved_sites(
                    conus_shp=conus_shp,
                    output_dir=common_output_dir,
                    metrics_df=openet_metrics_df,
                    openet_model=openet_model
                )  

            gridmet_metrics_df = metrics_df[metrics_df['SITE_ID'].isin(common_sites)]
            gridmet_metrics_df.to_csv(common_output_dir / "GridMET_Common_Sites_Metrics.csv", index=False)
            create_us_map_improved_sites(
                conus_shp=conus_shp,
                output_dir=common_output_dir,
                metrics_df=gridmet_metrics_df
        )

