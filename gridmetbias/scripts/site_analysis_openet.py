"""
This script performs a site analysis for OpenET data, generating scatter plots
of OpenET ET against flux tower ET for different models and versions.
It calculates error metrics such as R2, RMSE, MAE, MBE, and Pearson correlation,
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
import matplotlib.pyplot as plt
from shutil import copytree
from pathlib import Path
from sklearn.metrics import mean_absolute_error


def flux_et_plots(
        df_site: pd.DataFrame,
        site_id: str,
        version: str,
        corr_uncorr_type: str,
        output_dir: str
) -> pd.DataFrame:
    """
    Generate scatter plots of OpenET ET against flux tower ET for different models and versions.
    This function calculates error metrics such as R2, RMSE, MAE, MBE, and Pearson correlation,
    and saves the results in a CSV file.

    Args:
        df_site (pd.DataFrame): DataFrame containing the site data.
        site_id (str): Site identifier.
        version (str): Version of the data.
        corr_uncorr_type (str): Type of correction (e.g., 'corr', 'uncorr').
        output_dir (str): Directory to save the output plots and metrics. 

    Returns:
        pd.DataFrame: DataFrame containing the calculated metrics for each model.
    """

    # flux_calcs = ["Closed", "Unclosed"]
    flux_calcs = ["Closed"]  # Assuming we only want to plot the closed flux calculations
    openet_models = [
        "EEMETRIC", 
        "SSEBOP", 
        "SIMS", 
        "GEESEBAL",
        "PTJPL",
        "DISALEXI",
        "ensemble_mean"
    ]
    dt_type = 'day' if 'daily' in str(output_dir) else 'month'
    metrics_df = pd.DataFrame()
    x_lim_max = 250 if dt_type == 'month' else 15  # Adjust max limit based on dt_type
    y_lim_max = 250 if dt_type == 'month' else 15  # Adjust max limit based on dt_type
    for flux_calc in flux_calcs:
        # Define the output file name based on version and type
        output_file = Path(output_dir) / f"{site_id}_{corr_uncorr_type}_{flux_calc.lower()}_scatter_plots_{version}.png"
        # Create subplots for each OpenET model with 3 rows and 3 columns
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
        axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easier iteration
        plt.rcParams['font.size'] = 14
        # Iterate over each OpenET model and create scatter plots
        for i, openet_model in enumerate(openet_models):
            et_flux_val = df_site[[openet_model, flux_calc]]
            # remove NaN and infinite values
            et_flux_val = et_flux_val.replace([np.inf, -np.inf], np.nan).dropna()
            if et_flux_val.empty:
                print(f"No data available for {openet_model} - {flux_calc} at {site_id}. Skipping plot.")
                continue
            # Extract the OpenET ET and flux tower ET values
            et_val = et_flux_val[openet_model]
            flux_actual = et_flux_val[flux_calc]
            ax = axes[i]
            sns.scatterplot(x=et_val, y=flux_actual, ax=ax)
            ax.set_title(openet_model)
            ax.set_xlabel(f"OpenET ET [mm/{dt_type}]")
            ax.set_ylabel(f"Station ET [mm/{dt_type}]")
            ax.grid()
            # plot 1:1 line
            max_val = max(et_val.max(), flux_actual.max())
            min_val = min(et_val.min(), flux_actual.min())
            ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='1:1 Line')
            ax.legend()
            ax.set_xlim(et_val.min() * 0.8, x_lim_max)  # Adjust x-limits to allow for better visibility
            ax.set_ylim(flux_actual.min() * 0.8, y_lim_max)  # Adjust y-limits to allow for better visibility
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Calculate error metrics R2, RMSE, MAE, and MBE

            r2 = np.corrcoef(et_val, flux_actual)[0, 1] ** 2
            # rmse = root_mean_squared_error(flux_actual, et_val)
            mae = mean_absolute_error(flux_actual, et_val)
            mbe = np.mean(flux_actual - et_val)

            # Add text box with error metrics
            metrics_text = (
                f"r$^2$: {r2:.2f}\n"
                # f"RMSE: {rmse:.2f} mm/{dt_type}\n"
                f"MAE: {mae:.2f} mm/{dt_type}\n"
                f"MBE: {mbe:.2f} mm/{dt_type}\n"
            )
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                    fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5)) 
            # Store metrics in DataFrame using pd.concat
            metrics_df = pd.concat([
                metrics_df,
                pd.DataFrame({
                    "SITE_ID": [site_id],
                    "version": [version],
                    "corr_uncorr_type": [corr_uncorr_type],
                    "flux_calc": [flux_calc],
                    "openet_model": [openet_model],
                    "r2": [r2],
                    # "RMSE": [rmse],
                    "MAE": [mae],
                    "MBE": [mbe]
                })
            ], ignore_index=True)
        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    return metrics_df


def cropland_site_analysis_openet(
        site_id: str | list[str],
        csv_dir: str,
        output_dir: str,
        station_metadata: str,
        return_only_metrics: bool = False,
        dt_type: str = 'monthly'
) -> None:
    """
    Perform site analysis for OpenET data over croplands.

    This function generates scatter plots of OpenET ET against flux tower ET for different models and versions.
    It calculates error metrics such as R2, RMSE, MAE, MBE, and Pearson correlation,
    and saves the results in a CSV file.

    Args:
        site_id (str or list[str]): Site identifier or list of site identifiers. If 'All', it processes all sites in the CSV.
        If a list, it processes only the specified sites. If 'West' or 'East', it processes sites in those regions based on the 100th meridian.
        csv_dir (str): Directory containing the CSV files with OpenET data.
        output_dir (str): Directory to save the output plots and metrics.
        station_metadata (str): Path to the station metadata Excel file.
        This is used to get the site geometry information.
        return_only_metrics (bool): If True, only return the metrics DataFrame without generating plots. Default is False.
        dt_type (str): The type of data to analyze (e.g., 'monthly', 'daily'). Default is 'monthly'.

    Returns:
        None
    """
    
    metrics_csv_path = Path(output_dir) / f"{site_id}_metrics.csv"
    if not return_only_metrics and not metrics_csv_path.exists():
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        versions = ["v3"]  # Example versions, adjust as needed
        corr_uncorr = ["corr", "uncorr"]
        metrics_df = pd.DataFrame()
        print("Starting site analysis and plotting for OpenET data...")
        station_metadata_df = pd.read_excel(station_metadata, skiprows=1)
        station_metadata_df = station_metadata_df[["Site ID", "Latitude", "Longitude"]]
        station_metadata_df.rename(columns={"Site ID": "SITE_ID"}, inplace=True)
        for version in versions:
            for corr_uncorr_type in corr_uncorr:
                # Load the data for the site
                df_path = Path(csv_dir) / f"merged_{dt_type}_{corr_uncorr_type}{version}.csv"
                df = pd.read_csv(df_path)
                df = df[df["General classification"] == "Croplands"]
                df = df.merge(
                    station_metadata_df,
                    on="SITE_ID",
                )
                if site_id == 'All':
                    site_ids = df["SITE_ID"].unique()
                elif isinstance(site_id, list):
                    site_ids = site_id
                elif site_id == 'West':
                    site_ids = df[df["Longitude"] < -100]["SITE_ID"].unique()
                elif site_id == 'East':
                    site_ids = df[df["Longitude"] >= -100]["SITE_ID"].unique()
                else:
                    site_ids = [site_id]
                for sid in site_ids:
                    df_site = df[df["SITE_ID"] == sid]
                    outdir = Path(output_dir) / f"{sid}/"
                    os.makedirs(outdir, exist_ok=True)
                    m_df = flux_et_plots(
                        df_site=df_site,
                        site_id=sid,
                        version=version,
                        corr_uncorr_type=corr_uncorr_type,
                        output_dir=outdir
                    )
                    metrics_df = pd.concat([metrics_df, m_df], ignore_index=True)

                    
                if site_id in ["All", "East", "West"] or isinstance(site_id, list):
                    df_sites = df[df["SITE_ID"].isin(site_ids)]
                    site_name = "All" if site_id == "All" else "East" if site_id == "East" else "West" if site_id == "West" else site_id
                    outdir = Path(output_dir) / f"{site_name}/"
                    os.makedirs(outdir, exist_ok=True)
                    m_list_df = flux_et_plots(
                        df_site=df_sites,
                        site_id=site_name,
                        version=version,
                        corr_uncorr_type=corr_uncorr_type,
                        output_dir=outdir
                    )
                    metrics_df = pd.concat([metrics_df, m_list_df], ignore_index=True)

        # Save the metrics DataFrame to a CSV file
        metrics_df = metrics_df.merge(station_metadata_df)
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Metrics saved to {metrics_csv_path}")
    else:  
        metrics_df = pd.read_csv(metrics_csv_path)
        print(f"Metrics loaded from {metrics_csv_path}")
    return metrics_df    


def query_metrics(
        metrics_df: pd.DataFrame, 
        plot_dir: str,
        openet_model = 'ensemble_mean',
        error_metric: str | list [str] = "MBE"
) -> pd.DataFrame:
    """
    Query the metrics dataframe to find the sites where the bias correction improves the OpenET model performance.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame containing the metrics for each site and model.
        plot_dir (str): Directory to copy the plots for the improved sites. The plots created by the `flux_et_plots` 
        function should be in this directory.
        openet_model (str): The OpenET model to filter the metrics. Default is 'ensemble_mean'.
        error_metric (str | list[str]): The error metric to use for filtering. Default is "MBE". 
        Combinations of metrics can be provided as a list. For example, ["r2", "RMSE", "MAE", "MBE"].

    Returns:
        A tuple of (pd.DataFrame, str): DataFrame containing the site IDs where the bias correction improves the 
        OpenET model performance. And the directory where the plots for these sites are saved.
    """

    # Filter the metrics DataFrame for corr and uncorr types
    metrics_df = metrics_df[metrics_df["openet_model"] == openet_model]
    corr_df = metrics_df[metrics_df["corr_uncorr_type"] == "corr"]
    uncorr_df = metrics_df[metrics_df["corr_uncorr_type"] == "uncorr"]

    # Merge the two DataFrames on site_id, version, flux_calc, and openet_model
    merged_df = pd.merge(
        corr_df, uncorr_df, 
        on=["SITE_ID", "version", "flux_calc", "openet_model"], 
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
        print(f"No sites found with improved performance for {openet_model} using metrics: {error_metric}")
        return []
    print(f"Sites with improved performance for {openet_model} using metrics: {error_metric}:")
    improved_dir = Path(plot_dir) / f"Improved_Sites/{openet_model}/{error_metric_type}"
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
    improved_sites.to_csv(improved_dir / f"improved_metrics.csv", index=False)
    return improved_sites, improved_dir


def download_conus_boundaries(output_dir: str) -> gpd.GeoDataFrame:
    """
    Download CONUS state boundaries from ESRI feature server.
    
    Args:
        output_dir (str): Directory to save the downloaded shapefile
        
    Returns:
        gpd.GeoDataFrame: CONUS state boundaries
    """
    import requests
    import geopandas as gpd
    from pathlib import Path
    
    # ESRI Feature Server URL
    base_url = "https://services1.arcgis.com/cRvLdSPAsRupRo7I/arcgis/rest/services/US_State_Boundaries_(CONUS)/FeatureServer/0/query"
    
    # Parameters for the query
    params = {
        'where': '1=1',  # Get all features
        'outFields': '*',  # Get all fields
        'f': 'geojson',   # Return as GeoJSON
        'returnGeometry': 'true'
    }
    
    output_path = Path(output_dir) / "conus_boundaries.geojson"
    
    # Check if file already exists
    if output_path.exists():
        print(f"CONUS boundaries already exist at {output_path}")
        return gpd.read_file(output_path)
    
    try:
        print("Downloading CONUS boundaries from ESRI feature server...")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        # Save the GeoJSON response
        with open(output_path, 'w') as f:
            f.write(response.text)
        
        # Load as GeoDataFrame
        gdf = gpd.read_file(output_path)
        print(f"Successfully downloaded CONUS boundaries to {output_path}")
        print(f"Downloaded {len(gdf)} state boundaries")
        
        return gdf
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading CONUS boundaries: {e}")
        return None
    except Exception as e:
        print(f"Error processing CONUS boundaries: {e}")
        return None


def create_us_map_improved_sites(
        conus_shp: str,
        output_dir: str,
        metrics_df: pd.DataFrame,
        openet_model: str = 'ensemble_mean'
) -> None:
    """
    Create a US map showing improved sites with different color codes based on improvement metrics.
    Sites are prioritized as: All metrics (R2, MAE, MBE) > MBE > MAE > R2
    
    Args:
        conus_shp (str): Path to the CONUS shapefile.
        output_dir (str): Directory to save the map.
        metrics_df (pd.DataFrame): DataFrame containing the metrics for each site and model.
        openet_model (str): The OpenET model to filter the metrics. Default is 'ensemble_mean'.
    
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
    
    # Define metric combinations in priority order
    metric_combinations = [
        (["r2", "MAE", "MBE"], "All Metrics", "#2E8B57", "s", 100),  # Sea Green, square, large
        (["MBE"], "MBE", "#FF4500", "^", 80),                    # Orange Red, triangle, medium-large
        (["MAE"], "MAE", "#4169E1", "o", 60),                    # Royal Blue, circle, medium
        (["r2"], "R²", "#DC143C", "D", 40),                     # Crimson, diamond, small
    ]
    title_dict = {
        'ensemble_mean': 'OpenET Ensemble',
        'EEMETRIC': 'eeMETRIC',
        'SSEBOP': 'SSEBop'
    }
    
    # Get improved sites for each metric combination
    improved_sites_dict = {}
    for metrics, label, color, marker, size in metric_combinations:
        try:
            improved_sites, improved_dir = query_metrics(
                metrics_df=metrics_df,
                plot_dir=output_dir,
                openet_model=openet_model,
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
        site_ids = data['sites']
        color = data['color']
        marker = data['marker']
        size = data['size']
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
            label=f"{label} ({len(site_ids)} sites)",
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
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # Save the map
    title = title_dict.get(openet_model, openet_model)
    plt.title(f"{title}", fontsize=24, fontweight='bold')
    plt.tight_layout()
    map_path = Path(output_dir) / f"{openet_model}_improved_sites_map.png"
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
        summary_path = Path(output_dir) / f"{openet_model}_improved_sites_summary.csv"
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
    
    # ca_site_id = "RIP760"  # Site in Central Valley, California
    site_id = "All"  # Sites west of the 100th meridian
    csv_directory = "../../Data/paired_flux_OpenET_data"  # Replace with actual path to CSV
    station_metadata = "../../Data/flux_ET_dataset/station_metadata.xlsx"
    conus_shp = '../../Data/states/states.shp'
    dt = ['daily', 'monthly']  # Data types to process

    for dt_type in dt:
        output_directory = f"../../Plots/Site_Analysis_OpenET/{dt_type}"  # Replace with actual output directory
        metrics_df = cropland_site_analysis_openet(
            site_id=site_id,
            csv_dir=csv_directory,
            output_dir=output_directory,
            station_metadata=station_metadata,
            return_only_metrics=False,
            dt_type=dt_type
        )
        print(metrics_df.columns)
        print(metrics_df.SITE_ID.unique())
        metrics_df = metrics_df[metrics_df.SITE_ID != 'List']  # Remove the 'List' site if it exists

        # Create US map showing all improved sites with different color codes
        for openet_model in metrics_df['openet_model'].unique():
            create_us_map_improved_sites(
                output_dir=output_directory,
                metrics_df=metrics_df,
                openet_model=openet_model,
                conus_shp=conus_shp
            )   
        