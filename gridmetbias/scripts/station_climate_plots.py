# Plot station variables grouped by Koppen climate classification
# author: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob


def create_station_climate_csv(
        station_metadata_csv: str,
        output_dir: str,
        station_xls_dir: str,
        climate_csv: str,
        station_id_col: str = "Station",
        station_id_col_xls: str = "Station ID",
        climate_col: str = "Climate_Abbreviation",
        climate_station_id_col: str = "STATION_ID",
        date_col: str = "Date"
) -> pd.DataFrame:
    """
    Create a CSV file with climate data for each station for each year from 2008 to 2020.

    Args:
        station_metadata_csv (str): Path to the CSV file containing station metadata.
        output_dir (str): Directory to save output files.
        station_xls_dir (str): Directory containing station Excel files.
        climate_csv (str): Path to the CSV file containing climate classification data.
        station_id_col (str): Name of the column containing station IDs.
        station_id_col_xls (str): Name of the column containing station IDs in the Excel files.
        climate_col (str): Name of the column containing climate classification codes.
        climate_station_id_col (str): Name of the column containing station IDs in the climate CSV
        date_col (str): Name of the column containing date values.

    Returns:
        pd.DataFrame: DataFrame containing climate data for each station.
    """
      

    final_parquet = f"{output_dir}/station_climate_data.parquet"
    if not os.path.exists(final_parquet):
        os.makedirs(output_dir, exist_ok=True)
        climate_df = pd.read_csv(climate_csv)
        climate_df = climate_df[[climate_station_id_col, climate_col]].rename(
            columns={climate_station_id_col: station_id_col_xls}
        )
        climate_df.loc[climate_df[station_id_col_xls] == 'NEW_004_CO', station_id_col_xls] = '1105_CO'
        station_metadata_df = pd.read_csv(station_metadata_csv)
        station_metadata_df = station_metadata_df.rename(columns={station_id_col: station_id_col_xls})
        station_climate_df = station_metadata_df.merge(climate_df, on=station_id_col_xls)
        climate_dict = {
            "Bsk": "Bsk + Bsh",
            "BSh": "Bsk + Bsh",
            "BWh": "Bwh + Bwk",
            "Bwk": "Bwh + Bwk",
            "Cfa": "Cfa",
            "Csa": "Csa + Csb",
            "Csb": "Csa + Csb",
            "Dfa": "Dfa + Dfb",
            "Dfb": "Dfa + Dfb"
        }
        # Assign 'BWh' to CIMIS Mexico stations with missing climate data
        station_climate_df[climate_col] = station_climate_df[climate_col].fillna('BWh')
        station_climate_df[climate_col] = station_climate_df[climate_col].map(climate_dict)
        station_df_list = []
        for f in glob(f"{station_xls_dir}*.xlsx"):
            try:
                station_df_list.append(pd.read_excel(f))
            except Exception:
                print(f"Error reading {f}")
        station_xls_df = pd.concat(station_df_list, ignore_index=True)
        print('Number of station records in Excel files:', station_xls_df.shape[0])
        print('Number of unique stations in Excel files:', station_xls_df[station_id_col_xls].nunique())
        station_xls_df[date_col] = pd.to_datetime(station_xls_df[date_col])
        pre_merge_cols = station_climate_df.columns.tolist()
        station_climate_df = station_climate_df.merge(
            station_xls_df,
            on=station_id_col_xls,
            suffixes=('_meta', '')
        )
        drop_cols = [col + '_meta' for col in pre_merge_cols if col + '_meta' in station_climate_df.columns]
        station_climate_df = station_climate_df.drop(columns=drop_cols)
        station_climate_df.to_parquet(final_parquet, index=False)
        print(f'Created station climate parquet file, which has {station_climate_df.shape[0]} stations...')
    else:
        station_climate_df = pd.read_parquet(final_parquet)
    return station_climate_df


def make_station_climate_plots(
    station_climate_df: pd.DataFrame,
    output_dir: str,
    climate_col: str = 'Climate_Abbreviation',
) -> None:
    """
    Create plots for the climate data of each station with both violin and KDE plots. Plots are created individually.

    Args:
        station_climate_df (pd.DataFrame): DataFrame containing climate data for each station.
        output_dir (str): Directory to save the plots.
        climate_col (str): Column name for climate classification in the DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    station_vars = [
        'ETo (mm/day)', 'ETr (mm/day)', 'TMax (C)', 'TAvg (C)', 'TMin (C)', 'Ea (kPa)', 'TDew (C)',
        'RHMax (%)', 'RHAvg (%)', 'RHMin (%)', 'Compiled Ea (kPa)', 'Rs (w/m2)',
        'Optimized TR Rs (w/m2)', 'Rso (w/m2)', 'Measured Uz (m/s)',
        'Anemometer Height (m)', 'Uz at 2m (m/s)', 'Precipitation (mm)'
    ]
    for station_var in station_vars:
        station_clim_df = station_climate_df.dropna(subset=[station_var]).copy()
        site_days = station_clim_df.shape[0]
        # print('Number of stations:', station_climate_df['Station ID'].nunique())
        print(f'Number of site-days for {station_var}: {site_days:,}')
    
        station_clim_df[climate_col] = station_clim_df[climate_col].astype(str)
        station_clim_df[climate_col] = station_clim_df[climate_col].replace('None', 'Other')
        climate = sorted(station_clim_df[climate_col].unique())

        # Use Paired color palette as requested
        colors = plt.cm.Paired(np.linspace(0, 1, len(climate)))
        climate_colors = dict(zip(climate, colors))

        # Create subplots with custom gridspec for better control
        fig = plt.figure(figsize=(16, 8))
        
        # Create custom gridspec - reduced colspan for 2 figures per plot
        # Left side for KDE (2 columns), right side for violin (2 columns) - no gap
        ax_kde = plt.subplot2grid((1, 5), (0, 0), colspan=3)
        ax_violin = plt.subplot2grid((1, 5), (0, 3), colspan=4)
        
        # Plot overall KDE on left axis with histogram bars
        overall_data = station_clim_df[station_var].copy()
        if len(overall_data) > 1:
            # Plot histogram bars in light gray with counts (not density)
            ax_kde.hist(
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
            
        
        # Create very narrow violin plot on right axis
        sns.violinplot(
            data=station_clim_df, 
            y=station_var, 
            hue=climate_col,
            ax=ax_violin,
            palette=climate_colors,
            hue_order=climate,
            #width=0.5,  # Make violin plots much narrower
            dodge=True
        )
        # # Set variable label with bigger font
        # var_label = f"({chr(97 + var_idx)})"
        # ax_violin.text(-0.1, 1.05, var_label, transform=ax_violin.transAxes, 
        #             fontsize=20, fontweight='bold', ha='left')
        
        # Style KDE axis (left) - only keep left and bottom spines
        if station_var in ['TMax (C)', 'TMin (C)', 'TAvg (C)', 'TDew (C)']:
            var_unit_label = station_var.replace(' (C)', ' (°C)')
        elif station_var in ['ETo (mm/day)', 'ETr (mm/day)']:
            var_unit_label = station_var.replace(' (mm/day)', ' (mm)')
        elif station_var in ['Measured Uz (m/s)', 'Uz at 2m (m/s)']:
            var_unit_label = station_var.replace(' (m/s)', ' (m s⁻¹)')
        elif station_var in ['Rs (w/m2)', 'Rso (w/m2)', 'Optimized TR Rs (w/m2)']:
            var_unit_label = station_var.replace(' (w/m2)', ' (W m⁻²)')
        else:
            var_unit_label = station_var
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
        
        # Style violin plot axis (right) - remove all spines
        ax_violin.set_ylabel('')  # Remove y-label since KDE axis has it
        ax_violin.set_xlabel('')
        ax_violin.tick_params(axis='both', labelsize=14)
        ax_violin.set_xticklabels([])  # Remove x-axis labels
        
        # Remove all spines for violin plot
        for spine in ax_violin.spines.values():
            spine.set_visible(False)
        ax_violin.grid(False)
        ax_violin.tick_params(left=False, bottom=False)  # Remove tick marks

        # Match y-axis limits between KDE and violin plot for non-ET variables
        y_min = min(ax_kde.get_ylim()[0], ax_violin.get_ylim()[0])
        y_max = max(ax_kde.get_ylim()[1], ax_violin.get_ylim()[1])
        ax_kde.set_ylim(y_min, y_max)
        ax_violin.set_ylim(y_min, y_max)
        
        # Remove y-axis labels and ticks from violin plot
        ax_violin.set_yticklabels([])
        ax_violin.tick_params(left=False)
        
        # Remove individual legends from each subplot
        if ax_violin.get_legend():
            ax_violin.get_legend().remove()
        if ax_kde.get_legend():
            ax_kde.get_legend().remove()

        # Add single legend for crops at the top of the figure with bigger font
        handles = []
        labels = []
        
        # Add overall sites legend first
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='#D3D3D3', alpha=0.4))
        n_total = station_clim_df.shape[0]
        labels.append(f'All sites ({n_total:,})')
        
        # Get crop colors for legend
        for climate_val in climate:
            n_climate = station_clim_df[station_clim_df[climate_col] == climate_val].shape[0]
            handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=climate_colors[climate_val], alpha=0.8))
            labels.append(f'{climate_val} ({n_climate:,})')
        # Create legend at the top with bigger font
        var_name = station_var.split('(')[0].strip()
        legend_title = f"Site-days of {var_name}"

        # Create legend at the top with bigger font
        fig.legend(
            handles, labels, 
            title=legend_title,
            loc='upper center', 
            bbox_to_anchor=(0.3, 1),
            ncol=min(len(climate) + 1, 2), 
            fontsize=18,
            title_fontsize=18,
            frameon=False # Remove legend border
        )  
        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{var_name}_climate_plots.png'), dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    station_metadata_csv = "../../Data/CONUS-AgWeather_v1/metadata_for_publication.csv"
    station_xls_dir = "../../Data/CONUS-AgWeather_v1/standardized_data/"
    climate_csv = "../../Data/Point bias data/Climate/u2_ms_merged_with_climate.csv"
    output_directory = "../../Data/supporting_files/Station_Climate/"
    station_climate_df = create_station_climate_csv(
        station_metadata_csv, 
        output_directory, 
        station_xls_dir,
        climate_csv
    )
    plot_dir = "../../Plots/Station_Climate/"
    make_station_climate_plots(station_climate_df, plot_dir)