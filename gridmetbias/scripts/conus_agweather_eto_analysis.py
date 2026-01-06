# Script to analyze CONUS-AgWeather pre- and post-QC ETo data
# Author: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Define paths
DATA_DIR = Path('../../Data/CONUS-AgWeather_v1/standardized_data')
OUTPUT_DIR = Path('../../Data/Outputs')
PLOT_DIR = Path('../../Plots/CONUS-AgWeather_v1_ETo_Stats')
CLIMATE_PARQUET = Path('../../Data/supporting_files/Station_Climate/station_climate_data.parquet')

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def get_xlsx_files():
    """Get list of all xlsx data files."""
    return sorted([f for f in DATA_DIR.glob('*.xlsx') if not f.name.startswith('~$')])


def process_station_file(xlsx_path):
    """
    Process a single station xlsx file.
    
    Returns:
        dict: Station data including daily and annual QC factors, or None if incomplete record
    """
    station_id = xlsx_path.stem.replace('_data', '')
    
    try:
        # Read corrected data and delta sheets
        df_corr = pd.read_excel(xlsx_path, sheet_name='Corrected Data')
        df_delta = pd.read_excel(xlsx_path, sheet_name='Delta (Corr - Orig)')
        
        # Ensure Date column is datetime
        df_corr['Date'] = pd.to_datetime(df_corr['Date'])
        df_delta['Date'] = pd.to_datetime(df_delta['Date'])
        
        # Get ETo columns
        eto_corr = df_corr[['Date', 'ETo (mm/day)']].copy()
        eto_delta = df_delta[['Date', 'ETo (mm/day)']].copy()
        
        # Merge on Date
        merged = pd.merge(eto_corr, eto_delta, on='Date', suffixes=('_corr', '_delta'))
        
        # Calculate original (pre-QC) ETo: Original = Corrected - Delta
        merged['ETo_orig'] = merged['ETo (mm/day)_corr'] - merged['ETo (mm/day)_delta']
        merged['ETo_corr'] = merged['ETo (mm/day)_corr']
        merged['ETo_delta'] = merged['ETo (mm/day)_delta']
        
        # Drop rows with NaN in ETo values
        merged = merged.dropna(subset=['ETo_orig', 'ETo_corr'])
        
        # Extract year
        merged['Year'] = merged['Date'].dt.year
        merged['DOY'] = merged['Date'].dt.dayofyear
        
        return {
            'station_id': station_id,
            'data': merged
        }
        
    except Exception as e:
        print(f"Error processing {station_id}: {e}")
        return None


def find_complete_year_stations(all_station_data):
    """
    Find stations and years with complete 365/366 day records.
    
    Returns:
        list: List of dicts with station_id, year, and data for complete years
    """
    complete_records = []
    
    for station_info in all_station_data:
        if station_info is None:
            continue
            
        station_id = station_info['station_id']
        df = station_info['data']
        
        # Group by year and count valid records
        for year, year_df in df.groupby('Year'):
            # Check for leap year
            expected_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
            
            # Count unique days with valid ETo data
            valid_days = year_df['DOY'].nunique()
            
            if valid_days >= expected_days:
                complete_records.append({
                    'station_id': station_id,
                    'year': year,
                    'data': year_df,
                    'expected_days': expected_days,
                    'actual_days': valid_days
                })
    
    return complete_records


def calculate_qc_factors(complete_records):
    """
    Calculate pre and post QC factors at daily and annual time steps.
    
    Returns:
        tuple: (daily_factors_df, annual_factors_df)
    """
    daily_records = []
    annual_records = []
    
    for record in complete_records:
        station_id = record['station_id']
        year = record['year']
        df = record['data']
        
        # Daily factors
        for _, row in df.iterrows():
            if row['ETo_orig'] > 0:  # Avoid division by zero
                ratio = row['ETo_corr'] / row['ETo_orig']
                daily_records.append({
                    'station_id': station_id,
                    'year': year,
                    'date': row['Date'],
                    'DOY': row['DOY'],
                    'ETo_orig': row['ETo_orig'],
                    'ETo_corr': row['ETo_corr'],
                    'ETo_delta': row['ETo_delta'],
                    'ratio_post_pre': ratio
                })
        
        # Annual factors (sum of daily ETo)
        annual_eto_orig = df['ETo_orig'].sum()
        annual_eto_corr = df['ETo_corr'].sum()
        annual_eto_delta = df['ETo_delta'].sum()
        
        if annual_eto_orig > 0:  # Avoid division by zero
            annual_ratio = annual_eto_corr / annual_eto_orig
            annual_records.append({
                'station_id': station_id,
                'year': year,
                'annual_ETo_orig': annual_eto_orig,
                'annual_ETo_corr': annual_eto_corr,
                'annual_ETo_delta': annual_eto_delta,
                'annual_ratio_post_pre': annual_ratio,
                'n_days': len(df)
            })
    
    daily_df = pd.DataFrame(daily_records)
    annual_df = pd.DataFrame(annual_records)
    
    return daily_df, annual_df



def merge_with_climate_data(daily_df, annual_df):
    """
    Merge daily and annual dataframes with climate classification data.
    
    Returns:
        tuple: (daily_df_with_climate, annual_df_with_climate)
    """
    # Read climate parquet file
    climate_df = pd.read_parquet(CLIMATE_PARQUET)
    
    # Get unique station-climate mapping
    climate_mapping = climate_df[['Station ID', 'Climate_Abbreviation']].drop_duplicates()
    climate_mapping = climate_mapping.rename(columns={'Station ID': 'station_id'})
    
    # Replace None climates with 'Other'
    climate_mapping['Climate_Abbreviation'] = climate_mapping['Climate_Abbreviation'].fillna('Other')
    climate_mapping['Climate_Abbreviation'] = climate_mapping['Climate_Abbreviation'].replace('None', 'Other')
    climate_mapping.loc[climate_mapping['Climate_Abbreviation'].isna(), 'Climate_Abbreviation'] = 'Other'
    
    # Merge with daily and annual dataframes
    daily_df_climate = daily_df.merge(climate_mapping, on='station_id', how='left')
    annual_df_climate = annual_df.merge(climate_mapping, on='station_id', how='left')
    
    # Fill any remaining NaN climates with 'Other'
    daily_df_climate['Climate_Abbreviation'] = daily_df_climate['Climate_Abbreviation'].fillna('Other')
    annual_df_climate['Climate_Abbreviation'] = annual_df_climate['Climate_Abbreviation'].fillna('Other')
    
    return daily_df_climate, annual_df_climate


def plot_eto_ratio_climate_histogram_violin(daily_df, annual_df, n_stations):
    """
    Create histograms and violin plots of ETo ratio by climate classification.
    Uses the same Paired color palette as station_climate_plots.py
    """
    climate_col = 'Climate_Abbreviation'
    
    # Get sorted unique climate classes
    all_climates = sorted(set(daily_df[climate_col].unique()) | set(annual_df[climate_col].unique()))
    
    # Use Paired color palette (same as station_climate_plots.py)
    colors = plt.cm.Paired(np.linspace(0, 1, len(all_climates)))
    climate_colors = dict(zip(all_climates, colors))
    
    # --- Daily ETo Ratio Plot ---
    fig = plt.figure(figsize=(16, 8))
    
    # Create custom gridspec
    ax_kde = plt.subplot2grid((1, 5), (0, 0), colspan=3)
    ax_violin = plt.subplot2grid((1, 5), (0, 3), colspan=4)
    
    # Filter extreme values
    daily_plot_df_all = daily_df.copy()
    daily_plot_df = daily_plot_df_all[(daily_plot_df_all['ratio_post_pre'] > 0.92) & 
                                   (daily_plot_df_all['ratio_post_pre'] < 1.2)]
    
    overall_data = daily_plot_df['ratio_post_pre'].dropna()
    orig_data = daily_plot_df_all['ratio_post_pre'].dropna()
    
    if len(overall_data) > 1:
        # Plot histogram bars
        ax_kde.hist(
            overall_data,
            bins=50,
            orientation='horizontal',
            alpha=0.4,
            color='#D3D3D3',
            density=False,
            edgecolor='#808080',
            linewidth=0.5
        )
        
        # Plot KDE line
        ax_kde_twin = ax_kde.twiny()
        sns.kdeplot(
            y=overall_data,
            ax=ax_kde_twin,
            color="#000406",
            alpha=0.9,
            linewidth=4,
            fill=False,
            label=f'All sites (n={len(orig_data):,})'
        )
        
        ax_kde_twin.set_xlabel('')
        ax_kde_twin.tick_params(top=False, labeltop=False)
        ax_kde_twin.spines['top'].set_visible(False)
        ax_kde_twin.spines['right'].set_visible(False)
    
    # Create violin plot
    sns.violinplot(
        data=daily_plot_df,
        y='ratio_post_pre',
        hue=climate_col,
        ax=ax_violin,
        palette=climate_colors,
        hue_order=all_climates,
        dodge=True
    )
    
    # Style KDE axis
    ax_kde.set_ylabel('ETo Ratio (Post-QC / Pre-QC)', fontsize=18)
    ax_kde.set_xlabel('site-days', fontsize=18)
    ax_kde.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))
    ax_kde.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax_kde.tick_params(axis='both', labelsize=14)
    ax_kde.spines['top'].set_visible(False)
    ax_kde.spines['right'].set_visible(False)
    ax_kde.spines['left'].set_linewidth(2)
    ax_kde.spines['bottom'].set_linewidth(2)
    ax_kde.grid(False)
    
    # Style violin plot axis
    ax_violin.set_ylabel('')
    ax_violin.set_xlabel('')
    ax_violin.tick_params(axis='both', labelsize=14)
    ax_violin.set_xticklabels([])
    for spine in ax_violin.spines.values():
        spine.set_visible(False)
    ax_violin.grid(False)
    ax_violin.tick_params(left=False, bottom=False)
    
    # Match y-axis limits
    y_min = min(ax_kde.get_ylim()[0], ax_violin.get_ylim()[0])
    y_max = max(ax_kde.get_ylim()[1], ax_violin.get_ylim()[1])
    ax_kde.set_ylim(y_min, y_max)
    ax_violin.set_ylim(y_min, y_max)
    ax_violin.set_yticklabels([])
    ax_violin.tick_params(left=False)
    
    # Remove legends from subplots
    if ax_violin.get_legend():
        ax_violin.get_legend().remove()
    if ax_kde.get_legend():
        ax_kde.get_legend().remove()
    
    # Create legend
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor='#D3D3D3', alpha=0.4)]
    labels = [f'All sites ({len(orig_data):,})']
    
    for climate_val in all_climates:
        n_climate = daily_plot_df_all[daily_plot_df_all[climate_col] == climate_val].shape[0]
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=climate_colors[climate_val], alpha=0.8))
        labels.append(f'{climate_val} ({n_climate:,})')
    
    fig.legend(
        handles, labels,
        title=f"Site-days of ETo ratio (n = {n_stations} stations)",
        loc='upper center',
        bbox_to_anchor=(0.3, 1),
        ncol=min(len(all_climates) + 1, 2),
        fontsize=18,
        title_fontsize=18,
        frameon=False
    )
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ETo_QC_Ratio_Daily_Climate_Violin.png', dpi=600, bbox_inches='tight')
    print(f"Daily climate plot saved to {PLOT_DIR / 'ETo_QC_Ratio_Daily_Climate_Violin.png'}")
    
    # --- Annual ETo Ratio Plot ---
    fig = plt.figure(figsize=(16, 8))
    
    ax_kde = plt.subplot2grid((1, 5), (0, 0), colspan=3)
    ax_violin = plt.subplot2grid((1, 5), (0, 3), colspan=4)
    
    # Filter extreme values
    annual_plot_df_all = annual_df.copy()
    annual_plot_df = annual_plot_df_all[(annual_plot_df_all['annual_ratio_post_pre'] > 0.92) & 
                                     (annual_plot_df_all['annual_ratio_post_pre'] < 1.2)]
    
    overall_data = annual_plot_df['annual_ratio_post_pre'].dropna()
    orig_data = annual_plot_df_all['annual_ratio_post_pre'].dropna()
    
    if len(overall_data) > 1:
        ax_kde.hist(
            overall_data,
            bins=30,
            orientation='horizontal',
            alpha=0.4,
            color='#D3D3D3',
            density=False,
            edgecolor='#808080',
            linewidth=0.5
        )
        
        ax_kde_twin = ax_kde.twiny()
        sns.kdeplot(
            y=overall_data,
            ax=ax_kde_twin,
            color="#000406",
            alpha=0.9,
            linewidth=4,
            fill=False,
            label=f'All sites (n={len(overall_data):,})'
        )
        
        ax_kde_twin.set_xlabel('')
        ax_kde_twin.tick_params(top=False, labeltop=False)
        ax_kde_twin.spines['top'].set_visible(False)
        ax_kde_twin.spines['right'].set_visible(False)
    
    sns.violinplot(
        data=annual_plot_df,
        y='annual_ratio_post_pre',
        hue=climate_col,
        ax=ax_violin,
        palette=climate_colors,
        hue_order=all_climates,
        dodge=True
    )
    
    ax_kde.set_ylabel('Annual ETo Ratio (Post-QC / Pre-QC)', fontsize=18)
    ax_kde.set_xlabel('site-years', fontsize=18)
    ax_kde.xaxis.set_major_locator(plt.MaxNLocator(nbins=4, integer=True))
    ax_kde.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax_kde.tick_params(axis='both', labelsize=14)
    ax_kde.spines['top'].set_visible(False)
    ax_kde.spines['right'].set_visible(False)
    ax_kde.spines['left'].set_linewidth(2)
    ax_kde.spines['bottom'].set_linewidth(2)
    ax_kde.grid(False)
    
    ax_violin.set_ylabel('')
    ax_violin.set_xlabel('')
    ax_violin.tick_params(axis='both', labelsize=14)
    ax_violin.set_xticklabels([])
    for spine in ax_violin.spines.values():
        spine.set_visible(False)
    ax_violin.grid(False)
    ax_violin.tick_params(left=False, bottom=False)
    
    y_min = min(ax_kde.get_ylim()[0], ax_violin.get_ylim()[0])
    y_max = max(ax_kde.get_ylim()[1], ax_violin.get_ylim()[1])
    ax_kde.set_ylim(y_min, y_max)
    ax_violin.set_ylim(y_min, y_max)
    ax_violin.set_yticklabels([])
    ax_violin.tick_params(left=False)
    
    if ax_violin.get_legend():
        ax_violin.get_legend().remove()
    if ax_kde.get_legend():
        ax_kde.get_legend().remove()
    
    # Get unique stations for annual data
    n_annual_stations = annual_plot_df_all['station_id'].nunique()
    
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor='#D3D3D3', alpha=0.4)]
    labels = [f'All sites ({len(orig_data):,})']
    
    for climate_val in all_climates:
        n_climate = annual_plot_df_all[annual_plot_df_all[climate_col] == climate_val].shape[0]
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=climate_colors[climate_val], alpha=0.8))
        labels.append(f'{climate_val} ({n_climate:,})')
    
    fig.legend(
        handles, labels,
        title=f"Site-years of ETo ratio (n = {n_annual_stations} stations)",
        loc='upper center',
        bbox_to_anchor=(0.3, 1),
        ncol=min(len(all_climates) + 1, 2),
        fontsize=18,
        title_fontsize=18,
        frameon=False
    )
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ETo_QC_Ratio_Annual_Climate_Violin.png', dpi=600, bbox_inches='tight')
    print(f"Annual climate plot saved to {PLOT_DIR / 'ETo_QC_Ratio_Annual_Climate_Violin.png'}")


def print_summary_statistics(daily_df, annual_df, complete_records):
    """Print summary statistics of the QC analysis."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS: Pre and Post QC ETo Analysis")
    print("="*80)
    
    print(f"\n--- Dataset Overview ---")
    print(f"Total complete station-years: {len(complete_records)}")
    unique_stations = len(set([r['station_id'] for r in complete_records]))
    print(f"Unique stations with complete records: {unique_stations}")
    print(f"Total daily observations: {len(daily_df):,}")
    
    # Print station count prominently
    print(f"\n*** Number of stations used in analysis: {unique_stations} ***")
    
    print(f"\n--- Daily ETo Statistics ---")
    print(f"Pre-QC ETo (mm/day):  Mean = {daily_df['ETo_orig'].mean():.3f}, "
          f"Median = {daily_df['ETo_orig'].median():.3f}, Std = {daily_df['ETo_orig'].std():.3f}")
    print(f"Post-QC ETo (mm/day): Mean = {daily_df['ETo_corr'].mean():.3f}, "
          f"Median = {daily_df['ETo_corr'].median():.3f}, Std = {daily_df['ETo_corr'].std():.3f}")
    print(f"Delta (mm/day):       Mean = {daily_df['ETo_delta'].mean():.4f}, "
          f"Median = {daily_df['ETo_delta'].median():.4f}, Std = {daily_df['ETo_delta'].std():.4f}")
    
    valid_ratios = daily_df['ratio_post_pre'].dropna()
    valid_ratios = valid_ratios[(valid_ratios > 0) & (valid_ratios < 10)]
    print(f"Ratio (Post/Pre):     Mean = {valid_ratios.mean():.4f}, "
          f"Median = {valid_ratios.median():.4f}, Std = {valid_ratios.std():.4f}")
    
    print(f"\n--- Annual ETo Statistics ---")
    print(f"Pre-QC Annual ETo (mm):  Mean = {annual_df['annual_ETo_orig'].mean():.1f}, "
          f"Median = {annual_df['annual_ETo_orig'].median():.1f}")
    print(f"Post-QC Annual ETo (mm): Mean = {annual_df['annual_ETo_corr'].mean():.1f}, "
          f"Median = {annual_df['annual_ETo_corr'].median():.1f}")
    print(f"Annual Ratio (Post/Pre): Mean = {annual_df['annual_ratio_post_pre'].mean():.4f}, "
          f"Median = {annual_df['annual_ratio_post_pre'].median():.4f}")
    
    print("\n" + "="*80)


def main(load_existing=True):
    """Main function to run the ETo QC analysis.
    
    Args:
        load_existing: If True, load existing CSV files instead of reprocessing xlsx files.
    """
    print("Starting CONUS-AgWeather ETo QC Analysis...")
    print(f"Data directory: {DATA_DIR}")
    
    # Define output paths
    daily_output_path = OUTPUT_DIR / 'daily_eto_qc_factors.csv'
    annual_output_path = OUTPUT_DIR / 'annual_eto_qc_factors.csv'
    daily_climate_output_path = OUTPUT_DIR / 'daily_eto_qc_factors_with_climate.csv'
    annual_climate_output_path = OUTPUT_DIR / 'annual_eto_qc_factors_with_climate.csv'
    
    # Check if we can load existing files
    if load_existing and daily_output_path.exists() and annual_output_path.exists():
        print("\nLoading existing processed data files...")
        daily_df = pd.read_csv(daily_output_path)
        annual_df = pd.read_csv(annual_output_path)
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        print(f"Loaded daily data: {len(daily_df):,} records")
        print(f"Loaded annual data: {len(annual_df):,} records")
        
        # Get unique stations count
        n_stations = daily_df['station_id'].nunique()
        n_station_years = len(annual_df)
        
        print(f"\n*** Number of stations used in analysis: {n_stations} ***")
        print(f"Total complete station-years: {n_station_years}")
        print(f"Total daily observations: {len(daily_df):,}")
        
    else:
        print("\nProcessing xlsx files from scratch...")
        # Get all xlsx files
        xlsx_files = get_xlsx_files()
        print(f"Found {len(xlsx_files)} station files")
        
        # Process all station files
        print("\nProcessing station files...")
        all_station_data = []
        for xlsx_path in tqdm(xlsx_files, desc="Processing stations"):
            result = process_station_file(xlsx_path)
            if result is not None:
                all_station_data.append(result)
        
        print(f"Successfully processed {len(all_station_data)} stations")
        
        # Find stations with complete year records
        print("\nFinding stations with complete 365/366 day records...")
        complete_records = find_complete_year_stations(all_station_data)
        print(f"Found {len(complete_records)} complete station-year records")
        
        if len(complete_records) == 0:
            print("No complete records found. Exiting.")
            return
        
        # Calculate QC factors
        print("\nCalculating pre and post QC factors...")
        daily_df, annual_df = calculate_qc_factors(complete_records)
        
        # Print summary statistics
        print_summary_statistics(daily_df, annual_df, complete_records)
        
        # Save results to CSV
        daily_df.to_csv(daily_output_path, index=False)
        annual_df.to_csv(annual_output_path, index=False)
        print(f"\nDaily factors saved to: {daily_output_path}")
        print(f"Annual factors saved to: {annual_output_path}")
        
        n_stations = daily_df['station_id'].nunique()
    
    # Create histogram plots
    print("\nGenerating ETo ratio histograms...")
    
    # Check if climate-merged files exist
    if load_existing and daily_climate_output_path.exists() and annual_climate_output_path.exists():
        print("\nLoading existing climate-merged data files...")
        daily_df_climate = pd.read_csv(daily_climate_output_path)
        annual_df_climate = pd.read_csv(annual_climate_output_path)
        daily_df_climate['date'] = pd.to_datetime(daily_df_climate['date'])
        print(f"Loaded daily climate data: {len(daily_df_climate):,} records")
        print(f"Loaded annual climate data: {len(annual_df_climate):,} records")
    else:
        # Merge with climate data
        print("\nMerging with climate classification data...")
        daily_df_climate, annual_df_climate = merge_with_climate_data(daily_df, annual_df)
        
        # Save climate-merged data
        daily_df_climate.to_csv(daily_climate_output_path, index=False)
        annual_df_climate.to_csv(annual_climate_output_path, index=False)
        print(f"\nDaily factors with climate saved to: {daily_climate_output_path}")
        print(f"Annual factors with climate saved to: {annual_climate_output_path}")
    
    # Get number of unique stations
    n_stations = daily_df_climate['station_id'].nunique()
    print(f"Number of unique stations in analysis: {n_stations}")
    
    print("\nGenerating climate-based ETo ratio histograms and violin plots...")
    plot_eto_ratio_climate_histogram_violin(daily_df_climate, annual_df_climate, n_stations)
    
    print("\nAnalysis complete!")
    
    return daily_df, annual_df


if __name__ == '__main__':
    daily_df, annual_df = main(load_existing=True)
