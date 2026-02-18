# Script to analyze CONUS-AgWeather pre- and post-QC ETo data
# Author: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

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


def plot_qc_effect_distribution(daily_df, annual_df, exclude_ratio_one=True):
    """
    Create distribution plots showing the effect of QC on ETo values.
    
    Plots show percent change between post-QC and pre-QC ETo values,
    optionally excluding days/years where no correction was applied (ratio=1).
    
    Args:
        daily_df: DataFrame with daily QC factors (must have 'ratio_post_pre' column)
        annual_df: DataFrame with annual QC factors (must have 'annual_ratio_post_pre' column)
        exclude_ratio_one: If True, exclude records where ratio equals 1 (no correction)
    """
    # Calculate percent change
    daily_df = daily_df.copy()
    annual_df = annual_df.copy()
    daily_df['pct_diff'] = (daily_df['ratio_post_pre'] - 1) * 100
    annual_df['pct_diff'] = (annual_df['annual_ratio_post_pre'] - 1) * 100
    
    # Store original counts
    daily_total = len(daily_df)
    annual_total = len(annual_df)
    daily_ratio_one = (daily_df['ratio_post_pre'] == 1).sum()
    annual_ratio_one = (annual_df['annual_ratio_post_pre'] == 1).sum()
    
    # Filter out ratio == 1 if requested
    if exclude_ratio_one:
        daily_filtered = daily_df[daily_df['ratio_post_pre'] != 1].copy()
        annual_filtered = annual_df[annual_df['annual_ratio_post_pre'] != 1].copy()
    else:
        daily_filtered = daily_df.copy()
        annual_filtered = annual_df.copy()
    
    # Trim to 1st-99th percentile for visualization
    daily_p1, daily_p99 = daily_filtered['pct_diff'].quantile([0.01, 0.99])
    annual_p1, annual_p99 = annual_filtered['pct_diff'].quantile([0.01, 0.99])
    
    daily_trimmed = daily_filtered[(daily_filtered['pct_diff'] >= daily_p1) & 
                                   (daily_filtered['pct_diff'] <= daily_p99)]
    annual_trimmed = annual_filtered[(annual_filtered['pct_diff'] >= annual_p1) & 
                                     (annual_filtered['pct_diff'] <= annual_p99)]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Color palette
    colors = {'all': '#1f77b4', 'filtered': '#ff7f0e', 'trimmed': '#2ca02c'}
    
    # --- Daily plots ---
    # Left: All data vs filtered (histogram)
    ax1 = axes[0, 0]
    ax1.hist(daily_df['pct_diff'].clip(-50, 50), bins=100, alpha=0.5, 
             label=f'All data (n={daily_total:,})', color=colors['all'], density=True)
    ax1.hist(daily_filtered['pct_diff'].clip(-50, 50), bins=100, alpha=0.5, 
             label=f'Excluding ratio=1 (n={len(daily_filtered):,})', color=colors['filtered'], density=True)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_xlabel('QC Percent Change (%)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Daily QC Effect: All Data vs Excluding No-Correction Days', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xlim(-50, 50)
    
    # Right: Filtered and trimmed (cleaner view)
    ax2 = axes[0, 1]
    ax2.hist(daily_trimmed['pct_diff'], bins=100, alpha=0.7, color=colors['trimmed'], 
             density=True, edgecolor='white', linewidth=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='No change')
    ax2.axvline(x=daily_trimmed['pct_diff'].median(), color='black', linestyle='-', linewidth=1.5, 
                label=f'Median: {daily_trimmed["pct_diff"].median():.2f}%')
    ax2.set_xlabel('QC Percent Change (%)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title(f'Daily QC Effect Distribution (1st-99th percentile, n={len(daily_trimmed):,})', 
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    
    # --- Annual plots ---
    # Left: All data vs filtered
    ax3 = axes[1, 0]
    ax3.hist(annual_df['pct_diff'], bins=50, alpha=0.5, 
             label=f'All data (n={annual_total:,})', color=colors['all'], density=True)
    ax3.hist(annual_filtered['pct_diff'], bins=50, alpha=0.5, 
             label=f'Excluding ratio=1 (n={len(annual_filtered):,})', color=colors['filtered'], density=True)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax3.set_xlabel('QC Percent Change (%)', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Annual QC Effect: All Data vs Excluding No-Correction Years', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    
    # Right: Filtered and trimmed
    ax4 = axes[1, 1]
    ax4.hist(annual_trimmed['pct_diff'], bins=50, alpha=0.7, color=colors['trimmed'], 
             density=True, edgecolor='white', linewidth=0.3)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='No change')
    ax4.axvline(x=annual_trimmed['pct_diff'].median(), color='black', linestyle='-', linewidth=1.5, 
                label=f'Median: {annual_trimmed["pct_diff"].median():.2f}%')
    ax4.set_xlabel('QC Percent Change (%)', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title(f'Annual QC Effect Distribution (1st-99th percentile, n={len(annual_trimmed):,})', 
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'qc_effect_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"QC effect comparison plot saved to {PLOT_DIR / 'qc_effect_distribution_comparison.png'}")
    
    # Print summary statistics
    print("\n=== QC Effect Summary Statistics ===")
    print(f"\nDaily Data:")
    print(f"  All records: {daily_total:,}")
    print(f"  Records with ratio=1 (no correction): {daily_ratio_one:,} ({daily_ratio_one/daily_total*100:.1f}%)")
    print(f"  Records with correction: {len(daily_filtered):,} ({len(daily_filtered)/daily_total*100:.1f}%)")
    print(f"  Median QC effect (filtered): {daily_filtered['pct_diff'].median():.2f}%")
    print(f"  Mean QC effect (filtered): {daily_filtered['pct_diff'].mean():.2f}%")
    
    print(f"\nAnnual Data:")
    print(f"  All records: {annual_total:,}")
    print(f"  Records with ratio=1: {annual_ratio_one:,} ({annual_ratio_one/annual_total*100:.1f}%)")
    print(f"  Records with correction: {len(annual_filtered):,} ({len(annual_filtered)/annual_total*100:.1f}%)")
    print(f"  Median QC effect (filtered): {annual_filtered['pct_diff'].median():.2f}%")
    print(f"  Mean QC effect (filtered): {annual_filtered['pct_diff'].mean():.2f}%")
    
    return daily_filtered, annual_filtered


def plot_qc_effect_by_climate(daily_df, annual_df, exclude_ratio_one=True):
    """
    Create distribution plots showing QC effect by climate classification.
    
    Args:
        daily_df: DataFrame with daily QC factors and Climate_Abbreviation column
        annual_df: DataFrame with annual QC factors and Climate_Abbreviation column
        exclude_ratio_one: If True, exclude records where ratio equals 1 (no correction)
    """
    climate_col = 'Climate_Abbreviation'
    
    # Check if climate column exists
    if climate_col not in daily_df.columns or climate_col not in annual_df.columns:
        print("Warning: Climate_Abbreviation column not found. Skipping climate plots.")
        return
    
    # Calculate percent change
    daily_df = daily_df.copy()
    annual_df = annual_df.copy()
    daily_df['pct_diff'] = (daily_df['ratio_post_pre'] - 1) * 100
    annual_df['pct_diff'] = (annual_df['annual_ratio_post_pre'] - 1) * 100
    
    # Filter out ratio == 1 if requested
    if exclude_ratio_one:
        daily_filtered = daily_df[daily_df['ratio_post_pre'] != 1].copy()
        annual_filtered = annual_df[annual_df['annual_ratio_post_pre'] != 1].copy()
    else:
        daily_filtered = daily_df.copy()
        annual_filtered = annual_df.copy()
    
    # Get sorted unique climate classes
    all_climates = sorted(set(daily_filtered[climate_col].unique()) | set(annual_filtered[climate_col].unique()))
    
    # Use Paired color palette
    colors = plt.cm.Paired(np.linspace(0, 1, len(all_climates)))
    climate_colors = dict(zip(all_climates, colors))
    
    # --- Daily QC Effect by Climate (Box Plot) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Trim to 1st-99th percentile for visualization
    daily_p1, daily_p99 = daily_filtered['pct_diff'].quantile([0.01, 0.99])
    annual_p1, annual_p99 = annual_filtered['pct_diff'].quantile([0.01, 0.99])
    
    daily_trimmed = daily_filtered[(daily_filtered['pct_diff'] >= daily_p1) & 
                                   (daily_filtered['pct_diff'] <= daily_p99)]
    annual_trimmed = annual_filtered[(annual_filtered['pct_diff'] >= annual_p1) & 
                                     (annual_filtered['pct_diff'] <= annual_p99)]
    
    # Daily boxplot
    ax1 = axes[0]
    box_data_daily = [daily_trimmed[daily_trimmed[climate_col] == c]['pct_diff'].values 
                      for c in all_climates]
    bp1 = ax1.boxplot(box_data_daily, labels=all_climates, patch_artist=True, 
                      showfliers=False, widths=0.6)
    for patch, color in zip(bp1['boxes'], [climate_colors[c] for c in all_climates]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Climate Zone', fontsize=12)
    ax1.set_ylabel('QC Percent Change (%)', fontsize=12)
    ax1.set_title('Daily QC Effect by Climate Zone\n(Excluding No-Correction Days, 1st-99th percentile)', 
                  fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add sample size labels
    for i, climate in enumerate(all_climates):
        n = len(daily_trimmed[daily_trimmed[climate_col] == climate])
        ax1.text(i + 1, ax1.get_ylim()[1], f'n={n:,}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Annual boxplot
    ax2 = axes[1]
    box_data_annual = [annual_trimmed[annual_trimmed[climate_col] == c]['pct_diff'].values 
                       for c in all_climates]
    bp2 = ax2.boxplot(box_data_annual, labels=all_climates, patch_artist=True, 
                      showfliers=False, widths=0.6)
    for patch, color in zip(bp2['boxes'], [climate_colors[c] for c in all_climates]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Climate Zone', fontsize=12)
    ax2.set_ylabel('QC Percent Change (%)', fontsize=12)
    ax2.set_title('Annual QC Effect by Climate Zone\n(Excluding No-Correction Years, 1st-99th percentile)', 
                  fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add sample size labels
    for i, climate in enumerate(all_climates):
        n = len(annual_trimmed[annual_trimmed[climate_col] == climate])
        ax2.text(i + 1, ax2.get_ylim()[1], f'n={n:,}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'qc_effect_by_climate_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"QC effect by climate boxplot saved to {PLOT_DIR / 'qc_effect_by_climate_boxplot.png'}")
    
    # --- Violin plots by climate ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Daily violin plot
    ax1 = axes[0]
    sns.violinplot(data=daily_trimmed, x=climate_col, y='pct_diff', ax=ax1,
                   palette=climate_colors, order=all_climates, inner='box', cut=0)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Climate Zone', fontsize=12)
    ax1.set_ylabel('QC Percent Change (%)', fontsize=12)
    ax1.set_title('Daily QC Effect Distribution by Climate Zone\n(Excluding No-Correction Days, 1st-99th percentile)', 
                  fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Annual violin plot
    ax2 = axes[1]
    sns.violinplot(data=annual_trimmed, x=climate_col, y='pct_diff', ax=ax2,
                   palette=climate_colors, order=all_climates, inner='box', cut=0)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Climate Zone', fontsize=12)
    ax2.set_ylabel('QC Percent Change (%)', fontsize=12)
    ax2.set_title('Annual QC Effect Distribution by Climate Zone\n(Excluding No-Correction Years, 1st-99th percentile)', 
                  fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'qc_effect_by_climate_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"QC effect by climate violin plot saved to {PLOT_DIR / 'qc_effect_by_climate_violin.png'}")
    
    # --- Print statistics by climate ---
    print("\n=== QC Effect Statistics by Climate Zone ===")
    print("\nDaily QC Effect (filtered, excluding ratio=1):")
    print("-" * 70)
    print(f"{'Climate':<10} {'Count':>12} {'Median %':>12} {'Mean %':>12} {'Std %':>12}")
    print("-" * 70)
    for climate in all_climates:
        subset = daily_trimmed[daily_trimmed[climate_col] == climate]['pct_diff']
        print(f"{climate:<10} {len(subset):>12,} {subset.median():>12.2f} {subset.mean():>12.2f} {subset.std():>12.2f}")
    
    print("\nAnnual QC Effect (filtered, excluding ratio=1):")
    print("-" * 70)
    print(f"{'Climate':<10} {'Count':>12} {'Median %':>12} {'Mean %':>12} {'Std %':>12}")
    print("-" * 70)
    for climate in all_climates:
        subset = annual_trimmed[annual_trimmed[climate_col] == climate]['pct_diff']
        print(f"{climate:<10} {len(subset):>12,} {subset.median():>12.2f} {subset.mean():>12.2f} {subset.std():>12.2f}")


def plot_qc_effect_combined_histogram_violin(daily_df, annual_df):
    """
    Create combined histogram + violin plots for QC'ed data excluding ratio=1.
    Similar layout to plot_eto_ratio_climate_histogram_violin but shows percent change
    and only includes records where QC was actually applied (ratio != 1).
    
    Args:
        daily_df: DataFrame with daily QC factors and Climate_Abbreviation column
        annual_df: DataFrame with annual QC factors and Climate_Abbreviation column
    """
    climate_col = 'Climate_Abbreviation'
    
    # Check if climate column exists
    if climate_col not in daily_df.columns or climate_col not in annual_df.columns:
        print("Warning: Climate_Abbreviation column not found. Skipping combined plots.")
        return
    
    # Calculate percent change
    daily_df = daily_df.copy()
    annual_df = annual_df.copy()
    daily_df['pct_diff'] = (daily_df['ratio_post_pre'] - 1) * 100
    annual_df['pct_diff'] = (annual_df['annual_ratio_post_pre'] - 1) * 100
    
    # Filter out ratio == 1 (no correction applied)
    daily_filtered = daily_df[daily_df['ratio_post_pre'] != 1].copy()
    annual_filtered = annual_df[annual_df['annual_ratio_post_pre'] != 1].copy()
    
    # Get sorted unique climate classes
    all_climates = sorted(set(daily_filtered[climate_col].unique()) | set(annual_filtered[climate_col].unique()))
    
    # Use Paired color palette
    colors = plt.cm.Paired(np.linspace(0, 1, len(all_climates)))
    climate_colors = dict(zip(all_climates, colors))
    
    # Trim to 1st-99th percentile for visualization
    daily_p1, daily_p99 = daily_filtered['pct_diff'].quantile([0.01, 0.99])
    annual_p1, annual_p99 = annual_filtered['pct_diff'].quantile([0.01, 0.99])
    
    daily_trimmed = daily_filtered[(daily_filtered['pct_diff'] >= daily_p1) & 
                                   (daily_filtered['pct_diff'] <= daily_p99)]
    annual_trimmed = annual_filtered[(annual_filtered['pct_diff'] >= annual_p1) & 
                                     (annual_filtered['pct_diff'] <= annual_p99)]
    
    # Get number of unique stations
    n_stations_daily = daily_filtered['station_id'].nunique()
    n_stations_annual = annual_filtered['station_id'].nunique()
    
    # ========== DAILY PLOT ==========
    fig = plt.figure(figsize=(16, 8))
    
    # Create custom gridspec
    ax_kde = plt.subplot2grid((1, 5), (0, 0), colspan=3)
    ax_violin = plt.subplot2grid((1, 5), (0, 3), colspan=4)
    
    overall_data = daily_trimmed['pct_diff'].dropna()
    orig_data = daily_filtered['pct_diff'].dropna()
    
    if len(overall_data) > 1:
        # Plot histogram bars
        ax_kde.hist(
            overall_data,
            bins=50,
            orientation='horizontal',
            alpha=0.4,
            color='#A9A9A9',
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
    
    # Add reference lines
    data_mean = overall_data.mean()
    data_std = overall_data.std()
    ax_kde.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_kde.axhline(y=overall_data.median(), color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    ax_kde.axhline(y=data_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    # Add ±std lines
    ax_kde.axhline(y=data_mean + data_std, color='green', linestyle='-.', linewidth=1.2, alpha=0.5)
    ax_kde.axhline(y=data_mean - data_std, color='green', linestyle='-.', linewidth=1.2, alpha=0.5)
    # Add shaded region for ±std
    ax_kde.axhspan(data_mean - data_std, data_mean + data_std, alpha=0.1, color='green')
    
    # Create violin plot
    sns.violinplot(
        data=daily_trimmed,
        y='pct_diff',
        hue=climate_col,
        ax=ax_violin,
        palette=climate_colors,
        hue_order=all_climates,
        dodge=True,
        inner='box',
        cut=0
    )
    
    # Add reference line to violin plot
    ax_violin.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Style KDE axis
    ax_kde.set_ylabel('ETo QC Percent Change (%)', fontsize=18)
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
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor='#A9A9A9', alpha=0.4, edgecolor='#808080', linewidth=1)]
    labels = [f'All QC\'d sites ({len(orig_data):,})']
    
    # Add reference line legends (using trimmed data stats)
    trimmed_mean = overall_data.mean()
    trimmed_std = overall_data.std()
    handles.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5))
    labels.append('No change (0%)')
    handles.append(plt.Line2D([0], [0], color='blue', linestyle='-', linewidth=1.5))
    labels.append(f'Median ({overall_data.median():.2f}%)')
    handles.append(plt.Line2D([0], [0], color='green', linestyle=':', linewidth=1.5))
    labels.append(f'Mean ({trimmed_mean:.2f}%)')
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.2, edgecolor='green', linestyle='-.'))
    labels.append(f'±1 Std ({trimmed_std:.2f}%)')
    
    for climate_val in all_climates:
        n_climate = daily_filtered[daily_filtered[climate_col] == climate_val].shape[0]
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=climate_colors[climate_val], alpha=0.8))
        labels.append(f'{climate_val} ({n_climate:,})')
    
    fig.legend(
        handles, labels,
        title=f"Daily ETo QC Effect - Excluding No-Correction Days (n = {n_stations_daily} stations)",
        loc='upper center',
        bbox_to_anchor=(0.35, 1),
        ncol=3,
        fontsize=14,
        title_fontsize=14,
        frameon=False
    )
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ETo_QC_PctDiff_Daily_Climate_Violin_Filtered.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Daily ETo QC effect climate plot saved to {PLOT_DIR / 'ETo_QC_PctDiff_Daily_Climate_Violin_Filtered.png'}")
    
    # ========== ANNUAL PLOT ==========
    fig = plt.figure(figsize=(16, 8))
    
    ax_kde = plt.subplot2grid((1, 5), (0, 0), colspan=3)
    ax_violin = plt.subplot2grid((1, 5), (0, 3), colspan=4)
    
    overall_data = annual_trimmed['pct_diff'].dropna()
    orig_data = annual_filtered['pct_diff'].dropna()
    
    if len(overall_data) > 1:
        # Plot histogram bars
        ax_kde.hist(
            overall_data,
            bins=30,
            orientation='horizontal',
            alpha=0.4,
            color='#A9A9A9',
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
    
    # Add reference lines
    data_mean = overall_data.mean()
    data_std = overall_data.std()
    ax_kde.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_kde.axhline(y=overall_data.median(), color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    ax_kde.axhline(y=data_mean, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    # Add ±std lines
    ax_kde.axhline(y=data_mean + data_std, color='green', linestyle='-.', linewidth=1.2, alpha=0.5)
    ax_kde.axhline(y=data_mean - data_std, color='green', linestyle='-.', linewidth=1.2, alpha=0.5)
    # Add shaded region for ±std
    ax_kde.axhspan(data_mean - data_std, data_mean + data_std, alpha=0.1, color='green')
    
    # Create violin plot
    sns.violinplot(
        data=annual_trimmed,
        y='pct_diff',
        hue=climate_col,
        ax=ax_violin,
        palette=climate_colors,
        hue_order=all_climates,
        dodge=True,
        inner='box',
        cut=0
    )
    
    # Add reference line to violin plot
    ax_violin.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Style KDE axis
    ax_kde.set_ylabel('ETo QC Percent Change (%)', fontsize=18)
    ax_kde.set_xlabel('site-years', fontsize=18)
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
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor='#A9A9A9', alpha=0.4, edgecolor='#808080', linewidth=1)]
    labels = [f'All QC\'d sites ({len(orig_data):,})']
    
    # Add reference line legends (using trimmed data stats)
    trimmed_mean = overall_data.mean()
    trimmed_std = overall_data.std()
    handles.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5))
    labels.append('No change (0%)')
    handles.append(plt.Line2D([0], [0], color='blue', linestyle='-', linewidth=1.5))
    labels.append(f'Median ({overall_data.median():.2f}%)')
    handles.append(plt.Line2D([0], [0], color='green', linestyle=':', linewidth=1.5))
    labels.append(f'Mean ({trimmed_mean:.2f}%)')
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.2, edgecolor='green', linestyle='-.'))
    labels.append(f'±1 Std ({trimmed_std:.2f}%)')
    
    for climate_val in all_climates:
        n_climate = annual_filtered[annual_filtered[climate_col] == climate_val].shape[0]
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=climate_colors[climate_val], alpha=0.8))
        labels.append(f'{climate_val} ({n_climate:,})')
    
    fig.legend(
        handles, labels,
        title=f"Annual ETo QC Effect - Excluding No-Correction Years (n = {n_stations_annual} stations)",
        loc='upper center',
        bbox_to_anchor=(0.35, 1),
        ncol=3,
        fontsize=14,
        title_fontsize=14,
        frameon=False
    )
    
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ETo_QC_PctDiff_Annual_Climate_Violin_Filtered.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Annual ETo QC effect climate plot saved to {PLOT_DIR / 'ETo_QC_PctDiff_Annual_Climate_Violin_Filtered.png'}")


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
    
    # print("\nGenerating climate-based ETo ratio histograms and violin plots...")
    # plot_eto_ratio_climate_histogram_violin(daily_df_climate, annual_df_climate, n_stations)
    
    # # Generate QC effect distribution plots (excluding ratio=1 to highlight QC effect)
    # print("\nGenerating QC effect distribution plots (excluding no-correction records)...")
    # plot_qc_effect_distribution(daily_df_climate, annual_df_climate, exclude_ratio_one=True)
    
    # # Generate QC effect plots by climate zone
    # print("\nGenerating QC effect plots by climate zone...")
    # plot_qc_effect_by_climate(daily_df_climate, annual_df_climate, exclude_ratio_one=True)
    
    # Generate combined histogram + violin plots for QC effect (excluding ratio=1)
    print("\nGenerating combined histogram + climate violin plots for QC effect...")
    plot_qc_effect_combined_histogram_violin(daily_df_climate, annual_df_climate)
    
    print("\nAnalysis complete!")
    
    return daily_df, annual_df


if __name__ == '__main__':
    daily_df, annual_df = main(load_existing=True)
