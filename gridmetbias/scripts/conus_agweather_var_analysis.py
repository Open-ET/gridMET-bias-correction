# Script to analyze CONUS-AgWeather variables (Rs, Rso, etc.) pre- and post-QC
# Author: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Define paths
DATA_DIR = Path('../../Data/CONUS-AgWeather_v1/standardized_data')
OUTPUT_DIR = Path('../../Data/Outputs')
PLOT_DIR = Path('../../Plots/CONUS-AgWeather_v1_Var_Stats')

# Create output directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_station_data(station_id):
    """
    Load corrected and original data for a given station.
    
    Args:
        station_id: Station identifier (e.g., '635_NV')
    
    Returns:
        DataFrame with corrected and original values merged
    """
    xlsx_path = DATA_DIR / f'{station_id}_data.xlsx'
    
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Station file not found: {xlsx_path}")
    
    # Read corrected data and delta sheets
    df_corr = pd.read_excel(xlsx_path, sheet_name='Corrected Data')
    df_delta = pd.read_excel(xlsx_path, sheet_name='Delta (Corr - Orig)')
    
    # Ensure Date column is datetime
    df_corr['Date'] = pd.to_datetime(df_corr['Date'])
    df_delta['Date'] = pd.to_datetime(df_delta['Date'])
    
    # Merge on Date
    merged = pd.merge(
        df_corr[['Date', 'Rs (w/m2)', 'Rso (w/m2)', 'Optimized TR Rs (w/m2)']],
        df_delta[['Date', 'Rs (w/m2)', 'Rso (w/m2)']],
        on='Date',
        suffixes=('_corr', '_delta')
    )
    
    # Calculate original values: Original = Corrected - Delta
    merged['Rs_orig'] = merged['Rs (w/m2)_corr'] - merged['Rs (w/m2)_delta']
    merged['Rs_corr'] = merged['Rs (w/m2)_corr']
    merged['Rso_orig'] = merged['Rso (w/m2)_corr'] - merged['Rso (w/m2)_delta']
    merged['Rso_corr'] = merged['Rso (w/m2)_corr']
    merged['Optimized_TR_Rs'] = merged['Optimized TR Rs (w/m2)']
    
    # Calculate percent difference: (Corrected - Original) / Original * 100
    merged['Rs_pct_diff'] = np.where(
        merged['Rs_orig'] != 0,
        (merged['Rs_corr'] - merged['Rs_orig']) / merged['Rs_orig'] * 100,
        np.nan
    )
    merged['Rso_pct_diff'] = np.where(
        merged['Rso_orig'] != 0,
        (merged['Rso_corr'] - merged['Rso_orig']) / merged['Rso_orig'] * 100,
        np.nan
    )
    
    return merged


def plot_station_rs_comparison(station_id, save_plot=True):
    """
    Create a figure with three subplots for a given station:
    (a) Original Rs and Rso at daily timestep
    (b) Corrected Rs and Rso at daily timestep
    (c) Percent difference between corrected and original
    
    Args:
        station_id: Station identifier (e.g., '635_NV')
        save_plot: Whether to save the plot to file
    """
    # Load data
    df = load_station_data(station_id)
    
    # Filter to where both Rs and Rso have valid data (overlap in time)
    df = df.dropna(subset=['Rs_orig', 'Rs_corr', 'Rso_orig', 'Rso_corr'])
    
    # Create figure with 3 subplots (not sharing x-axis so each can have year labels)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    plt.rcParams.update({'font.size': 16})
    
    # Color-blind friendly colors (Nature Publishing Group palette)
    color_rs = '#3C5488'   # Dark blue (NPG) - high contrast
    color_rso = '#E64B35'  # Vermillion/red (NPG) - high contrast
    
    # --- (a) Original Rs and Rso ---
    ax1 = axes[0]
    ax1.plot(df['Date'], df['Rs_orig'], color=color_rs, linewidth=0.8, alpha=0.8, label='Rs (Original)')
    ax1.plot(df['Date'], df['Rso_orig'], color=color_rso, linewidth=0.8, alpha=0.8, label='Rso (Original)')
    ax1.set_ylabel('Solar Radiation (W m⁻²)', fontsize=16)
    ax1.text(-0.02, 1.03, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=16, frameon=False)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # --- (b) Corrected Rs and Rso ---
    ax2 = axes[1]
    ax2.plot(df['Date'], df['Rs_corr'], color=color_rs, linewidth=0.8, alpha=0.8, label='Rs (Corrected)')
    ax2.plot(df['Date'], df['Rso_corr'], color=color_rso, linewidth=0.8, alpha=0.8, label='Rso (Corrected)')
    ax2.set_ylabel('Solar Radiation (W m⁻²)', fontsize=16)
    ax2.text(-0.02, 1.03, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=16, frameon=False)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # --- (c) Percent difference ---
    ax3 = axes[2]
    ax3.plot(df['Date'], df['Rs_pct_diff'], color=color_rs, linewidth=0.8, alpha=0.8, label='Rs % Difference')
    ax3.plot(df['Date'], df['Rso_pct_diff'], color=color_rso, linewidth=0.8, alpha=0.8, label='Rso % Difference')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Percent Difference (%)', fontsize=16)
    ax3.set_xlabel('Date', fontsize=16)
    ax3.text(-0.02, 1.03, '(c)', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')
    ax3.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=16, frameon=False)
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis with year tick marks for ALL subplots
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.set_xlim(df['Date'].min(), df['Date'].max())
        ax.tick_params(axis='both', labelsize=16)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add vertical lines at year boundaries for all subplots
    years = df['Date'].dt.year.unique()
    for ax in axes:
        for year in years[1:]:  # Skip first year
            year_start = pd.Timestamp(f'{year}-01-01')
            if year_start >= df['Date'].min() and year_start <= df['Date'].max():
                ax.axvline(x=year_start, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    if save_plot:
        output_path = PLOT_DIR / f'{station_id}_Rs_Rso_comparison.png'
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    # Print summary statistics
    print(f"\n--- Summary Statistics for Station {station_id} ---")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Number of observations: {len(df):,}")
    print(f"\nRs Statistics:")
    print(f"  Original:  Mean = {df['Rs_orig'].mean():.2f}, Median = {df['Rs_orig'].median():.2f}")
    print(f"  Corrected: Mean = {df['Rs_corr'].mean():.2f}, Median = {df['Rs_corr'].median():.2f}")
    print(f"  % Diff:    Mean = {df['Rs_pct_diff'].mean():.2f}%, Median = {df['Rs_pct_diff'].median():.2f}%")
    print(f"\nRso Statistics:")
    print(f"  Original:  Mean = {df['Rso_orig'].mean():.2f}, Median = {df['Rso_orig'].median():.2f}")
    print(f"  Corrected: Mean = {df['Rso_corr'].mean():.2f}, Median = {df['Rso_corr'].median():.2f}")
    print(f"  % Diff:    Mean = {df['Rso_pct_diff'].mean():.2f}%, Median = {df['Rso_pct_diff'].median():.2f}%")
    
    return df


def load_station_rh_data(station_id):
    """
    Load RHMax and RHMin corrected and original data for a given station.
    
    Args:
        station_id: Station identifier (e.g., '1069_MT')
    
    Returns:
        DataFrame with corrected and original RH values merged
    """
    xlsx_path = DATA_DIR / f'{station_id}_data.xlsx'
    
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Station file not found: {xlsx_path}")
    
    # Read corrected data and delta sheets
    df_corr = pd.read_excel(xlsx_path, sheet_name='Corrected Data')
    df_delta = pd.read_excel(xlsx_path, sheet_name='Delta (Corr - Orig)')
    
    # Ensure Date column is datetime
    df_corr['Date'] = pd.to_datetime(df_corr['Date'])
    df_delta['Date'] = pd.to_datetime(df_delta['Date'])
    
    # Merge on Date for RH columns
    merged = pd.merge(
        df_corr[['Date', 'RHMax (%)', 'RHMin (%)']],
        df_delta[['Date', 'RHMax (%)', 'RHMin (%)']],
        on='Date',
        suffixes=('_corr', '_delta')
    )
    
    # Calculate original values: Original = Corrected - Delta
    merged['RHMax_orig'] = merged['RHMax (%)_corr'] - merged['RHMax (%)_delta']
    merged['RHMax_corr'] = merged['RHMax (%)_corr']
    merged['RHMin_orig'] = merged['RHMin (%)_corr'] - merged['RHMin (%)_delta']
    merged['RHMin_corr'] = merged['RHMin (%)_corr']
    
    # Calculate percent difference: (Corrected - Original) / Original * 100
    merged['RHMax_pct_diff'] = np.where(
        merged['RHMax_orig'] != 0,
        (merged['RHMax_corr'] - merged['RHMax_orig']) / merged['RHMax_orig'] * 100,
        np.nan
    )
    merged['RHMin_pct_diff'] = np.where(
        merged['RHMin_orig'] != 0,
        (merged['RHMin_corr'] - merged['RHMin_orig']) / merged['RHMin_orig'] * 100,
        np.nan
    )
    
    return merged


def plot_station_rh_comparison(station_id, save_plot=True):
    """
    Create a figure with three subplots for a given station:
    (a) Original RHMax and RHMin at daily timestep
    (b) Corrected RHMax and RHMin at daily timestep
    (c) Percent difference between corrected and original
    
    Args:
        station_id: Station identifier (e.g., '1069_MT')
        save_plot: Whether to save the plot to file
    """
    # Load data
    df = load_station_rh_data(station_id)
    
    # Filter to where both RHMax and RHMin have valid data (overlap in time)
    df = df.dropna(subset=['RHMax_orig', 'RHMax_corr', 'RHMin_orig', 'RHMin_corr'])
    
    # Create figure with 3 subplots (not sharing x-axis so each can have year labels)
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    plt.rcParams.update({'font.size': 16})
    
    # Color-blind friendly colors (Nature Publishing Group palette)
    color_rhmax = '#3C5488'   # Dark blue (NPG)
    color_rhmin = '#E64B35'   # Vermillion/coral (NPG)
    
    # --- (a) Original RHMax and RHMin ---
    ax1 = axes[0]
    ax1.plot(df['Date'], df['RHMax_orig'], color=color_rhmax, linewidth=0.8, alpha=0.8, label='RHMax (Original)')
    ax1.plot(df['Date'], df['RHMin_orig'], color=color_rhmin, linewidth=0.8, alpha=0.8, label='RHMin (Original)')
    ax1.set_ylabel('Relative Humidity (%)', fontsize=16)
    ax1.text(-0.02, 1.03, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=16, frameon=False)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # --- (b) Corrected RHMax and RHMin ---
    ax2 = axes[1]
    ax2.plot(df['Date'], df['RHMax_corr'], color=color_rhmax, linewidth=0.8, alpha=0.8, label='RHMax (Corrected)')
    ax2.plot(df['Date'], df['RHMin_corr'], color=color_rhmin, linewidth=0.8, alpha=0.8, label='RHMin (Corrected)')
    ax2.set_ylabel('Relative Humidity (%)', fontsize=16)
    ax2.text(-0.02, 1.03, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=16, frameon=False)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # --- (c) Percent difference ---
    ax3 = axes[2]
    ax3.plot(df['Date'], df['RHMax_pct_diff'], color=color_rhmax, linewidth=0.8, alpha=0.8, label='RHMax % Difference')
    ax3.plot(df['Date'], df['RHMin_pct_diff'], color=color_rhmin, linewidth=0.8, alpha=0.8, label='RHMin % Difference')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Percent Difference (%)', fontsize=16)
    ax3.set_xlabel('Date', fontsize=16)
    ax3.text(-0.02, 1.03, '(c)', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='bottom', ha='right')
    ax3.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=16, frameon=False)
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis with year tick marks for ALL subplots
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.set_xlim(df['Date'].min(), df['Date'].max())
        ax.tick_params(axis='both', labelsize=16)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add vertical lines at year boundaries for all subplots
    years = df['Date'].dt.year.unique()
    for ax in axes:
        for year in years[1:]:  # Skip first year
            year_start = pd.Timestamp(f'{year}-01-01')
            if year_start >= df['Date'].min() and year_start <= df['Date'].max():
                ax.axvline(x=year_start, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    if save_plot:
        output_path = PLOT_DIR / f'{station_id}_RHMax_RHMin_comparison.png'
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    # Print summary statistics
    print(f"\n--- Summary Statistics for Station {station_id} ---")
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Number of observations: {len(df):,}")
    print(f"\nRHMax Statistics:")
    print(f"  Original:  Mean = {df['RHMax_orig'].mean():.2f}%, Median = {df['RHMax_orig'].median():.2f}%")
    print(f"  Corrected: Mean = {df['RHMax_corr'].mean():.2f}%, Median = {df['RHMax_corr'].median():.2f}%")
    print(f"  % Diff:    Mean = {df['RHMax_pct_diff'].mean():.2f}%, Median = {df['RHMax_pct_diff'].median():.2f}%")
    print(f"\nRHMin Statistics:")
    print(f"  Original:  Mean = {df['RHMin_orig'].mean():.2f}%, Median = {df['RHMin_orig'].median():.2f}%")
    print(f"  Corrected: Mean = {df['RHMin_corr'].mean():.2f}%, Median = {df['RHMin_corr'].median():.2f}%")
    print(f"  % Diff:    Mean = {df['RHMin_pct_diff'].mean():.2f}%, Median = {df['RHMin_pct_diff'].median():.2f}%")
    
    return df


def main(station_id='635_NV', variable='rs'):
    """
    Main function to run variable analysis for a given station.
    
    Args:
        station_id: Station identifier (e.g., '635_NV', '1069_MT')
        variable: Variable type to analyze ('rs' for Rs/Rso, 'rh' for RHMax/RHMin)
    """
    if variable.lower() == 'rs':
        print(f"Analyzing Rs and Rso data for station {station_id}...")
        df = plot_station_rs_comparison(station_id)
    elif variable.lower() == 'rh':
        print(f"Analyzing RHMax and RHMin data for station {station_id}...")
        df = plot_station_rh_comparison(station_id)
    else:
        raise ValueError(f"Unknown variable type: {variable}. Use 'rs' or 'rh'.")
    
    print("\nAnalysis complete!")
    return df


if __name__ == '__main__':
    # Example: Rs/Rso analysis for station 635_NV
    df = main(station_id='635_NV', variable='rs')
    
    # Example: RHMax/RHMin analysis for station 1069_MT
    df = main(station_id='1069_MT', variable='rh')
