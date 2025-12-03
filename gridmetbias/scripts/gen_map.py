# Script to generate maps of evapotranspiration observation sites
# This script reads a CSV file containing site information and plots the number of observations, years of data, and average completeness.
# Author: Christian Dunkerly (christian.dunkerly@dri.edu)
# Modified by: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

import pandas as pd
import numpy as np
import geopandas
import re
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib
from matplotlib.lines import Line2D

# Read data first to check actual ranges
df = pd.read_csv("../../Data/openet_ground_station_master_list_cleaned_v4.csv")
df = df.loc[df['included'] == True]

# Check actual data ranges to set appropriate bins
print("Actual data ranges:")
print(f"etr_obs_count: {df['etr_obs_count'].min():.0f} - {df['etr_obs_count'].max():.0f}")
print(f"record_length: {df['record_length'].min():.0f} - {df['record_length'].max():.0f}")
print(f"Average Annual Completeness (%): {df['Average Annual Completeness (%)'].min():.1f} - {df['Average Annual Completeness (%)'].max():.1f}")

# Count how many points are below 3000
below_3000 = (df['etr_obs_count'] < 3000).sum()
print(f"Points below 3000 observations: {below_3000}")

params_dict = {
    'var_name': {0: 'etr_obs_count',
                 1: 'record_length',
                 2: 'Average Annual Completeness (%)'},
    'legend_title': {0: 'Quality-controlled ETo observations',
                     1: 'Years of ETo data',
                     2: 'Average coverage per year (%)'},
    'title': {0: 'Number of ETo observations',
              1: 'Years of ETo data',
              2: 'Average annual completeness (%) across record'},
    'label': {0: '(a)',
              1: '(b)',
              2: '(c)'},
    'cmap': {0: 'prism_r', 1: 'prism_r', 2: 'prism_r'},
    # Use actual minimum values to ensure all data is captured
    'custom_bins': {0: [400, 3000, 6000, 9000, 12000, 15000],  # Start from a round number
                    1: [1, 5, 10, 15, 20, 25],
                    2: [40, 70, 80, 90, 95, 100]}
}

print(f"Using bins for observations: {params_dict['custom_bins'][0]}")

fig, axes = plt.subplots(3, 1, figsize=(25, 36))

geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf_points = geopandas.GeoDataFrame(df[params_dict['var_name'].values()], geometry=geometry, crs="EPSG:4326")
states = geopandas.read_file("../../Plots/Site_Analysis_GridMET/monthly/conus_boundaries.geojson")
contiguous_states = states[~states['NAME'].isin(['Alaska', 'Hawaii', 'Puerto Rico'])]

for i in np.arange(3):
    ax = axes[i]
    plt.rcParams.update({'font.size': 22})
    contiguous_states.plot(ax=ax, color='#e3e3e3', edgecolor='black')
    
    # Create discrete colormap with exactly 5 colors
    base_cmap = matplotlib.colormaps.get_cmap(params_dict['cmap'][i])
    colors = [base_cmap(j/4) for j in range(5)]  # 5 evenly spaced colors
    discrete_cmap = mcolors.ListedColormap(colors)
    
    # Debug: Print classification for first subplot
    if i == 0:
        print(f"\nClassification for subplot {i}:")
        bins = params_dict['custom_bins'][i]
        for j in range(len(bins)-1):
            if j == 0:
                # First bin: anything less than the second bin value
                count = (gdf_points[params_dict['var_name'][i]] < bins[j+1]).sum()
                print(f"Bin < {bins[j+1]:.0f}: {count} points")
            else:
                count = ((gdf_points[params_dict['var_name'][i]] >= bins[j]) & 
                        (gdf_points[params_dict['var_name'][i]] < bins[j+1])).sum()
                if j == len(bins)-2:  # Last bin includes upper bound
                    count = ((gdf_points[params_dict['var_name'][i]] >= bins[j]) & 
                            (gdf_points[params_dict['var_name'][i]] <= bins[j+1])).sum()
                print(f"Bin {bins[j]:.0f} to {bins[j+1]:.0f}: {count} points")
    
    # Create a categorical column for better control over classification
    data_col = gdf_points[params_dict['var_name'][i]]
    bins = params_dict['custom_bins'][i]
    
    # Create categorical labels
    categorical = pd.cut(data_col, bins=bins, labels=False, include_lowest=True)
    gdf_points[f'{params_dict["var_name"][i]}_cat'] = categorical
    
    # Plot with categorical data
    gdf_points.plot(
        ax=ax,
        column=f'{params_dict["var_name"][i]}_cat',
        cmap=discrete_cmap,
        marker='o',
        markersize=80,
        alpha=0.9,
        edgecolors='black',
        linewidth=0.5,
        legend=False,
        categorical=True  # Ensure categorical plotting
    )

    # Create custom legend with all 5 colors and labels
    custom_bins = params_dict['custom_bins'][i]
    new_labels = []
    new_handles = []
    num_intervals = len(custom_bins) - 1
    
    for idx in range(num_intervals):
        start_val = int(custom_bins[idx])
        end_val = int(custom_bins[idx + 1])
        
        if i == 0:  # Observation count - use comma notation and special handling for first bin
            new_text = f'{start_val:,} – {end_val:,}'
        else:  # Years and completeness - no commas needed for smaller numbers
            new_text = f'{start_val} – {end_val}'
        
        new_labels.append(new_text)
        # Create a circular marker (dot) with the corresponding color
        new_handles.append(Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=colors[idx], markersize=15,
                                alpha=0.9))
    
    # Create a new legend with all 5 entries
    ax.legend(new_handles, new_labels, 
             title=params_dict['legend_title'][i],
             loc="lower right",
             fontsize=22,
             title_fontsize=22)

    # Customize the plot
    ax.set_title(params_dict['label'][i], fontdict={'fontsize': '30', 'fontweight': 'bold'})
    ax.set_xlabel('Longitude ($^\\circ$)', fontsize=30)
    ax.set_ylabel('Latitude ($^\\circ$)', fontsize=30)
    ax.set_aspect('equal')

    # Set map limits to focus on the contiguous US
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24, 50)
    
    # Properly set tick label font sizes
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

# Save the figure
plt.tight_layout()
plt.savefig(f'../../Plots/station_map_conus_agweather.png', dpi=600, bbox_inches='tight', facecolor='white')