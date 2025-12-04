"""
Main script for calculating biases in each weather station variable and making box plots
and CSV files with statistics grouped by east/west U.S., and climate zones. Run this script
after running "data_formatting.py"
Author: Dr. John Volk (john.volk@dri.edu)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


plot_variable = "annual_mean"  # desired variable annual_mean, "summer_mean", "growseason_mean", etc.

time_period_labels = {
    "summer_mean": "Summer (JJA)",
    "Jan_mean": "January",
    "Feb_mean": "February",
    "Mar_mean": "March",
    "Apr_mean": "April",
    "May_mean": "May",
    "Jun_mean": "June",
    "Jul_mean": "July",
    "Aug_mean": "August",
    "Sep_mean": "September",
    "Oct_mean": "October",
    "Nov_mean": "November",
    "Dec_mean": "December",
    "growseason_mean": "Growing Season (AMJJASO)",
    "annual_mean": "Annual"
}

# names of variables
var_names = ["eto_mm", "u2_ms", "etr_mm", "ea_kpa", "srad_wm2", "tmin_c", "tmax_c"]

# climate zone renaming
lumped_zones = {
    "Bsk": "Arid Steppe (Bsk + Bsh)",
    "BSh": "Arid Steppe (Bsk + Bsh)",
    "BWh": "Desert (Bwh + Bwk)",
    "Bwk": "Desert (Bwh + Bwk)",
    "Cfa": "Humid Subtropical (Cfa)",
    "Csa": "Mediterranean (Csa + Csb)",
    "Csb": "Mediterranean (Csa + Csb)",
    "Dfa": "Humid Continental (Dfa + Dfb)",
    "Dfb": "Humid Continental (Dfa + Dfb)"
}

# renaming variables
simplified_names = {
    "eto_mm": "ETo",
    "u2_ms": "u2",
    "etr_mm": "ETr",
    "ea_kpa": "ea",
    "srad_wm2": "srad",
    "tmin_c": "tmin",
    "tmax_c": "tmax"
}

def generate_boxplots(var_name):
    data_path = f"../Data/Point bias data/Climate/{var_name}_merged_with_climate.csv"
    merged_data = pd.read_csv(data_path)

    if var_name in ["tmin_c", "tmax_c"]:
        merged_data[plot_variable] = -merged_data[plot_variable]  # Flip the sign for tmin and tmax
    else:
        merged_data[plot_variable] = 1 / merged_data[plot_variable]  # gridMET bias (grid/station)

    plots_folder = f"../Plots/Boxplots/{time_period_labels.get(plot_variable)}"
    os.makedirs(plots_folder, exist_ok=True)

    # remap lumped zones
    merged_data['Lumped_Zone'] = merged_data['Climate_Abbreviation'].map(lumped_zones)
    valid_data = merged_data[merged_data['Lumped_Zone'].notna()]
    # east/west sites
    valid_data['Region'] = valid_data['STATION_LON'].apply(lambda x: 'West' if x < -100 else 'East')
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=300, sharex=False)
    plt.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05, left=0.1, right=0.95)
    time_period_label = time_period_labels.get(plot_variable, plot_variable)

    # Boxplot 1: All sites
    sns.boxplot(x=valid_data[plot_variable], ax=axes[0], palette="Set2")
    axes[0].set_title(f"{simplified_names[var_name]} bias ({time_period_label}) Across All Sites", fontsize=16, pad=10)
    axes[0].set_xlabel(f"{simplified_names[var_name]} bias", fontsize=12)

    # Boxplot 2: East vs. West of 100th meridian
    sns.boxplot(x='Region', y=plot_variable, data=valid_data, ax=axes[1], palette="Set3")
    axes[1].set_title(f"{simplified_names[var_name]} bias ({time_period_label}) by Region (East/West of 100th Meridian)", fontsize=16, pad=10)
    axes[1].set_xlabel("")
    axes[1].set_ylabel(f"{simplified_names[var_name]} bias", fontsize=12)

    # Boxplot 3: By Lumped Climate Zone
    sns.boxplot(x='Lumped_Zone', y=plot_variable, data=valid_data, ax=axes[2], palette="Pastel2")
    axes[2].set_title(f"{simplified_names[var_name]} bias ({time_period_label}) by Lumped Climate Zone", fontsize=16, pad=10)
    axes[2].set_xlabel("")
    axes[2].set_ylabel(f"{simplified_names[var_name]} bias", fontsize=12)
    axes[2].tick_params(axis='x', rotation=30, labelsize=10)  # Rotate for better readability

    plot_output_path = os.path.join(plots_folder, f"{simplified_names[var_name]}_Bias_Boxplots_{plot_variable}.png")
    plt.tight_layout()
    plt.savefig(plot_output_path, format="png", dpi=300)


    # Compute and save summary statistics for each grouping into one CSV file
    count_column = plot_variable.replace("_mean", "_count")
    stats_list = []
    
    def summarize(df_sub, name):
        vals = df_sub[plot_variable].dropna()
        pts  = df_sub[count_column].fillna(0).astype(int)
        return {
            'Group':       name,
            'n stations':  len(df_sub),
            'n days':    int(pts.sum()),
            'min':         vals.min(),
            'max':         vals.max(),
            'median':      vals.median(),
            'q1':          vals.quantile(0.25),      
            'q3':          vals.quantile(0.75),      
            'mean':        vals.mean(),
            'std':         vals.std()
        }
	# All sites
    stats_list.append(summarize(valid_data, 'All Sites'))

    # East/west
    for region, df_region in valid_data.groupby('Region'):
        stats_list.append(summarize(df_region, region))

    # By Lumped Climate Zone
    for zone, df_zone in valid_data.groupby('Lumped_Zone'):
        stats_list.append(summarize(df_zone, zone))
        

    stats_df = pd.DataFrame(stats_list).round(2)
    stats_output_path = os.path.join(plots_folder, f"{simplified_names[var_name]}_Bias_Stats_{plot_variable}.csv")
    stats_df.to_csv(stats_output_path, index=False)
    plt.close(fig)

if __name__ == '__main__':
    for var in var_names:
        generate_boxplots(var)


