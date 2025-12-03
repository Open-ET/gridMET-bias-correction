"""
Main script for running the correlation analysis on the GridMET bias ratio data.
Author: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)
"""

import calendar
import os
from glob import glob
import biaslibs as blibs


if __name__ == '__main__':
    # Define the file paths
    csv_files = glob('../Data/Point bias data/*.csv')
    csv_files = [f for f in csv_files if 'climate' not in f.split(os.sep)[-1]]
    et_files = [f for f in csv_files if f.split('/')[-1].startswith('et')]
    other_files = [f for f in csv_files if f not in et_files]
    month_list = [calendar.month_abbr[m] for m in range(1, 13)]
    month_col_pattern = f"^({'|'.join(month_list)})_mean$"
    plot_dir = '../Plots/'
    climate_shp = '../Data/climateClass_poly_diss/climateClass_poly_diss.shp'
    gcloud_project = 'ee-grid-obs-comp'
    daily_station_file_dir = '../Data/flux_ET_dataset/daily_data_files/'
    monthly_station_file_dir = '../Data/flux_ET_dataset/monthly_data_files/'
    gridmet_csv = '../Data/flux_data/openet_reference_et_summary_all_sites_bias_corr_paper.csv'
    metadata_xlsx = '../Data/flux_ET_dataset/station_metadata.xlsx'
    print('Working on the correlation plots...')
    for all_pval in [True, False]:
        blibs.plot_bias_corr_matrix_all(
            et_files, other_files, 
            month_col_pattern, 
            plot_dir,
            show_all_pvalues=all_pval,
            annot_pvalues=False
        )
        blibs.plot_bias_corr_matrix_lon(
            et_files, other_files, 
            month_col_pattern, 
            plot_dir,
            show_all_pvalues=all_pval,
            annot_pvalues=False
        )
        blibs.plot_bias_corr_matrix_climate(
            et_files, other_files, 
            climate_shp,
            month_col_pattern,
            plot_dir,
            show_all_pvalues=all_pval,
            annot_pvalues=False
        )

    print('gridMET bias analysis...')
    blibs.gridmet_bias_comp_analysis(
        gridmet_csv, metadata_xlsx, 
        daily_station_file_dir, 
        monthly_station_file_dir,
        plot_dir
    )
    print('Working on the bias distributions...')
    flux_et_files = ["../Plots/GridMET_Plots/All/GridMET_Monthly_All_Station_Data.csv"]
    for season_col in ['summer_mean']: #['annual_mean', 'growseason_mean', 'summer_mean']:
        print(f'\nWorking with agricultural weather stations using {season_col} data...')
        blibs.plot_irr_crop_bias_distributions(
            et_files, other_files, 
            plot_dir,
            season_col=season_col,
            verbose=True
        )
        # blibs.plot_ag_bias_distributions(
        #     et_files, other_files, 
        #     plot_dir,
        #     season_col=season_col,
        #     num_ag_cats=3
        # )
        print(f'\nWorking with flux stations using {season_col} data...')
        blibs.plot_irr_crop_bias_distributions(
            flux_et_files, other_files=[], 
            plot_dir='../Plots/Flux/',
            season_col=season_col,
            verbose=True,
            flux_et=True
        )
    