# This script performs site analysis for GridMET ETo and OpenET actual ET data for each site.
import pandas as pd
import zipfile
from glob import glob


if __name__ == '__main__':
    gridmet_daily_csv_file = '../../Plots/GridMET_Plots/All/GridMET_Daily_All_Station_Data.csv'
    gridmet_monthly_csv_file = '../../Plots/GridMET_Plots/All/GridMET_Monthly_All_Station_Data.csv'
    paired_openet_daily_csv_files = glob('../../Data/paired_flux_OpenET_data/merged_daily*corr*.csv')
    paired_openet_monthly_csv_files = glob('../../Data/paired_flux_OpenET_data/merged_monthly*corr*.csv')
    site_col = 'SITE_ID'
    date_col = 'DATE'
    gridmet_daily_df = pd.read_csv(gridmet_daily_csv_file)
    gridmet_monthly_df = pd.read_csv(gridmet_monthly_csv_file)
    for openet_daily_csv_file in paired_openet_daily_csv_files:
        openet_daily_df = pd.read_csv(openet_daily_csv_file)
        for join_type in ['inner', 'left', 'right']:
            merged_daily_df = pd.merge(
                openet_daily_df, gridmet_daily_df, 
                on=[site_col, date_col], how=join_type
            ).drop_duplicates()
            base_name = openet_daily_csv_file.split('/')[-1]
            prefix = openet_daily_csv_file.split('_')[-1]
            merged_daily_df.to_csv(openet_daily_csv_file.replace(
                base_name, f'openet_gridmet_merged_daily_jtype_{join_type}_{prefix}'), 
                index=False
            )

    for openet_monthly_csv_file in paired_openet_monthly_csv_files:
        openet_monthly_df = pd.read_csv(openet_monthly_csv_file)
        openet_monthly_df[date_col] = openet_monthly_df[date_col].apply(
            lambda x: pd.to_datetime(x).strftime('%Y-%m')
        )
        for join_type in ['inner', 'left', 'right']:
            merged_monthly_df = pd.merge(
                openet_monthly_df, gridmet_monthly_df, 
                on=[site_col, date_col], how=join_type
            ).drop_duplicates()
            base_name = openet_monthly_csv_file.split('/')[-1]
            prefix = openet_monthly_csv_file.split('_')[-1]
            merged_monthly_df.to_csv(openet_monthly_csv_file.replace(
                base_name, f'openet_gridmet_merged_monthly_jtype_{join_type}_{prefix}'), 
                index=False
            )

    merged_files = glob('../../Data/paired_flux_OpenET_data/openet_gridmet_merged*.csv')
    # zip the merged files
    zip_file_name = '../../Data/paired_flux_OpenET_data/merged_gridmet_openet_files.zip'
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_LZMA) as zip_file:
        for file in merged_files:
            zip_file.write(file, arcname=file.split('/')[-1])
    print(f'Merged files zipped into {zip_file_name}')