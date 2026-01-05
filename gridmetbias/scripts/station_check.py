# Check station IDs
# Author: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)
import pandas as pd
from glob import glob

if __name__ == "__main__":
    metadata_station_csv = '../../Data/review_nature_ground_stations_v1.2/metadata_for_publication.csv'
    var_list = [
        'ea',
        'eto',
        'etr',
        'srad',
        'tmax',
        'tmin',
        'u2'
    ]
    
    metadata_df = pd.read_csv(metadata_station_csv)
    metadata_station_ids = metadata_df.Station.unique().tolist()
    for var in var_list:
        var_csv = glob(f'../../Data/Point bias data/{var}_*.csv')[0]
        var_df = pd.read_csv(var_csv)
        var_station_ids = var_df.STATION_ID.unique().tolist()
        missing_stations = set(metadata_station_ids) - set(var_station_ids)
        if len(missing_stations) > 0:
            print(f"Variable: {var} - Missing Stations: {len(missing_stations)}")
            print(missing_stations)
            print(f'Total Stations in {var}: {len(var_station_ids)}')
            print("\n")
        else:
            print(f"Variable: {var} - All stations present.")
