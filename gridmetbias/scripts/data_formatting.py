"""
Perform join of station ETo bias results with koppen climate zone codes, add 
climate zone information.

Author: Dr. John Volk (john.volk@dri.edu)
Modified by: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)
"""
import pandas as pd
import geopandas as gpd

# all weather variables
var_names = ["eto_mm", "u2_ms", "etr_mm", "ea_kpa", "srad_wm2", "tmin_c", "tmax_c"]

def merge_data(var_name):
	if var_name in ["eto_mm", "etr_mm"]:
		bias_path = f"../Data/Point bias data/{var_name}_summary_comp_all_yrs.csv"
		geojson_path = f"../Data/Point bias data/Climate/{var_name}_summary_comp_all_yrs_climate.geojson"
	else:
		bias_path = f"../Data/Point bias data/{var_name}_summary_comp_merged.csv"
		geojson_path = f"../Data/Point bias data/Climate/{var_name}_summary_comp_merged_climate.geojson"
	koppen_info_path = "../Data/koppen_ID_info.csv"
	
	bias_data = pd.read_csv(bias_path)
	climate_data = gpd.read_file(geojson_path)
	koppen_info = pd.read_csv(koppen_info_path)

	# join on 'gridcode' column (koppen zone ID)
	merged_data = bias_data.merge(
		climate_data[['STATION_ID', 'gridcode']], on='STATION_ID', how='inner'
	)
	# add climate descriptors
	climate_zone_info = koppen_info[['gridcode', 'Code', 'Description']]
	merged_data = merged_data.merge(climate_zone_info, on='gridcode', how='left')
	merged_data.rename(
		columns={'Code': 'Climate_Abbreviation', 'Description': 'Climate_Zone'}, inplace=True
	)
	# Save
	output_path = f"../Data/Point bias data/Climate/{var_name}_merged_with_climate.csv"
	merged_data.to_csv(output_path, index=False)

if __name__ == '__main__':
	for v in var_names:
		merge_data(v)

