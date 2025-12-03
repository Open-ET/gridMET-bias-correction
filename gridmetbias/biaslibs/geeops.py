"""
Contains functons related to GEE operations and data extraction.
Author: Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)
"""

import ee
import numpy as np
import os
import geopandas as gpd
import pandas as pd


def categorize_irr_ag_fraction(
        bias_df: pd.DataFrame, 
        num_ag_cats: int = 3, 
        buffer_applied: bool = True
) -> pd.DataFrame:
    """
    Categorize the irrigation and agriculture fractions into low, medium, and 
    high categories.

    Args:
        bias_df (pd.DataFrame): The DataFrame containing the bias ratios.
        num_ag_cats (int): The number of categories for agriculture. Default is 3.
        buffer_applied (bool): Flag indicating if buffer was applied during GEE extraction.

    Returns:
        The DataFrame with an additional column for the categories.
    """
    
    irr_col = 'Irrigation Fraction'
    irrmapper_col = 'IrrMapper Fraction'
    lanid_col = 'LANID Fraction'
    nlcd_ag_col = 'NLCD_Ag Fraction'
    cdl_ag_col = 'CDL_Ag Fraction'
    op_cols = [irr_col, nlcd_ag_col, cdl_ag_col]
    new_cols = ['Irrigation', 'NLCD Agriculture', 'CDL Agriculture']
    categories = []
    lt, ht = 0, 0
    for op_col, new_col in zip(op_cols, new_cols):
        if op_col == irr_col:
            bias_df[op_col] = np.where(
                bias_df[irrmapper_col] == -9999,
                bias_df[lanid_col],
                bias_df[irrmapper_col]
            )
        bias_df[op_col] = bias_df[op_col].replace(-9999, 0)
        bias_df[op_col] *= 100
    
        col_check = op_col != cdl_ag_col if not buffer_applied else op_col == irr_col
        if col_check:
            if buffer_applied:
                lt = 25
                ht = 75
            else:
                lt = int(np.ceil(bias_df[op_col].quantile(0.33)))
                lt_round = int(round(lt, -1))
                lt = 5 if lt_round == 0 else lt_round
                ht = int(bias_df[op_col].quantile(0.67).round(-1))

            if num_ag_cats == 3:
                # Define the categories
                categories = [
                    f'low, < {lt} %', f'medium, [{lt}-{ht}] %', f'high, > {ht} %'
                ]
                bins = [-float('inf'), lt, ht, float('inf')]
            else:
                # define two categories only
                lt = 20
                categories = [f'low, < {lt} %', f'high, > {lt} %']
                bins = [-float('inf'), lt, float('inf')]
        # Create a new column for the categories using pd.cut
        bias_df[new_col] = pd.cut(
            bias_df[op_col], 
            bins=bins, 
            labels=categories, 
            include_lowest=True
        )
    bias_df = bias_df.reset_index(drop=True)
    return bias_df


def fix_cdl_classes(bias_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the CDL classes in the bias DataFrame.

    Args:
        bias_df: The DataFrame containing the bias ratios.

    Returns:
        The DataFrame with the fixed CDL classes.
    """
    cdl_um_col = 'CDL Unmasked'
    cdl_irrmapper_col = 'CDL IrrMapper'
    cdl_lanid_col = 'CDL LANID'
    cdl_col = 'CDL'
    cropland_classes = {
        1: "Corn",
        2: "Cotton",
        5: "Soybeans",
        22: "Wheat",
        23: "Wheat",
        24: "Wheat",
        26: "Soybeans",
        36: "Alfalfa",
        225: "Corn",
        226: "Corn",
        230: "Wheat",
        232: "Cotton",
        234: "Wheat",
        236: "Wheat",
        237: "Corn",
        238: "Cotton",
        239: "Cotton",
        240: "Soybeans",
        241: "Corn",
        254: "Soybeans"
    }

    # Set grassland/pasture (176) classes to alfalfa (36) for irrigated areas
    bias_df[cdl_irrmapper_col] = bias_df[cdl_irrmapper_col].replace(176, 36)
    bias_df[cdl_lanid_col] = bias_df[cdl_lanid_col].replace(176, 36)
    bias_df[cdl_col] = np.where(
        bias_df[cdl_irrmapper_col] == -9999,
        np.where(
            bias_df[cdl_lanid_col] == -9999, 
            bias_df[cdl_um_col], 
            bias_df[cdl_lanid_col]
        ),
        bias_df[cdl_irrmapper_col]
    )
    bias_df[cdl_col] = bias_df[cdl_col].replace(-9999, np.nan)
    bias_df = bias_df.dropna().reset_index(drop=True).copy()
    crop_type_col = 'Crop Type'
    bias_df[crop_type_col] = bias_df[cdl_col].apply(
        lambda x: cropland_classes.get(x, 'Other')
    )
    return bias_df


def get_irr_crop_data(
        input_bias_vector_file: str,
        output_bias_vector_file: str,
        gcloud_project: str = 'ee-grid-obs-comp',
        start_year_col: str = 'start_year',
        end_year_col: str = 'end_year',
        lat_col: str = 'STATION_LAT',
        lon_col: str = 'STATION_LON',
        station_col: str = 'STATION_ID',
        num_ag_cats: int = 3,
        verbose: bool = False,
        buffer_size: int = 1500
) -> pd.DataFrame:
    """
    Extract irrigation density (IrrMapper and LANID) and crop type (CDL) data
    at the nativate gridMET resolution (~4 km) from Google Earth Engine based on 
    the given bias vector file locations.
    
    Args:
        input_bias_vector_file: The file path to the bias vector file.
        output_bias_vector_file: The name of the output file.
        gcloud_project: The Google Cloud project ID.
        start_year_col: The column name for the start year in the bias vector 
        file.
        end_year_col: The column name for the end year in the bias vector file.
        lat_col: The column name for the latitude in the bias vector file.
        lon_col: The column name for the longitude in the bias vector file.
        station_col: The column name for the station ID in the bias vector file.
        num_ag_cats: The number of categories for agriculture. Default is 3.
        verbose: A boolean flag to print the data extraction process.
        buffer_size: The size of the buffer around the station point. If 0, then no buffer is applied.

    Returns:
        The updated bias dataframe with IrrMapper, LANID, and CDL data.   
    """
    use_buffer = buffer_size > 0
    gee_dir = 'GEE_Data_Buffer/' if use_buffer else 'GEE_Data/'
    output_dir = os.path.dirname(output_bias_vector_file) + '/' + gee_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_vector_name = output_bias_vector_file.split(os.sep)[-1]
    prefix = output_vector_name.split('.')[0]
    output_bias_vector_file = f'{output_dir}{output_vector_name}'
    output_csv_file = f'{output_dir}{prefix}_gee_data.csv'
    file_exists = os.path.exists(output_bias_vector_file)
    # file_exists = False  # disable file check for debugging purposes
    if not file_exists:
        ee.Initialize(
            project=gcloud_project,
            opt_url='https://earthengine-highvolume.googleapis.com'
        )    
        irr_mapper_ic = ee.ImageCollection('UMT/Climate/IrrMapper_RF/v1_2')    
        lanidv2_img_list = [
            'users/xyhuwmir4/LANID_postCls/LANID_v2',
            'users/xyhuwmir/LANID/update/LANID2018-2020'
        ]
        landidv2_img_all = ee.Image(lanidv2_img_list[0]) \
                        .addBands(ee.Image(lanidv2_img_list[1]))
        lanid_valid_years = range(1997, 2021)
        cdl_ic = ee.ImageCollection('USDA/NASS/CDL')
        nlcd_ic = ee.ImageCollection(
            'projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER'
        )
        gridmet_scale = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET') \
                        .first().projection().nominalScale().getInfo()
        irr_mapper_proj = irr_mapper_ic.first().projection()
        cdl_proj = cdl_ic.first().projection()
        nlcd_proj = nlcd_ic.first().projection()

        # Read the bias vector file
        bias_vector = gpd.read_file(input_bias_vector_file).sort_values(by=[station_col])
        max_pixels = 65536
        no_data = -9999
        for idx, row in bias_vector.iterrows():
            station_id = row[station_col]
            if station_id in ['124_CA', '139_CA']:  # CIMIS Mexico stations
                # set all fractions to 1
                bias_vector.loc[idx, 'NLCD_Ag Count'] = no_data
                bias_vector.loc[idx, 'CDL_Ag Count'] = no_data
                bias_vector.loc[idx, 'IrrMapper Count'] = no_data
                bias_vector.loc[idx, 'LANID Count'] = no_data
                bias_vector.loc[idx, 'CDL Unmasked'] = 36 # alfalfa
                bias_vector.loc[idx, 'CDL IrrMapper'] = 36 # alfalfa
                bias_vector.loc[idx, 'CDL LANID'] = 36 # alfalfa
                bias_vector.loc[idx, 'IrrMapper Fraction'] = 1
                bias_vector.loc[idx, 'LANID Fraction'] = 1
                bias_vector.loc[idx, 'NLCD_Ag Fraction'] = 1
                bias_vector.loc[idx, 'CDL_Ag Fraction'] = 1
                if verbose:
                    print(f'\nStation: {station_id} (CIMIS Mexico station, all fractions set to 1)')
                continue
            try:
                start_year = int(row[start_year_col])
                end_year = int(row[end_year_col])
                nlcd_start_year = start_year
                nlcd_end_year = end_year
                if nlcd_end_year < 1985:
                    nlcd_end_year = 1985
                nlcd_start_year = f'{nlcd_start_year}-01-01'
                nlcd_end_year = f'{nlcd_end_year}-12-31'
                nlcd_year = nlcd_ic.filterDate(
                    nlcd_start_year, nlcd_end_year
                ).mode()
                nlcd_mask = nlcd_year.eq(82)
                nlcd_ag = nlcd_year.updateMask(nlcd_mask).rename('NLCD_Ag')

                irrmapper_start_year = start_year
                irrmapper_end_year = end_year
                if irrmapper_end_year < 1986:
                    irrmapper_end_year = 1986
                irrmapper_start_year = f'{irrmapper_start_year}-01-01'
                irrmapper_end_year = f'{irrmapper_end_year}-12-31'
                
                # Extract IrrMapper data
                irr_mapper_max = irr_mapper_ic \
                    .filterDate(irrmapper_start_year, irrmapper_end_year) \
                    .select('classification') \
                    .mosaic()
                
                mask = irr_mapper_max.eq(0)
                irr_mapper_max = irr_mapper_max.updateMask(mask).remap([0], [1])
                irr_mapper_max = irr_mapper_max.rename('IrrMapper')

                # Extract LANID data
                lanid_bands = [ 
                    f'irMap{str(y)[-2:]}' for y in range(
                        start_year, end_year + 1
                    ) if y in lanid_valid_years
                ]
                if len(lanid_bands) == 0:
                    lanid_bands = [
                        f'irMap{str(y)[-2:]}' for y in lanid_valid_years
                    ]
                lanid_img_max = landidv2_img_all.select(lanid_bands[0])
                for lanid_band in lanid_bands[1:]:
                    lanid_img_max = lanid_img_max.max(
                        landidv2_img_all.select(lanid_band)
                    )
                    lanid_img_max = lanid_img_max.setDefaultProjection(
                        crs=lanid_img_max.projection(), 
                        scale=30
                    )
                mask = lanid_img_max.eq(1)
                lanid_img_max = lanid_img_max.updateMask(mask).rename('LANID')

                # Extract CDL data
                cdl_start_year = start_year
                cdl_end_year = end_year
                if cdl_end_year < 2008:
                    cdl_end_year = 2008
                cdl_start_year = f'{cdl_start_year}-01-01'
                cdl_end_year = f'{cdl_end_year}-12-31'
                cdl_year = cdl_ic.filterDate(cdl_start_year, cdl_end_year) \
                            .select('cropland') \
                            .mode() \
                            .setDefaultProjection(
                                crs=cdl_proj, 
                                scale=30
                            )
                cdl_irrmapper = cdl_year.multiply(irr_mapper_max)
                cdl_lanid = cdl_year.multiply(lanid_img_max)
                cdl_mask_1 = cdl_year.gte(1).And(cdl_year.lte(59))
                cdl_mask_2 = cdl_year.gte(66).And(cdl_year.lte(77))
                cdl_mask_3 = cdl_year.gte(204)
                cdl_mask_4 = cdl_year.eq(92)
                cdl_mask = cdl_mask_1.Or(cdl_mask_2) \
                            .Or(cdl_mask_3).Or(cdl_mask_4)
                cdl_ag = cdl_year.updateMask(cdl_mask).rename('CDL_Ag')

                if not use_buffer:
                    # Reduce NLCD to gridMET resolution and sum up ag pixels
                    target_scale = gridmet_scale
                    nlcd_ag_gridmet = nlcd_ag.setDefaultProjection(
                        crs=nlcd_proj, 
                        scale=30
                        ).reduceResolution(
                        reducer=ee.Reducer.count(),
                        bestEffort=True,
                        maxPixels=max_pixels
                    ).reproject(cdl_proj, scale=gridmet_scale)

                    # Reduce CDL to gridMET resolution and sum up ag pixels
                    cdl_ag_gridmet = cdl_ag.reduceResolution(
                        reducer=ee.Reducer.count(),
                        bestEffort=True,
                        maxPixels=max_pixels
                    ).reproject(cdl_proj, scale=gridmet_scale)
                    
                    # Reduce IrrMapper to gridMET resolution and sum up irrigated pixels
                    irr_mapper_gridmet = irr_mapper_max.setDefaultProjection(
                        crs=irr_mapper_proj, 
                        scale=30
                        ).reduceResolution(
                        reducer=ee.Reducer.count(),
                        bestEffort=True,
                        maxPixels=max_pixels
                    ).reproject(cdl_proj, scale=gridmet_scale)

                    # Reduce LANID V2 to gridMET resolution and sum up irrigated pixels
                    lanidv2_gridmet = lanid_img_max.setDefaultProjection(
                            crs=lanid_img_max.projection(), 
                            scale=30
                        ). reduceResolution(
                        reducer=ee.Reducer.count(),
                        bestEffort=True,
                        maxPixels=max_pixels
                    ).reproject(cdl_proj, scale=gridmet_scale)

                    mode_reducer = ee.Reducer.mode(maxRaw=max_pixels)
                    
                    # Reduce CDL to gridMET resolution and take the mode
                    cdl_gridmet = cdl_year.reduceResolution(
                        reducer=mode_reducer,
                        bestEffort=True,
                        maxPixels=max_pixels
                    ).reproject(cdl_proj, scale=gridmet_scale)

                    cdl_gridmet_irrmapper = cdl_irrmapper.reduceResolution(
                        reducer=mode_reducer,
                        bestEffort=True,
                        maxPixels=max_pixels
                    ).reproject(cdl_proj, scale=gridmet_scale)

                    cdl_gridmet_lanid = cdl_lanid.reduceResolution(
                        reducer=mode_reducer,
                        bestEffort=True,
                        maxPixels=max_pixels
                    ).reproject(cdl_proj, scale=gridmet_scale)


                    # Get the geometry
                    geom = ee.Geometry.Point(row[lon_col], row[lat_col])
                    
                    # Extract the data at the given point
                    nlcd_ag_val = nlcd_ag_gridmet.sample(
                        region=geom,
                        scale=gridmet_scale,
                        numPixels=1
                    ).first().getInfo()

                    cdl_ag_val = cdl_ag_gridmet.sample(
                        region=geom,
                        scale=gridmet_scale,
                        numPixels=1
                    ).first().getInfo()

                    irr_mapper_val = irr_mapper_gridmet.sample(
                        region=geom,
                        scale=gridmet_scale,
                        numPixels=1
                    ).first().getInfo()

                    lanid_val = lanidv2_gridmet.sample(
                        region=geom,
                        scale=gridmet_scale,
                        numPixels=1
                    ).first().getInfo()

                    cdl_val_unmasked = cdl_gridmet.sample(
                        region=geom,
                        scale=gridmet_scale,
                        numPixels=1
                    ).first().getInfo()

                    cdl_val_irrmapper = cdl_gridmet_irrmapper.sample(
                        region=geom,
                        scale=gridmet_scale,
                        numPixels=1
                    ).first().getInfo()

                    cdl_val_lanid = cdl_gridmet_lanid.sample(
                        region=geom,
                        scale=gridmet_scale,
                        numPixels=1
                    ).first().getInfo()
                else:
                    # Create a buffer around the point
                    target_scale = buffer_size
                    geom = ee.Geometry.Point(row[lon_col], row[lat_col]).buffer(target_scale)
                    nlcd_ag_val = nlcd_ag.reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=geom,
                        scale=30
                    ).get('NLCD_Ag').getInfo()
                    cdl_ag_val = cdl_ag.reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=geom,
                        scale=30
                    ).get('CDL_Ag').getInfo()
                    irr_mapper_val = irr_mapper_max.reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=geom,
                        scale=30
                    ).get('IrrMapper').getInfo()
                    lanid_val = lanid_img_max.reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=geom,
                        scale=30
                    ).get('LANID').getInfo()
                    cdl_val_unmasked = cdl_year.reduceRegion(
                        reducer=ee.Reducer.mode(),
                        geometry=geom,
                        scale=30
                    ).get('cropland').getInfo()
                    cdl_val_irrmapper = cdl_irrmapper.reduceRegion(
                        reducer=ee.Reducer.mode(),
                        geometry=geom,
                        scale=30
                    ).get('cropland').getInfo()
                    cdl_val_lanid = cdl_lanid.reduceRegion(
                        reducer=ee.Reducer.mode(),
                        geometry=geom,
                        scale=30
                    ).get('cropland').getInfo()

                total_area = target_scale ** 2

                nlcd_ag_val = no_data if nlcd_ag_val is None else \
                    nlcd_ag_val['properties']['NLCD_Ag'] if not use_buffer else nlcd_ag_val
                nlcd_ag_frac = no_data if nlcd_ag_val is None else nlcd_ag_val  * 900 / total_area
                nlcd_ag_frac = nlcd_ag_frac if nlcd_ag_frac <= 1 else 1

                cdl_ag_val = no_data if cdl_ag_val is None else \
                    cdl_ag_val['properties']['CDL_Ag'] if not use_buffer else cdl_ag_val
                cdl_ag_frac = no_data if cdl_ag_val is None else cdl_ag_val  * 900 / total_area
                cdl_ag_frac = cdl_ag_frac if cdl_ag_frac <= 1 else 1

                irr_mapper_val = no_data if irr_mapper_val is None else \
                    irr_mapper_val['properties']['IrrMapper'] if not use_buffer else irr_mapper_val
                irr_mapper_frac = no_data if irr_mapper_val is None else irr_mapper_val  * 900 / total_area
                irr_mapper_frac = irr_mapper_frac if irr_mapper_frac <= 1 else 1

                lanid_val = no_data if lanid_val is None else \
                    lanid_val['properties']['LANID'] if not use_buffer else lanid_val
                lanid_frac = no_data if lanid_val is None else lanid_val  * 900 / total_area
                lanid_frac = lanid_frac if lanid_frac <= 1 else 1

                cdl_val_unmasked = no_data if cdl_val_unmasked is None else \
                    cdl_val_unmasked['properties']['cropland'] if not use_buffer else int(cdl_val_unmasked)
                cdl_val_irrmapper = no_data if cdl_val_irrmapper is None else \
                    cdl_val_irrmapper['properties']['cropland'] if not use_buffer else int(cdl_val_irrmapper)
                cdl_val_lanid = no_data if cdl_val_lanid is None else \
                    cdl_val_lanid['properties']['cropland'] if not use_buffer else int(cdl_val_lanid)
                
                if verbose:
                    print('\nStation:', row[station_col])
                    print(
                        'NLCD Ag:', nlcd_ag_val, 'CDL Ag:', cdl_ag_val,
                        'IrrMapper:', irr_mapper_val, 'LANID:', lanid_val
                    )
                    print('CDL (UM, IrrMapper, LANID):', cdl_val_unmasked, 
                        cdl_val_irrmapper, cdl_val_lanid
                    )
                    print(
                        'NLCD Ag Fraction:', np.round(nlcd_ag_frac, 3), 
                        'CDL Ag Fraction:', np.round(cdl_ag_frac, 3),
                        'IrrMapper Fraction:', np.round(irr_mapper_frac, 3), 
                        'LANID Fraction:', np.round(lanid_frac, 3)
                    )
                
                # Update the bias vector file
                bias_vector.loc[idx, 'NLCD_Ag Count'] = nlcd_ag_val
                bias_vector.loc[idx, 'CDL_Ag Count'] = cdl_ag_val
                bias_vector.loc[idx, 'IrrMapper Count'] = irr_mapper_val
                bias_vector.loc[idx, 'LANID Count'] = lanid_val
                bias_vector.loc[idx, 'CDL Unmasked'] = cdl_val_unmasked
                bias_vector.loc[idx, 'CDL IrrMapper'] = cdl_val_irrmapper
                bias_vector.loc[idx, 'CDL LANID'] = cdl_val_lanid
                bias_vector.loc[idx, 'IrrMapper Fraction'] = irr_mapper_frac
                bias_vector.loc[idx, 'LANID Fraction'] = lanid_frac
                bias_vector.loc[idx, 'NLCD_Ag Fraction'] = nlcd_ag_frac
                bias_vector.loc[idx, 'CDL_Ag Fraction'] = cdl_ag_frac
            except ee.EEException as e:
                print(f'Error processing {start_year} - {end_year}...')
                print(f'Error: {e}')
        # Save the updated bias vector file as geojson and csv
        bias_vector.to_file(output_bias_vector_file)
    else:
        if os.path.exists(output_csv_file):
            bias_df = pd.read_csv(output_csv_file)
            bias_vector = gpd.read_file(output_bias_vector_file)
        else:
            bias_vector = gpd.read_file(output_bias_vector_file)
    if 'geometry' in bias_vector.columns:
        bias_df = bias_vector.drop(columns=['geometry'])
    bias_df = fix_cdl_classes(bias_df.copy(deep=True))
    bias_df = categorize_irr_ag_fraction(bias_df, num_ag_cats, use_buffer)
    bias_df.to_csv(output_csv_file, index=False)
    return bias_df
