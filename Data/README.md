# Data Directory

This directory contains input datasets and analysis outputs for the gridMET bias correction and CONUS-AgWeather analysis.

## Directory Structure

```
Data/
├── koppen_ID_info.csv                        # Köppen climate zone information
├── openet_ground_station_master_list_cleaned_v4.csv  # Master station list
├── climateClass_poly_diss/                   # Climate classification shapefiles
├── CONUS-AgWeather_v1/                       # CONUS-AgWeather dataset
│   ├── metadata_for_publication.csv
│   ├── standardized_data_xlsx/               # Station Excel files (Corrected + Delta)
│   ├── standardized_data_parquet/            # Same daily data in Parquet
│   ├── after_qc_composite_graphs/            # Composite graphs after QC
│   ├── before_qc_composite_graphs/           # Composite graphs before QC
│   ├── log_files/                            # QC processing logs
│   └── variable_qc_graphs/                   # Variable-specific QC visualizations
├── flux_data/                                # GridMET reference ET data
├── flux_ET_dataset/                          # Flux tower ET observations
│   ├── daily_data_files/
│   ├── monthly_data_files/
│   └── station_metadata.xlsx
├── flux_gridmet/                             # Paired flux-gridMET data
├── metadata/                                 # Station metadata
├── Outputs/                                  # Analysis output files
├── paired_flux_OpenET_data/                  # Merged flux and OpenET data
├── Point_bias_data/                          # Station-level bias summaries
│   └── Climate/                              # Climate-joined bias data
├── states/                                   # US state boundary shapefiles
└── supporting_files/                         # Additional supporting data
    └── Station_Climate/                      # Station climate parquet files
```

The data required for this project are available from Zenodo:

**CONUS-AgWeather_v1**: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18122156.svg)](https://doi.org/10.5281/zenodo.18122156)

**Input and Output Datasets and Plots for gridMET bias correction**: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18673483.svg)](https://doi.org/10.5281/zenodo.18673483)

Download the data archive and extract its contents into the `Data/` directory. The CONUS-AgWeather_v1 zip archive from Zenodo must also be extracted within `Data/` so that the `CONUS-AgWeather_v1/` directory resides at `Data/CONUS-AgWeather_v1/`.

---

## Input Data Requirements by Script

### Main Script: `corr_analysis_gridmet.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Point bias CSVs | `Point_bias_data/*.csv` | Station-level bias ratio summaries for all variables |
| Climate shapefile | `climateClass_poly_diss/climateClass_poly_diss.shp` | Köppen climate zone polygons |
| Daily station files | `flux_ET_dataset/daily_data_files/` | Daily flux tower observations |
| Monthly station files | `flux_ET_dataset/monthly_data_files/` | Monthly flux tower observations |
| GridMET reference ET | `flux_data/openet_reference_et_summary_all_sites_bias_corr_paper.csv` | GridMET ETo data for all sites |
| Station metadata | `flux_ET_dataset/station_metadata.xlsx` | Station information and coordinates |

### Script: `data_formatting.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Bias summaries (ETo/ETr) | `Point_bias_data/{var}_summary_comp_all_yrs.csv` | ETo and ETr bias summaries |
| Bias summaries (other) | `Point_bias_data/{var}_summary_comp_merged.csv` | Other variable bias summaries |
| Climate GeoJSON | `Point_bias_data/Climate/{var}_*_climate.geojson` | Climate-joined bias data |
| Köppen info | `koppen_ID_info.csv` | Climate zone codes and descriptions |

### Script: `boxplots_stats.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Climate-merged data | `Point_bias_data/Climate/{var}_merged_with_climate.csv` | Output from `data_formatting.py` |

### Script: `gen_map.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Station metadata | `CONUS-AgWeather_v1/metadata_for_publication.csv` | Station IDs, names, lat/lon, networks |
| Per-station data | `CONUS-AgWeather_v1/standardized_data_parquet/*_corrected.parquet` | Daily weather observations per station |
| States shapefile | `states/states.shp` | US state boundaries |

### Script: `conus_agweather_eto_analysis.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Station Excel files | `CONUS-AgWeather_v1/standardized_data_xlsx/*.xlsx` | Standardized station data with QC |
| Climate parquet | `supporting_files/Station_Climate/station_climate_data.parquet` | Station climate classifications |

### Script: `conus_agweather_var_analysis.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Station Excel files | `CONUS-AgWeather_v1/standardized_data_xlsx/*.xlsx` | Standardized station data |

### Script: `OpenET_flux_grouped_scatter_plots.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Monthly corrected | `paired_flux_OpenET_data/merged_monthly_corrv3.csv` | Bias-corrected monthly data |
| Monthly uncorrected | `paired_flux_OpenET_data/merged_monthly_uncorrv3.csv` | Uncorrected monthly data |
| Daily corrected | `paired_flux_OpenET_data/merged_daily_corrv3.csv` | Bias-corrected daily data |
| Daily uncorrected | `paired_flux_OpenET_data/merged_daily_uncorrv3.csv` | Uncorrected daily data |

### Script: `site_analysis_gridmet_openet.py`

| Data File | Path | Description |
|-----------|------|-------------|
| OpenET daily files | `paired_flux_OpenET_data/merged_daily*corr*.csv` | Paired flux-OpenET daily data |
| OpenET monthly files | `paired_flux_OpenET_data/merged_monthly*corr*.csv` | Paired flux-OpenET monthly data |

### Script: `station_climate_plots.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Station metadata | CSV with station information | Station IDs and coordinates |
| Station Excel files | `CONUS-AgWeather_v1/standardized_data_xlsx/*.xlsx` | Station observations |
| Climate classification | `Point_bias_data/Climate/*_merged_with_climate.csv` | Station climate zones |

### Script: `monthly_climos.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Monthly corrected | `paired_flux_OpenET_data/merged_monthly_corrv3.csv` | Bias-corrected monthly data |
| Monthly uncorrected | `paired_flux_OpenET_data/merged_monthly_uncorrv3.csv` | Uncorrected monthly data |

### Script: `monthly_error_delta_bias_heatmaps.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Monthly corrected | `paired_flux_OpenET_data/merged_monthly_corrv3.csv` | Bias-corrected monthly data |
| Monthly uncorrected | `paired_flux_OpenET_data/merged_monthly_uncorrv3.csv` | Uncorrected monthly data |

### Script: `monthly_ET_vs_ETo_error_scatter.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Monthly corrected ET | `paired_flux_OpenET_data/merged_monthly_corrv3.csv` | Bias-corrected monthly data |
| Monthly uncorrected ET | `paired_flux_OpenET_data/merged_monthly_uncorrv3.csv` | Uncorrected monthly data |
| Monthly ETo data | `flux_gridmet/flux_gridmet_monthly.csv` | Paired flux-gridMET monthly data |

---

## Data File Descriptions

### Root-Level Files

| File | Description |
|------|-------------|
| `koppen_ID_info.csv` | Köppen-Geiger climate classification codes (gridcode), abbreviations (Code), and full descriptions |
| `openet_ground_station_master_list_cleaned_v4.csv` | Master list of weather stations with coordinates, observation counts, record lengths, and completeness metrics |

### `climateClass_poly_diss/`

Köppen-Geiger climate classification shapefile for the contiguous United States.

| File | Description |
|------|-------------|
| `climateClass_poly_diss.shp` | Climate zone polygons |
| `climateClass_poly_diss.dbf` | Attribute table with gridcode values |
| `climateClass_poly_diss.prj` | Projection information |

### `CONUS-AgWeather_v1/`

CONUS-AgWeather dataset containing quality-controlled agricultural weather station data (~6.3 GB). Available from [Zenodo](https://doi.org/10.5281/zenodo.18122156).

| Subdirectory/File | Description | Size |
|-------------------|-------------|------|
| `standardized_data_xlsx/` | Excel files with corrected data and delta (correction) values | ~1.2 GB |
| `standardized_data_parquet/` | Same daily data as Parquet for fast loading (corrected + delta per station) | ~606 MB |
| `variable_qc_graphs/` | Variable-specific QC visualizations | ~2.1 GB |
| `after_qc_composite_graphs/` | Composite graphs after QC | ~1.2 GB |
| `before_qc_composite_graphs/` | Composite graphs before QC | ~1.2 GB |
| `log_files/` | QC processing logs | ~3.1 MB |
| `metadata_for_publication.csv` | Station metadata for publication | ~148 KB |

**Excel file sheets:**
- `Corrected Data` - QC-corrected observations
- `Delta (Corr - Orig)` - Correction amounts applied
- `Original Data` - Pre-QC observations

### `flux_data/`

| File | Description |
|------|-------------|
| `openet_reference_et_summary_all_sites_bias_corr_paper.csv` | GridMET reference ET (ETo, ETr) data matched to flux tower sites |

### `flux_ET_dataset/`

Flux tower evapotranspiration observations (~204 MB).

| File/Directory | Description |
|----------------|-------------|
| `daily_data_files/` | Daily flux tower observations |
| `monthly_data_files/` | Monthly aggregated observations |
| `station_metadata.xlsx` | Station coordinates, names, and metadata |

### `flux_gridmet/`

Paired flux tower and gridMET reference ET data (~12 MB).

| File | Description |
|------|-------------|
| `flux_gridmet_daily.csv` | Daily paired flux-gridMET data |
| `flux_gridmet_monthly.csv` | Monthly paired flux-gridMET data |

### `metadata/`

| File | Description |
|------|-------------|
| `station_metadata_from_flux_gridmet.csv` | Station metadata extracted from flux-gridMET pairing |

### `paired_flux_OpenET_data/`

Merged flux tower and OpenET model estimates.

| File Pattern | Description |
|--------------|-------------|
| `merged_daily_corrv3.csv` | All sites, daily, bias-corrected |
| `merged_daily_uncorrv3.csv` | All sites, daily, uncorrected |
| `merged_monthly_corrv3.csv` | All sites, monthly, bias-corrected |
| `merged_monthly_uncorrv3.csv` | All sites, monthly, uncorrected |

**Generated files (created by `site_analysis_gridmet_openet.py`):**
| `openet_gridmet_merged_*_jtype_*.csv` | GridMET + OpenET merged with different join types |

**Columns include:** SITE_ID, DATE, Latitude, Longitude, General classification, OpenET model estimates (EEMETRIC, SSEBOP, SIMS, GEESEBAL, PTJPL, DISALEXI, ensemble_mean), Closed/Unclosed flux values

### `Point_bias_data/`

Station-level bias ratio summaries (~18 MB).

| File Pattern | Description |
|--------------|-------------|
| `eto_mm_summary_comp_all_yrs.csv` | ETo bias ratios (station/gridMET) |
| `etr_mm_summary_comp_all_yrs.csv` | ETr bias ratios |
| `{var}_summary_comp_merged.csv` | Other variable bias ratios (ea, u2, srad, tmin, tmax) |

**Key columns:**
- `STATION_ID` - Station identifier
- `STATION_LAT`, `STATION_LON` - Coordinates
- `{month}_mean` - Monthly mean bias ratios (Jan-Dec)
- `annual_mean`, `summer_mean`, `growseason_mean` - Seasonal aggregates
- `start_year`, `end_year` - Record period

#### `Point_bias_data/Climate/`

Climate-joined bias data and GEE extraction outputs.

| File Pattern | Description |
|--------------|-------------|
| `{var}_merged_with_climate.csv` | Bias data with Köppen zone codes |
| `{var}_*_climate.geojson` | Spatial join with climate polygons |
| `{var}_*_irr_crop_*.csv` | Irrigated cropland subset |
| `{var}_*_noirr_crop_*.csv` | Non-irrigated cropland subset |
| `GEE_Data_Buffer/` | Google Earth Engine extracted data (IrrMapper, LANID, CDL) |

### `Outputs/`

Analysis output files.

| File | Description |
|------|-------------|
| `daily_eto_qc_factors.csv` | Daily ETo QC correction factors |
| `daily_eto_qc_factors_with_climate.csv` | Daily factors with climate zones |
| `annual_eto_qc_factors.csv` | Annual ETo QC correction factors |
| `annual_eto_qc_factors_with_climate.csv` | Annual factors with climate zones |

### `states/`

US state boundary shapefile for mapping (~42 MB).

| File | Description |
|------|-------------|
| `states.shp` | State polygon boundaries |
| `states.dbf` | Attribute table with STATE_ABBR |
| `states.prj` | Projection (typically EPSG:4326) |

### `supporting_files/`

Additional supporting datasets (~320 MB).

| Subdirectory | Description | Size |
|--------------|-------------|------|
| `Station_Climate/` | Station climate classification parquet files | ~320 MB |

---

## External Data Requirements

Data directories are available from Zenodo. CONUS-AgWeather_v1 has its own DOI; all other data and plots are under a separate DOI.

| Directory | Size | Source |
|-----------|------|--------|
| `CONUS-AgWeather_v1/` | ~6.3 GB | [Zenodo](https://doi.org/10.5281/zenodo.18122156) |
| `supporting_files/` | ~320 MB | [Zenodo](https://doi.org/10.5281/zenodo.18673483) |
| `flux_ET_dataset/` | ~204 MB | [Zenodo](https://doi.org/10.5281/zenodo.18673483) |
| `Outputs/` | ~173 MB | [Zenodo](https://doi.org/10.5281/zenodo.18673483) |
| `paired_flux_OpenET_data/` | ~436 MB | [Zenodo](https://doi.org/10.5281/zenodo.18673483) |
| `states/` | ~42 MB | [Zenodo](https://doi.org/10.5281/zenodo.18673483) |
| `flux_data/` | ~25 MB | [Zenodo](https://doi.org/10.5281/zenodo.18673483) |
| `Point_bias_data/` | ~18 MB | [Zenodo](https://doi.org/10.5281/zenodo.18673483) |
| `flux_gridmet/` | ~12 MB | [Zenodo](https://doi.org/10.5281/zenodo.18673483) |
| `climateClass_poly_diss/` | ~8 MB | [Zenodo](https://doi.org/10.5281/zenodo.18673483) |

---

## Variable Naming Conventions

| Variable Code | Description | Units |
|---------------|-------------|-------|
| `eto_mm` | Reference ET (short reference) | mm |
| `etr_mm` | Reference ET (tall reference) | mm |
| `ea_kpa` | Actual vapor pressure | kPa |
| `u2_ms` | Wind speed at 2m | m/s |
| `srad_wm2` | Solar radiation | W/m² |
| `tmin_c` | Minimum temperature | °C |
| `tmax_c` | Maximum temperature | °C |

---

## Data Sources

- **gridMET:** University of Idaho gridded meteorological data (~4 km resolution)
- **OpenET:** Satellite-based ET estimates from multiple models
- **IrrMapper:** University of Montana irrigation mapping (Western US)
- **LANID:** Landsat-based irrigation dataset (1997-2020)
- **USDA CDL:** Cropland Data Layer crop classification
- **NLCD:** National Land Cover Database
- **Köppen-Geiger:** Climate classification system

---

## Citations

**Journal Articles:**

Volk, J. M., Dunkerly, C., Majumdar, S., Huntington, J. L., Minor, B. A., Kim, Y., Morton, C. G., ReVelle, P., Kilic, A., Melton, F., Allen, R. G., Pearson, C., Purdy, A. J., & Caldwell, T. G. (2026). 
Assessing and Correcting Bias in Gridded Reference Evapotranspiration over Agricultural Lands Across the Contiguous United States. _Accepted in Agricultural Water Management_. Preprint: https://doi.org/10.31223/X54F38

Dunkerly, C., Volk, J. M., Majumdar, S.,  Huntington, J. L., Allen, R. G., Pearson, C., Kim, Y., Morton, C. G., Minor, B. A., ReVelle, P., Kilic, A., Melton, F., Purdy, A. J., & Caldwell, T. G. (2026). A Benchmark Dataset of Agricultural Weather Stations over the Contiguous United States for Evapotranspiration Applications. _Accepted in Nature Scientific Data_. https://doi.org/10.1038/s41597-026-07819-7. Preprint: https://doi.org/10.31223/X56T9Z.

**Data Releases:**

Volk, J., Dunkerly, C., Majumdar, S., Huntington, J., Minor, B., Kim, Y., Morton, C., ReVelle, P., Kilic, A., Melton, F., Allen, R., Pearson, C., Purdy, A., & Caldwell, T. (2026). CONUS Gridded Reference Evapotranspiration Bias Correction: Inputs, Station Validation, and Outputs (gridMET/OpenET) [Data set]. _Zenodo_. https://doi.org/10.5281/zenodo.18673483

Dunkerly, C., Volk, J. M., Majumdar, S., Huntington, J. L., Allen, R. G., Pearson, C., Kim, Y., Morton, C. G., Minor, B. A., ReVelle, P., Kilic, A., Melton, F., Purdy, A. J., & Caldwell, T. G. (2026). CONUS-AgWeather, a high-quality benchmark daily agricultural weather station dataset for evapotranspiration applications in the Contiguous United States [Data set]. _Zenodo_. https://doi.org/10.5281/zenodo.18122156.
