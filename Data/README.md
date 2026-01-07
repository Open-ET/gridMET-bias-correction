# Data Directory

This directory contains input datasets and analysis outputs for the gridMET bias correction and CONUS-AgWeather analysis.

## Directory Structure

```
Data/
├── koppen_ID_info.csv                        # Köppen climate zone information
├── openet_ground_station_master_list_cleaned_v4.csv  # Master station list
├── climateClass_poly_diss/                   # Climate classification shapefiles
├── CONUS-AgWeather_v1/                       # CONUS-AgWeather dataset [EXTERNAL]
├── flux_data/                                # GridMET reference ET data
├── flux_ET_dataset/                          # Flux tower ET observations
├── paired_flux_OpenET_data/                  # Merged flux and OpenET data
├── Point bias data/                          # Station-level bias summaries
├── Outputs/                                  # Analysis output files
├── states/                                   # US state boundary shapefiles (extract from states.zip)
└── supporting_files/                         # Additional supporting data [EXTERNAL]
```

**Note:** Directories marked `[EXTERNAL]` contain large datasets not included in the repository. See [External Data Requirements](#external-data-requirements) for download instructions.

---

## Input Data Requirements by Script

### Main Script: `corr_analysis_gridmet.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Point bias CSVs | `Point bias data/*.csv` | Station-level bias ratio summaries for all variables |
| Climate shapefile | `climateClass_poly_diss/climateClass_poly_diss.shp` | Köppen climate zone polygons |
| Daily station files | `flux_ET_dataset/daily_data_files/` | Daily flux tower observations |
| Monthly station files | `flux_ET_dataset/monthly_data_files/` | Monthly flux tower observations |
| GridMET reference ET | `flux_data/openet_reference_et_summary_all_sites_bias_corr_paper.csv` | GridMET ETo data for all sites |
| Station metadata | `flux_ET_dataset/station_metadata.xlsx` | Station information and coordinates |

### Script: `data_formatting.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Bias summaries (ETo/ETr) | `Point bias data/{var}_summary_comp_all_yrs.csv` | ETo and ETr bias summaries |
| Bias summaries (other) | `Point bias data/{var}_summary_comp_merged.csv` | Other variable bias summaries |
| Climate GeoJSON | `Point bias data/Climate/{var}_*_climate.geojson` | Climate-joined bias data |
| Köppen info | `koppen_ID_info.csv` | Climate zone codes and descriptions |

### Script: `boxplots_stats.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Climate-merged data | `Point bias data/Climate/{var}_merged_with_climate.csv` | Output from `data_formatting.py` |

### Script: `gen_map.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Station master list | `openet_ground_station_master_list_cleaned_v4.csv` | Station locations and metadata |
| States shapefile | `states/states.shp` | US state boundaries |

### Script: `conus_agweather_eto_analysis.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Station Excel files | `CONUS-AgWeather_v1/standardized_data/*.xlsx` | Standardized station data with QC |
| Climate parquet | `supporting_files/Station_Climate/station_climate_data.parquet` | Station climate classifications |

### Script: `conus_agweather_var_analysis.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Station Excel files | `CONUS-AgWeather_v1/standardized_data/*.xlsx` | Standardized station data |

### Script: `OpenET_flux_grouped_scatter_plots.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Monthly corrected | `paired_flux_OpenET_data/merged_monthly_corrv2.csv` | Bias-corrected monthly data |
| Monthly uncorrected | `paired_flux_OpenET_data/merged_monthly_uncorrv2.csv` | Uncorrected monthly data |
| Daily corrected | `paired_flux_OpenET_data/merged_daily_corrv2.csv` | Bias-corrected daily data |
| Daily uncorrected | `paired_flux_OpenET_data/merged_daily_uncorrv2.csv` | Uncorrected daily data |

### Script: `site_analysis_gridmet_openet.py`

| Data File | Path | Description |
|-----------|------|-------------|
| OpenET daily files | `paired_flux_OpenET_data/merged_daily*corr*.csv` | Paired flux-OpenET daily data |
| OpenET monthly files | `paired_flux_OpenET_data/merged_monthly*corr*.csv` | Paired flux-OpenET monthly data |

### Script: `station_climate_plots.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Station metadata | CSV with station information | Station IDs and coordinates |
| Station Excel files | `CONUS-AgWeather_v1/standardized_data/*.xlsx` | Station observations |
| Climate classification | `Point bias data/Climate/*_merged_with_climate.csv` | Station climate zones |

### Script: `station_crop_plots.py`

| Data File | Path | Description |
|-----------|------|-------------|
| Station DataFrame | With station ID, longitude, latitude | Station locations |
| Station Excel files | Station data directory | Station observations |

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

### `CONUS-AgWeather_v1/` [EXTERNAL]

CONUS-AgWeather dataset containing quality-controlled agricultural weather station data (~5.7 GB). Available from [Zenodo](https://doi.org/10.5281/zenodo.18122156).

| Subdirectory/File | Description |
|-------------------|-------------|
| `metadata_for_publication.csv` | Station metadata for publication |
| `standardized_data/` | Excel files with corrected data, original data, and delta (correction) values |
| `after_qc_composite_graphs/` | Composite graphs after QC |
| `before_qc_composite_graphs/` | Composite graphs before QC |
| `log_files/` | QC processing logs |
| `variable_qc_graphs/` | Variable-specific QC visualizations |

**Excel file sheets:**
- `Corrected Data` - QC-corrected observations
- `Delta (Corr - Orig)` - Correction amounts applied
- `Original Data` - Pre-QC observations

### `flux_data/`

| File | Description |
|------|-------------|
| `openet_reference_et_summary_all_sites_bias_corr_paper.csv` | GridMET reference ET (ETo, ETr) data matched to flux tower sites |

### `flux_ET_dataset/`

Flux tower evapotranspiration observations.

| File/Directory | Description |
|----------------|-------------|
| `daily_data_files/` | Daily flux tower observations |
| `monthly_data_files/` | Monthly aggregated observations |
| `graphical_files/` | Visualization outputs |
| `station_metadata.xlsx` | Station coordinates, names, and metadata |
| `variable_explanation.xlsx` | Variable definitions and units |

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

### `Point bias data/`

Station-level bias ratio summaries.

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

#### `Point bias data/Climate/`

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

US state boundary shapefile for mapping (~42 MB). The repository includes `states.zip` which can be extracted to create this directory:

```bash
cd Data/
unzip states.zip
```

| File | Description |
|------|-------------|
| `states.shp` | State polygon boundaries |
| `states.dbf` | Attribute table with STATE_ABBR |
| `states.prj` | Projection (typically EPSG:4326) |

### `supporting_files/` [EXTERNAL]

Additional supporting datasets (~1.3 GB total).

| Subdirectory | Description | Size |
|--------------|-------------|------|
| `Station_Climate/` | Station climate classification parquet files | ~320 MB |
| `Station_CDL/` | Station CDL (Cropland Data Layer) data | ~1 GB |

---

## External Data Requirements

The following directories contain large datasets that must be obtained separately:

| Directory | Size | Source |
|-----------|------|--------|
| `CONUS-AgWeather_v1/` | ~5.7 GB | [Zenodo](https://doi.org/10.5281/zenodo.18122156) |
| `supporting_files/` | ~1.3 GB | Generated by analysis scripts or contact authors |

**Note:** `states/` can be created by extracting the included `states.zip` file.

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
