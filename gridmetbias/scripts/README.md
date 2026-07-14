# gridMET Bias Analysis Scripts

This directory contains Python scripts for analyzing gridMET reference evapotranspiration (ETo) bias and related weather variables. The scripts generate various visualizations, statistical analyses, and data processing workflows.

## Table of Contents
- [Script Overview](#script-overview)
- [Data Dependencies](#data-dependencies)
- [Execution Order](#execution-order)

---

## Script Overview

### 1. `data_formatting.py`
**Purpose:** Performs spatial joins of station ETo bias results with Köppen climate zone codes and adds climate zone information.

**Authors:** Dr. John Volk, Dr. Sayantan Majumdar

**Input Data Files:**
| File | Path |
|------|------|
| Bias summary (ETo/ETr) | `../../Data/Point_bias_data/{var_name}_summary_comp_all_yrs.csv` |
| Bias summary (other vars) | `../../Data/Point_bias_data/{var_name}_summary_comp_merged.csv` |
| Climate GeoJSON | `../../Data/Point_bias_data/Climate/{var_name}_summary_comp_*_climate.geojson` |
| Köppen info | `../../Data/koppen_ID_info.csv` |

**Output Data Files:**
| File | Path |
|------|------|
| Merged climate data | `../../Data/Point_bias_data/Climate/{var_name}_merged_with_climate.csv` |

**Variables Processed:** `eto_mm`, `u2_ms`, `etr_mm`, `ea_kpa`, `srad_wm2`, `tmin_c`, `tmax_c`

---

### 2. `boxplots_stats.py`
**Purpose:** Creates box plots and calculates summary statistics for weather station variable biases, grouped by East/West U.S. regions and Köppen climate zones.

**Authors:** Dr. John Volk, Dr. Sayantan Majumdar

**Prerequisites:** Run `data_formatting.py` first.

**Input Data Files:**
| File | Path |
|------|------|
| Merged climate data | `../../Data/Point_bias_data/Climate/{var_name}_merged_with_climate.csv` |

**Output Files:**
| File | Path |
|------|------|
| Box plot figures | `../../Plots/Boxplots/{time_period}/{var}_Bias_Boxplots_{time_period}.png` |
| Statistics CSV | `../../Plots/Boxplots/{time_period}/{var}_Bias_Stats_{time_period}.csv` |

**Configurable Options:**
- `plot_variable`: Time period for analysis (`annual_mean`, `summer_mean`, `growseason_mean`, monthly means)

---

### 3. `gen_map.py`
**Purpose:** Generates per-variable CONUS coverage maps for every variable in the CONUS-AgWeather standardized data. For each variable, produces a three-panel figure showing days of observations, years of observations, and average annual record completeness (%) at each station. Per-station, per-variable stats are computed once and cached to `variable_stats.csv` so re-plotting is fast.

**Authors:** Christian Dunkerly, Dr. Sayantan Majumdar

**Input Data Files:**
| File | Path |
|------|------|
| Station metadata | `../../Data/CONUS-AgWeather_v1/metadata_for_publication.csv` |
| Per-station data (Parquet) | `../../Data/CONUS-AgWeather_v1/standardized_data_parquet/*_corrected.parquet` |
| States shapefile | `../../Data/states/states.shp` |

**Output Files:**
| File | Path |
|------|------|
| Per-variable maps (×18) | `../../Plots/Variable_Maps/{Variable}_map.png` |
| Cached stats | `../../Plots/Variable_Maps/variable_stats.csv` |

**Map Variables:** ETo, ETr, TMax, TAvg, TMin, Ea, TDew, RHMax, RHAvg, RHMin, Compiled Ea, Rs, Optimized TR Rs, Rso, Measured Uz, Anemometer Height, Uz at 2m, Precipitation

**Completeness convention:** Per-year ratio = (valid days for this variable in year `y`) / (days the station was active in year `y`), averaged across all calendar years in the station's record. Matches the original `station_map` figure's definition.

---

### 4. `conus_agweather_eto_analysis.py`
**Purpose:** Analyzes CONUS-AgWeather pre- and post-QC ETo data, calculating daily and annual QC factors.

**Author:** Dr. Sayantan Majumdar

**Input Data Files:**
| File | Path |
|------|------|
| Station Excel files | `../../Data/CONUS-AgWeather_v1/standardized_data_xlsx/*.xlsx` |
| Climate parquet | `../../Data/supporting_files/Station_Climate/station_climate_data.parquet` |

**Output Files:**
| File | Path |
|------|------|
| Output data | `../../Data/Outputs/` |
| ETo statistics plots | `../../Plots/CONUS-AgWeather_v1_ETo_Stats/` |

**Required Excel Sheets:** `Corrected Data`, `Delta (Corr - Orig)`

---

### 5. `conus_agweather_var_analysis.py`
**Purpose:** Analyzes CONUS-AgWeather variables (Rs, Rso, etc.) pre- and post-QC, generating comparison plots.

**Author:** Dr. Sayantan Majumdar

**Input Data Files:**
| File | Path |
|------|------|
| Station Excel files | `../../Data/CONUS-AgWeather_v1/standardized_data_xlsx/*.xlsx` |

**Output Files:**
| File | Path |
|------|------|
| Output data | `../../Data/Outputs/` |
| Variable statistics plots | `../../Plots/CONUS-AgWeather_v1_Var_Stats/` |

**Variables Analyzed:** Rs (solar radiation), Rso (clear-sky solar radiation), Optimized TR Rs

---

### 6. `OpenET_flux_grouped_scatter_plots.py`
**Purpose:** Creates scatter plots comparing OpenET versus flux tower ET before and after ETo bias correction.

**Author:** Dr. John Volk

**Input Data Files:**
| File | Path |
|------|------|
| Monthly corrected | `../../Data/paired_flux_OpenET_data/merged_monthly_corrv3.csv` |
| Monthly uncorrected | `../../Data/paired_flux_OpenET_data/merged_monthly_uncorrv3.csv` |
| Daily corrected | `../../Data/paired_flux_OpenET_data/merged_daily_corrv3.csv` |
| Daily uncorrected | `../../Data/paired_flux_OpenET_data/merged_daily_uncorrv3.csv` |

**Output Files:**
| File | Path |
|------|------|
| Scatter plots | `../../Plots/OpenET_accuracy/Figure6_croplands_monthly_openet_vs_flux.jpg` |

**Land Type Categories:** Croplands, Evergreen Forests, Grasslands, Mixed Forests, Shrublands, Wetland/Riparian

---

### 7. `site_analysis_gridmet_openet.py`
**Purpose:** Merges GridMET ETo and OpenET actual ET data with flux tower data for each site.

**Authors:** Dr. Sayantan Majumdar, Dr. John Volk

**Input Data Files:**
| File | Path |
|------|------|
| GridMET daily | `../../Plots/GridMET_Plots/All/GridMET_Daily_All_Station_Data.csv` |
| GridMET monthly | `../../Plots/GridMET_Plots/All/GridMET_Monthly_All_Station_Data.csv` |
| OpenET daily | `../../Data/paired_flux_OpenET_data/merged_daily*corr*.csv` |
| OpenET monthly | `../../Data/paired_flux_OpenET_data/merged_monthly*corr*.csv` |

**Output Files:**
| File | Path |
|------|------|
| Merged daily files | `../../Data/paired_flux_OpenET_data/openet_gridmet_merged_daily_jtype_{join_type}_{version}.csv` |
| Merged monthly files | `../../Data/paired_flux_OpenET_data/openet_gridmet_merged_monthly_jtype_{join_type}_{version}.csv` |
| Zipped archive | `../../Data/paired_flux_OpenET_data/merged_gridmet_openet_files.zip` |

**Join Types:** `inner`, `left`, `right`

---

### 8. `site_analysis_gridmet.py`
**Purpose:** Performs site analysis for gridMET reference ET data, generating scatter plots of gridMET ET against flux tower ET.

**Author:** Dr. Sayantan Majumdar, Dr. John Volk

**Input Data Files:**
| File | Path |
|------|------|
| Merged GridMET flux data | Generated from `corr_analysis_gridmet.py` |

**Output Files:**
| File | Path |
|------|------|
| Scatter plots | `../../Plots/Site_Analysis_GridMET/{site_id}/` |
| Metrics CSV | `../../Plots/Site_Analysis_GridMET/All_cropland_sites_gridmet_metrics.csv` |

**Calculated Metrics:** R², MAE, MBE

**Dependencies:** Imports from `site_analysis_openet.py`

---

### 9. `site_analysis_openet.py`
**Purpose:** Performs site analysis for OpenET data, generating scatter plots of OpenET ET against flux tower ET for different models.

**Author:** Dr. Sayantan Majumdar, Dr. John Volk

**Input Data Files:**
| File | Path |
|------|------|
| Paired flux OpenET data | `../../Data/paired_flux_OpenET_data/` |

**Output Files:**
| File | Path |
|------|------|
| Scatter plots | `../../Plots/Site_Analysis_OpenET/{site_id}/` |
| Metrics CSV | `../../Plots/Site_Analysis_OpenET/*.csv` |

**OpenET Models Analyzed:** EEMETRIC, SSEBOP, SIMS, GEESEBAL, PTJPL, DISALEXI, ensemble_mean

**Calculated Metrics:** R², MAE, MBE

---

### 10. `station_climate_plots.py`
**Purpose:** Creates plots for station variables grouped by Köppen climate classification, including KDE and violin plots.

**Author:** Dr. Sayantan Majumdar

**Input Data Files:**
| File | Path |
|------|------|
| Station metadata | CSV file with station information |
| Station Excel files | `../../Data/CONUS-AgWeather_v1/standardized_data_xlsx/*.xlsx` |
| Climate classification | Climate CSV with station IDs and Köppen codes |

**Output Files:**
| File | Path |
|------|------|
| Station climate parquet | `{output_dir}/station_climate_data.parquet` |
| Climate plots | `../../Plots/Station_Climate/` |

**Station Variables Plotted:**
- ETo, ETr (mm/day)
- Temperature: TMax, TAvg, TMin, TDew (°C)
- Humidity: RHMax, RHAvg, RHMin (%)
- Vapor pressure: Ea, Compiled Ea (kPa)
- Solar radiation: Rs, Rso, Optimized TR Rs (W/m²)
- Wind: Uz at 2m (m/s)
- Precipitation (mm)

**Climate Classifications:** Bsk+Bsh, Bwh+Bwk, Cfa, Csa+Csb, Dfa+Dfb

---

### 11. `monthly_climos.py`
**Purpose:** Creates a 2×2 panel plot of croplands monthly ET climatology comparing flux-tower closed/unclosed ET with corrected and uncorrected OpenET models (Figure 7).

**Author:** Dr. John Volk

**Input Data Files:**
| File | Path |
|------|------|
| Monthly corrected | `../../Data/paired_flux_OpenET_data/merged_monthly_corrv3.csv` |
| Monthly uncorrected | `../../Data/paired_flux_OpenET_data/merged_monthly_uncorrv3.csv` |

**Output Files:**
| File | Path |
|------|------|
| Climatology plot | `../../Plots/OpenET_accuracy/Figure7_croplands_monthly_climatology.jpg` |

**OpenET Models Plotted:** Ensemble, eeMETRIC, SIMS, SSEBop

---

### 12. `monthly_error_delta_bias_heatmaps.py`
**Purpose:** Generates monthly absolute error reduction heatmaps with bias sign overlay across land cover types and OpenET models (Figure 8). Visualizes how ETo bias correction affects absolute error in OpenET models.

**Author:** Dr. John Volk

**Input Data Files:**
| File | Path |
|------|------|
| Monthly corrected | `../../Data/paired_flux_OpenET_data/merged_monthly_corrv3.csv` |
| Monthly uncorrected | `../../Data/paired_flux_OpenET_data/merged_monthly_uncorrv3.csv` |

**Output Files:**
| File | Path |
|------|------|
| Heatmap plot | `../../Plots/OpenET_accuracy/Figure8_absolute_error_reduction_heatmaps.jpg` |

**OpenET Models Plotted:** Ensemble, eeMETRIC, SIMS, SSEBop

**Land Cover Types:** Croplands, Evergreen Forests, Grasslands, Mixed Forests, Shrublands, Wetlands

---

### 13. `monthly_ET_vs_ETo_error_scatter.py`
**Purpose:** Creates scatter plots of absolute improvement in monthly OpenET ET (at EC sites) after applying ETo bias correction versus improvement in ETo at the same flux stations, grouped by land cover type (Figure 9).

**Author:** Dr. John Volk

**Input Data Files:**
| File | Path |
|------|------|
| Monthly corrected ET | `../../Data/paired_flux_OpenET_data/merged_monthly_corrv3.csv` |
| Monthly uncorrected ET | `../../Data/paired_flux_OpenET_data/merged_monthly_uncorrv3.csv` |
| Monthly ETo data | `../../Data/flux_gridmet/flux_gridmet_monthly.csv` |

**Output Files:**
| File | Path |
|------|------|
| Scatter plot | `../../Plots/OpenET_accuracy/Figure9_error_reduction_scatter_by_landcover.jpg` |

**OpenET Models Plotted:** Ensemble, eeMETRIC, SIMS, SSEBop

**Land Cover Types:** Croplands, Evergreen Forests, Grasslands, Mixed Forests, Shrublands, Wetlands

---

### 14. `convert_to_parquet.py`
**Purpose:** Packages the CONUS-AgWeather_v1 dataset for distribution. Strips the empty "Filled Data" sheet from every station xlsx file (verified all-NaN across the dataset), converts each remaining sheet to Parquet for fast loading, then builds the distributable `CONUS-AgWeather_v1.zip` archive (xlsx + parquet + QC plots + metadata + Variable_Maps), excluding `.DS_Store` files. Uses multiprocessing for the xlsx and Parquet steps.

**Author:** Dr. Sayantan Majumdar

**Input Data Files:**
| File | Path |
|------|------|
| Station Excel files | `../../Data/CONUS-AgWeather_v1/standardized_data_xlsx/*.xlsx` |
| Variable maps | `../../Plots/Variable_Maps/` |

**Output Files:**
| File | Path |
|------|------|
| Per-sheet Parquet files | `../../Data/CONUS-AgWeather_v1/standardized_data_parquet/{base}_corrected.parquet`, `{base}_delta.parquet` |
| Distributable archive | `../../CONUS-AgWeather_v1.zip` |

**Notes:** Idempotent — re-running detects xlsx files already stripped and removes stale `*_filled.parquet` files automatically. Run this whenever the Variable_Maps are regenerated to refresh the zip.

---


## Data Dependencies

### Core Data Directories
```
Data/
├── CONUS-AgWeather_v1/
│   └── standardized_data_xlsx/          # Station Excel files
├── Point_bias_data/
│   └── Climate/                    # Climate-merged bias data
├── flux_gridmet/                   # Paired flux-gridMET monthly/daily data
├── paired_flux_OpenET_data/        # Flux tower and OpenET merged data
├── koppen_ID_info.csv              # Köppen climate zone info
├── openet_ground_station_master_list_cleaned_v4.csv
├── states/                         # State boundaries shapefile
└── supporting_files/
    └── Station_Climate/            # Climate parquet files
```

### Output Directories
```
Plots/
├── Boxplots/                       # Bias boxplots by time period
├── CONUS-AgWeather_v1_ETo_Stats/   # ETo analysis plots
├── CONUS-AgWeather_v1_Var_Stats/   # Variable analysis plots
├── GridMET_Plots/                  # GridMET comparison plots
├── OpenET_accuracy/                # OpenET vs flux scatter plots
├── Site_Analysis_GridMET/          # GridMET site analysis
├── Site_Analysis_OpenET/           # OpenET site analysis
├── Station_Climate/                # Climate-grouped station plots
└── Crop_Bias_Distributions/        # CDL crop-grouped plots
```

---

## Execution Order

For a complete analysis workflow, run scripts in the following order:

1. **Data Preparation and Visualization for gridMET bias correction paper:**
   - `data_formatting.py` - Merge bias data with climate zones
   - `boxplots_stats.py` - Create bias boxplots and statistics
   - `OpenET_flux_grouped_scatter_plots.py` - Croplands monthly OpenET vs flux scatter plots
   - `monthly_climos.py` - Monthly ET climatology
   - `monthly_error_delta_bias_heatmaps.py` - Error reduction heatmaps
   - `monthly_ET_vs_ETo_error_scatter.py` - Error reduction scatter by land cover
   - `site_analysis_gridmet_openet.py` - Merge GridMET and OpenET data
   - `site_analysis_openet.py` - OpenET site analysis
   - `site_analysis_gridmet.py` - GridMET site analysis   

2. **CONUS-AgWeather Visualization and Analysis:**
   - `gen_map.py` - Generate station location maps
   - `station_climate_plots.py` - Climate-grouped plots
   - `conus_agweather_var_analysis.py` - Analyze individual variables
   - `conus_agweather_eto_analysis.py` - Analyze ETo specifically
---

## Required Python Packages

```
pandas
numpy
geopandas
matplotlib
seaborn
scipy
scikit-learn
earthengine-api
pyarrow
tqdm
openpyxl
```

---

## Authors

- Dr. John Volk (john.volk@dri.edu)
- Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)
- Christian Dunkerly (christian.dunkerly@dri.edu)

## Citations

**Journal Articles:**

Volk, J. M., Dunkerly, C., Majumdar, S., Huntington, J. L., Minor, B. A., Kim, Y., Morton, C. G., ReVelle, P., Kilic, A., Melton, F., Allen, R. G., Pearson, C., Purdy, A. J., & Caldwell, T. G. (2026). 
Assessing and Correcting Bias in Gridded Reference Evapotranspiration over Agricultural Lands Across the Contiguous United States. _Accepted in Agricultural Water Management_. Preprint: https://doi.org/10.31223/X54F38

Dunkerly, C., Volk, J. M., Majumdar, S.,  Huntington, J. L., Allen, R. G., Pearson, C., Kim, Y., Morton, C. G., Minor, B. A., ReVelle, P., Kilic, A., Melton, F., Purdy, A. J., & Caldwell, T. G. (2026). A Benchmark Dataset of Agricultural Weather Stations over the Contiguous United States for Evapotranspiration Applications. _Accepted in Nature Scientific Data_. https://doi.org/10.1038/s41597-026-07819-7. Preprint: https://doi.org/10.31223/X56T9Z.

**Data Releases:**

Volk, J., Dunkerly, C., Majumdar, S., Huntington, J., Minor, B., Kim, Y., Morton, C., ReVelle, P., Kilic, A., Melton, F., Allen, R., Pearson, C., Purdy, A., & Caldwell, T. (2026). CONUS Gridded Reference Evapotranspiration Bias Correction: Inputs, Station Validation, and Outputs (gridMET/OpenET) [Data set]. _Zenodo_. https://doi.org/10.5281/zenodo.18673483

Dunkerly, C., Volk, J. M., Majumdar, S., Huntington, J. L., Allen, R. G., Pearson, C., Kim, Y., Morton, C. G., Minor, B. A., ReVelle, P., Kilic, A., Melton, F., Purdy, A. J., & Caldwell, T. G. (2026). CONUS-AgWeather, a high-quality benchmark daily agricultural weather station dataset for evapotranspiration applications in the Contiguous United States [Data set]. _Zenodo_. https://doi.org/10.5281/zenodo.18122156.