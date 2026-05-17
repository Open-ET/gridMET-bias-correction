# gridMET Bias Correction and CONUS-AgWeather

This repository contains code, data processing pipelines, and visualization tools for analyzing biases in gridMET reference evapotranspiration (ETo) and developing the CONUS-AgWeather quality-controlled weather station dataset.

## Overview

The gridMET dataset provides high-resolution (~4 km) daily surface meteorological data across the contiguous United States. This project:

1. **Quantifies systematic biases** in gridMET ETo and meteorological variables compared to ground-based weather station observations
2. **Develops bias correction factors** stratified by geographic region, Köppen climate zone, irrigation density, and crop type
3. **Provides the CONUS-AgWeather dataset** - a quality-controlled benchmark of daily agricultural weather station data for ET applications

## Citations

**Journal Articles:**

Volk, J. M., Dunkerly, C., Majumdar, S., Huntington, J. L., Minor, B. A., Kim, Y., Morton, C. G., ReVelle, P., Kilic, A., Melton, F., Allen, R. G., Pearson, C., Purdy, A. J., & Caldwell, T. G. (2026). 
Assessing and Correcting Bias in Gridded Reference Evapotranspiration over Agricultural Lands Across the Contiguous United States. _Under review in Agricultural Water Management_. Preprint: https://doi.org/10.31223/X54F38

Dunkerly, C., Volk, J. M., Majumdar, S.,  Huntington, J. L., Allen, R. G., Pearson, C., Kim, Y., Morton, C. G., Minor, B. A., ReVelle, P., Kilic, A., Melton, F., Purdy, A. J., & Caldwell, T. G. (2026). A Benchmark Dataset of Agricultural Weather Stations over the Contiguous United States for Evapotranspiration Applications. _Under review in Nature Scientific Data_. Preprint: https://doi.org/10.31223/X56T9Z.

**Data Releases:**

Volk, J., Dunkerly, C., Majumdar, S., Huntington, J., Minor, B., Kim, Y., Morton, C., ReVelle, P., Kilic, A., Melton, F., Allen, R., Pearson, C., Purdy, A., & Caldwell, T. (2026). CONUS Gridded Reference Evapotranspiration Bias Correction: Inputs, Station Validation, and Outputs (gridMET/OpenET) [Data set]. _Zenodo_. https://doi.org/10.5281/zenodo.18673484

Dunkerly, C., Volk, J. M., Majumdar, S., Huntington, J. L., Allen, R. G., Pearson, C., Kim, Y., Morton, C. G., Minor, B. A., ReVelle, P., Kilic, A., Melton, F., Purdy, A. J., & Caldwell, T. G. (2026). CONUS-AgWeather, a high-quality benchmark daily agricultural weather station dataset for evapotranspiration applications in the Contiguous United States (1.0.1) [Data set]. _Zenodo_. https://doi.org/10.5281/zenodo.20261765.

## Disk Space Requirements

The full repository requires approximately **11 GB** of disk space. Below is a breakdown by directory:

| Directory | Size | Description |
|-----------|------|-------------|
| `Data/` | ~6.9 GB | Input datasets and analysis outputs (including CONUS-AgWeather)|
| `Data.zip` | ~565 MB | Compressed data archive (available from Zenodo) |
| `Plots.zip` | ~849 MB | Compressed plots archive (available from Zenodo) |
| `Plots/` | ~1.0 GB | Generated visualizations (see below) (unzipped Plots.zip) |
| `gridmetbias/` | ~472 KB | Python source code |

## Data Availability

The data required for this project are available from Zenodo:

**CONUS-AgWeather_v1**: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18122157.svg)](https://doi.org/10.5281/zenodo.18122157)

**Input and Output Datasets and Plots for gridMET bias correction**: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18673484.svg)](https://doi.org/10.5281/zenodo.18673484)

Download the data archive and extract its contents into the `Data/` directory. The CONUS-AgWeather_v1 zip archive from Zenodo must also be extracted within `Data/` so that the `CONUS-AgWeather_v1/` directory resides at `Data/CONUS-AgWeather_v1/`. See [Data/README.md](Data/README.md) for detailed instructions on data organization and file descriptions.

## Project Structure

```
gridMET-bias-correction/
├── gridmetbias/                              # Main Python package (see gridmetbias/README.md)
│   ├── corr_analysis_gridmet.py              # Main correlation analysis script
│   ├── biaslibs/                             # Bias correction libraries
│   │   ├── __init__.py
│   │   ├── biasops.py
│   │   └── geeops.py
│   └── scripts/                              # Analysis scripts (see scripts/README.md)
├── Data/                                     # Input datasets (available from Zenodo)
│   ├── README.md                             # Data documentation
│   ├── koppen_ID_info.csv
│   ├── openet_ground_station_master_list_cleaned_v4.csv
│   ├── climateClass_poly_diss/               # Climate classification shapefiles
│   ├── CONUS-AgWeather_v1/                   # QC'd weather station data (~5.7 GB)
│   │   ├── metadata_for_publication.csv
│   │   ├── standardized_data/                # Station Excel files with QC data
│   │   ├── after_qc_composite_graphs/        # Composite graphs after QC
│   │   ├── before_qc_composite_graphs/       # Composite graphs before QC
│   │   ├── log_files/                        # QC processing logs
│   │   └── variable_qc_graphs/               # Variable-specific QC visualizations
│   ├── flux_data/                            # Flux tower reference ET data
│   ├── flux_ET_dataset/                      # Flux ET observations
│   │   ├── daily_data_files/
│   │   ├── monthly_data_files/
│   │   └── station_metadata.xlsx
│   ├── flux_gridmet/                         # Paired flux-gridMET data
│   ├── metadata/                             # Station metadata
│   ├── Outputs/                              # Analysis outputs
│   ├── paired_flux_OpenET_data/              # Merged flux and OpenET data
│   ├── Point_bias_data/                      # Point-level bias data
│   ├── states/                               # US state boundary shapefiles
│   └── supporting_files/                     # Climate/CDL parquet files
├── Plots/                                    # Generated visualizations (~1.0 GB); available from Zenodo
│   ├── Boxplots/                             # Bias boxplots by region/climate
│   ├── Climate/                              # Climate-stratified correlations
│   ├── CONUS-AgWeather_v1_ETo_Stats/         # ETo QC analysis plots
│   ├── CONUS-AgWeather_v1_Var_Stats/         # Variable QC analysis plots
│   ├── Correlation_Plots_All/                # Correlation analysis plots
│   ├── Crop_Bias_Distributions/              # Crop type bias distributions
│   ├── East_vs_West/                         # Regional comparisons
│   ├── GridMET_Plots/                        # GridMET validation plots
│   ├── OpenET_accuracy/                      # OpenET vs flux scatter plots
│   ├── Site_Analysis_GridMET/                # Site-level GridMET analysis
│   ├── Site_Analysis_OpenET/                 # Site-level OpenET analysis
│   ├── Station_Climate/                      # Climate-grouped station plots
│   └── Variable_Maps                         # Station location maps for all CONUS-AgWeather variables
├── CONUS-AgWeather_v1.zip                    # CONUS-AgWeather compressed archive (~2.3 GB); available from Zenodo
├── Data.zip                                  # Compressed data archive (~565 MB); available from Zenodo
├── Plots.zip                                 # Compressed plots archive (~0.9 GB); available from Zenodo
├── LICENSE
└── README.md
```

## Documentation

Detailed documentation is available in each module:

- **[gridmetbias/README.md](gridmetbias/README.md)** - Main package documentation including `corr_analysis_gridmet.py` and `biaslibs/` library
- **[gridmetbias/scripts/README.md](gridmetbias/scripts/README.md)** - Detailed documentation for all analysis and visualization scripts
- **[Data/README.md](Data/README.md)** - Data directory documentation and input requirements
- **[Plots/README.md](Plots/README.md)** - Plots directory documentation and generating scripts

### Data Directory Summary

The `Data/` directory ([available from Zenodo](https://doi.org/10.5281/zenodo.18673484)) contains input datasets and analysis outputs. Key contents:

| Directory | Description | Status |
|-----------|-------------|--------|
| `CONUS-AgWeather_v1/` | QC'd weather station data (~5.7 GB) | [External - Zenodo](https://doi.org/10.5281/zenodo.18122157) |
| `supporting_files/` | Climate/CDL parquet files (~320 MB) | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `flux_ET_dataset/` | Flux tower ET observations | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `Outputs/` | Analysis outputs | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `paired_flux_OpenET_data/` | Merged flux and OpenET data | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `states/` | US state boundaries | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `flux_data/` | GridMET reference ET data | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `Point_bias_data/` | Station-level bias ratio summaries | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `flux_gridmet/` | Paired flux-gridMET data | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `metadata/` | Station metadata | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `climateClass_poly_diss/` | Köppen climate zone shapefiles | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |

See [Data/README.md](Data/README.md) for complete file descriptions and script requirements.

### Plots Directory Summary

The `Plots/` directory (~1.0 GB) contains all generated visualizations. These are **not included in the repository** (available from Zenodo) and are produced by running the analysis scripts in `gridmetbias/` (see [gridmetbias/scripts/README.md](gridmetbias/scripts/README.md) for details).

| Directory | Size | Generating Script | Description | Status |
|-----------|------|--------------------|-------------|--------|
| `Site_Analysis_OpenET/` | ~762 MB | `site_analysis_openet.py` | Site-level OpenET analysis | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `GridMET_Plots/` | ~124 MB | `corr_analysis_gridmet.py` | GridMET validation plots | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `Site_Analysis_GridMET/` | ~80 MB | `site_analysis_gridmet.py` | Site-level GridMET analysis | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `Climate/` | ~21 MB | `corr_analysis_gridmet.py` | Climate-stratified correlations | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `Station_Climate/` | ~18 MB | `station_climate_plots.py` | Climate-grouped distributions | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `Crop_Bias_Distributions/` | ~10 MB | `corr_analysis_gridmet.py` | Crop type bias distributions | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `CONUS-AgWeather_v1_Var_Stats/` | ~9.3 MB | `conus_agweather_var_analysis.py` | Variable QC analysis plots | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `East_vs_West/` | ~8.4 MB | `corr_analysis_gridmet.py` | Regional comparisons | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `Correlation_Plots_All/` | ~2.7 MB | `corr_analysis_gridmet.py` | Correlation analysis plots | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `Boxplots/` | ~2.6 MB | `boxplots_stats.py` | Bias boxplots by region/climate | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `CONUS-AgWeather_v1_ETo_Stats/` | ~2.2 MB | `conus_agweather_eto_analysis.py` | ETo QC analysis plots | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |
| `OpenET_accuracy/` | ~1.3 MB | `OpenET_flux_grouped_scatter_plots.py`, `monthly_climos.py`, `monthly_error_delta_bias_heatmaps.py`, `monthly_ET_vs_ETo_error_scatter.py` | OpenET vs flux accuracy plots | [External - Zenodo](https://doi.org/10.5281/zenodo.18673484) |

## Installation

### 1. Download and Install Anaconda/Miniconda

Either [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is required for managing Python packages (Python >= 3.10 recommended).

**Windows users:** After installation, open Anaconda Prompt and run `conda init powershell` to add conda to PowerShell.

**Linux/Mac users:** Ensure conda is added to your PATH (typically automatic). Restart your shell if needed.

Update conda: `conda update conda`

### 2. Create the Conda Environment

```bash
conda create -y -n gbias python=3.12
conda activate gbias
conda install -y -c conda-forge geopandas seaborn scipy earthengine-api openpyxl scikit-learn pyarrow tqdm
```

### 3. Google Earth Engine Authentication

This project uses the Google Earth Engine (GEE) Python API for geospatial data extraction.

1. Install [Google Cloud CLI](https://cloud.google.com/sdk/docs/install-sdk)
2. Create a GCloud project (e.g., `gee-gbias`) with GEE API enabled at https://console.cloud.google.com/
3. Configure the project:
   ```bash
   gcloud config set project gee-gbias
   gcloud auth application-default set-quota-project gee-gbias  # if prompted
   earthengine authenticate
   ```

See the [Earth Engine Python installation guide](https://developers.google.com/earth-engine/guides/python_install) for details.

## Usage

### Quick Start

```bash
cd gridmetbias/
python corr_analysis_gridmet.py
```

This runs the main analysis which includes:
- Correlation matrix generation (all stations, East/West split, by climate zone)
- GridMET vs. station bias comparison analysis
- Bias distribution plots by irrigation density and crop type
- Generates required files for `site_analysis_gridmet_openet.py` and other scripts

### Analysis Scripts

The `scripts/` directory contains specialized analysis tools. See [gridmetbias/scripts/README.md](gridmetbias/scripts/README.md) for detailed documentation.

| Script | Description |
|--------|-------------|
| `data_formatting.py` | Merge bias data with Köppen climate zones |
| `boxplots_stats.py` | Generate bias boxplots with summary statistics |
| `gen_map.py` | Create station location maps |
| `conus_agweather_eto_analysis.py` | CONUS-AgWeather ETo QC analysis |
| `conus_agweather_var_analysis.py` | CONUS-AgWeather variable (Rs, Rso) analysis |
| `site_analysis_gridmet.py` | Site-level gridMET validation |
| `site_analysis_openet.py` | Site-level OpenET validation |
| `site_analysis_gridmet_openet.py` | Merge gridMET and OpenET datasets |
| `OpenET_flux_grouped_scatter_plots.py` | OpenET vs. flux tower scatter plots |
| `monthly_climos.py` | Monthly climatology analysis |
| `monthly_error_delta_bias_heatmaps.py` | Monthly error/delta/bias heatmaps |
| `monthly_ET_vs_ETo_error_scatter.py` | Monthly ET vs ETo error scatter plots |
| `station_climate_plots.py` | Climate-grouped station visualizations |

To run individual scripts:
```bash
cd gridmetbias/scripts/
python <script_name>.py
```

## Authors

- Dr. John Volk (john.volk@dri.edu) - Desert Research Institute
- Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu) - Desert Research Institute
- Christian Dunkerly (christian.dunkerly@dri.edu) - Desert Research Institute

## License

See [LICENSE](LICENSE) for details.
