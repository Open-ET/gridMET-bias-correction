# gridMET Bias Correction and CONUS-AgWeather

This repository contains code, data processing pipelines, and visualization tools for analyzing biases in gridMET reference evapotranspiration (ETo) and developing the CONUS-AgWeather quality-controlled weather station dataset.

## Overview

The gridMET dataset provides high-resolution (~4 km) daily surface meteorological data across the contiguous United States. This project:

1. **Quantifies systematic biases** in gridMET ETo and meteorological variables compared to ground-based weather station observations
2. **Develops bias correction factors** stratified by geographic region, Köppen climate zone, irrigation density, and crop type
3. **Provides the CONUS-AgWeather dataset** - a quality-controlled benchmark of daily agricultural weather station data for ET applications

## Citations

Volk, J. M., Dunkerly, C., Majumdar, S., Huntington, J. L., Minor, B. A., Kim, Y., Morton, C. G., ReVelle, P., Kilic, A., Melton, F., Allen, R. G., Pearson, C., Purdy, A. J., & Caldwell, T. G. (2026). 
Assessing and Correcting Bias in Gridded Reference Evapotranspiration over Agricultural Lands Across the Contiguous United States. _In prep. for Agricultural Water Management_.

Dunkerly, C., Volk, J. M., Majumdar, S.,  Huntington, J. L., Allen, R. G., Pearson, C., Kim, Y., Morton, C. G., Minor, B. A., ReVelle, P., Kilic, A., Melton, F., Purdy, A. J., & Caldwell, T. G. (2026). 
CONUS-AgWeather, a high-quality benchmark daily agricultural weather station dataset for evapotranspiration applications in the Contiguous United States. _Under review in Nature Scientific Data_. https://doi.org/10.31223/X56T9Z. [Datset](https://doi.org/10.5281/zenodo.18122156)

## Disk Space Requirements

The full repository requires approximately **16 GB** of disk space. Below is a breakdown by directory:

| Directory | Size | Description |
|-----------|------|-------------|
| `Data/` | ~12 GB | Input datasets and analysis outputs |
| `Plots/` | ~1.5 GB | Generated visualizations |
| `gridmetbias/` | ~500 KB | Python source code |

### Data Directory Breakdown

| Subdirectory | Size |
|--------------|------|
| `CONUS-AgWeather_v1/` | ~5.7 GB |
| `supporting_files/` | ~1.3 GB |
| `standardized_data/` | ~1.1 GB |
| `paired_flux_OpenET_data/` | ~1.1 GB |
| `flux_ET_dataset/` | ~482 MB |
| `Outputs/` | ~174 MB |
| `Point bias data/` | ~46 MB |
| `states/` | ~42 MB |
| `flux_data/` | ~36 MB |
| `climateClass_poly_diss/` | ~8 MB |

## Data Availability

The data required for this project are available from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18122156.svg)](https://doi.org/10.5281/zenodo.18122156)

Download the data archive and extract its contents into the `Data/` directory. See [Data/README.md](Data/README.md) for detailed instructions on data organization and file descriptions.

## Project Structure

```
gridMET-bias-correction/
├── gridmetbias/                              # Main Python package (see gridmetbias/README.md)
│   ├── corr_analysis_gridmet.py              # Main correlation analysis script
│   ├── biaslibs/                             # Bias correction libraries
│   └── scripts/                              # Analysis scripts (see scripts/README.md)
├── Data/                                     # Input datasets (available from Zenodo)
│   ├── README.md                             # Data documentation
│   ├── koppen_ID_info.csv
│   ├── openet_ground_station_master_list_cleaned_v4.csv
│   ├── climateClass_poly_diss/               # Climate classification shapefiles
│   ├── flux_data/                            # Flux tower data
│   ├── flux_ET_dataset/                      # Flux ET dataset
│   ├── paired_flux_OpenET_data/              # Merged flux and OpenET data
│   ├── Point bias data/                      # Point-level bias data
│   ├── Outputs/                              # Analysis outputs
│   └── supporting_files/                     # Supporting data files
├── Plots/                                    # Generated visualizations
│   ├── README.md                             # Plots documentation
│   ├── Boxplots/
│   ├── Climate/
│   ├── Climate_IrrBias/
│   ├── CONUS-AgWeather_v1_ETo_Stats/
│   ├── CONUS-AgWeather_v1_Var_Stats/
│   ├── Correlation_Plots_All/
│   ├── Crop_Bias_Distributions/
│   ├── East_vs_West/
│   ├── Flux/
│   ├── GridMET_Plots/
│   ├── OpenET_accuracy/
│   ├── Site_Analysis_GridMET/
│   ├── Site_Analysis_OpenET/
│   └── Station_Climate/
├── LICENSE
└── README.md
```

## Documentation

Detailed documentation is available in each module:

- **[gridmetbias/README.md](gridmetbias/README.md)** - Main package documentation including `corr_analysis_gridmet.py` and `biaslibs/` library
- **[gridmetbias/scripts/README.md](gridmetbias/scripts/README.md)** - Detailed documentation for all analysis and visualization scripts
- **[Data/README.md](Data/README.md)** - Data directory documentation and input requirements
- **[Plots/README.md](Plots/README.md)** - Plot directory documentation and generating scripts

### Data Directory Summary

The `Data/` directory contains input datasets and analysis outputs. Key contents:

| Directory | Description | Status |
|-----------|-------------|--------|
| `Point bias data/` | Station-level bias ratio summaries | [External - Zenodo](https://doi.org/10.5281/zenodo.18122156) |
| `flux_data/` | GridMET reference ET data | [External - Zenodo](https://doi.org/10.5281/zenodo.18122156) |
| `flux_ET_dataset/` | Flux tower ET observations | [External - Zenodo](https://doi.org/10.5281/zenodo.18122156) |
| `paired_flux_OpenET_data/` | Merged flux and OpenET data | [External - Zenodo](https://doi.org/10.5281/zenodo.18122156) |
| `climateClass_poly_diss/` | Köppen climate zone shapefiles | [External - Zenodo](https://doi.org/10.5281/zenodo.18122156) |
| `states/` | US state boundaries | [External - Zenodo](https://doi.org/10.5281/zenodo.18122156) |
| `CONUS-AgWeather_v1/` | QC'd weather station data (~5.7 GB) | [External - Zenodo](https://doi.org/10.5281/zenodo.18122156) |
| `supporting_files/` | Climate/CDL parquet files (~1.3 GB) | [External - Zenodo](https://doi.org/10.5281/zenodo.18122156) |

See [Data/README.md](Data/README.md) for complete file descriptions and script requirements.

### Plots Directory Summary

The `Plots/` directory contains all generated visualizations. Key outputs:

| Directory | Generating Script | Description |
|-----------|-------------------|-------------|
| `Boxplots/` | `boxplots_stats.py` | Bias boxplots by region/climate |
| `Climate/` | `corr_analysis_gridmet.py` | Climate-stratified correlations |
| `East_vs_West/` | `corr_analysis_gridmet.py` | Regional comparisons |
| `GridMET_Plots/` | `corr_analysis_gridmet.py` | GridMET validation plots |
| `OpenET_accuracy/` | `OpenET_flux_grouped_scatter_plots.py` | OpenET vs flux accuracy |
| `Site_Analysis_GridMET/` | `site_analysis_gridmet.py` | Site-level GridMET analysis |
| `Site_Analysis_OpenET/` | `site_analysis_openet.py` | Site-level OpenET analysis |
| `Station_Climate/` | `station_climate_plots.py` | Climate-grouped distributions |
| `CONUS-AgWeather_v1_*` | `conus_agweather_*.py` | QC analysis plots |

See [Plots/README.md](Plots/README.md) for complete plot descriptions and contents.

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
conda install -y -c conda-forge rioxarray geopandas seaborn scipy earthengine-api openpyxl plotly python-kaleido dask-ml dask-jobqueue tqdm scikit-learn
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
| `station_climate_plots.py` | Climate-grouped station visualizations |
| `station_crop_plots.py` | Crop type-grouped visualizations (requires GEE) |

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
