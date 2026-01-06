# gridMET_ETo
gridMET bias correction analysis

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

## Project Structure

```
gridMET-bias-correction/
├── gridmetbias/                              # Main Python package
│   ├── corr_analysis_gridmet.py              # Main correlation analysis script
│   ├── biaslibs/                             # Bias correction libraries
│   └── scripts/                              # Additional analysis scripts
├── Plots/                                    # Generated visualizations
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
│   ├── Station_CDL/
│   └── Station_Climate/
├── LICENSE
└── README.md
```

## Citations
Volk, J. M., Dunkerly, C., Majumdar, S., Huntington, J. L., Minor, B. A., Kim, Y., Morton, C. G., ReVelle, P., Kilic, A., Melton, F., Allen, R. G., Pearson, C., Purdy, A. J., & Caldwell, T. G. (2026). 
Assessing and Correcting Bias in Gridded Reference Evapotranspiration over Agricultural Lands Across the Contiguous United States. _In prep. for Agricultural Water Management_.

Dunkerly, C., Volk, J. M., Majumdar, S.,  Huntington, J. L., Allen, R. G., Pearson, C., Kim, Y., Morton, C. G., Minor, B. A., ReVelle, P., Kilic, A., Melton, F., Purdy, A. J., & Caldwell, T. G. (2026). 
CONUS-AgWeather, a high-quality benchmark daily agricultural weather station dataset for evapotranspiration applications in the Contiguous United States. _In prep. for Nature Scientific Data. Zenodo_. https://doi.org/10.5281/zenodo.18122156

### 1. Download and install Anaconda/Miniconda
Either [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) is required for installing the Python 3 packages. 
It is recommended to install the latest version of Anaconda or miniconda (Python >= 3.10). If Anaconda or miniconda is already installed, skip this step. 

**For Windows users:** Once installed, open the Anaconda terminal (called Ananconda Prompt), and run ```conda init powershell``` to add ```conda``` to Windows PowerShell path.

**For Linux/Mac users:** Make sure ```conda``` is added to path. Typically, conda is automatically added to path after installation. It may be necessary to restart the current shell session to add conda to path.

The conda package manager can be updated by running the following command: ```conda update conda```

Anaconda is a Python distribution and environment manager. Miniconda is a free minimal installer for conda. These will help in installing the correct packages and Python version to run the codes.


### 2. Setting up the conda environment

```
conda create -y -n gbias python=3.12
conda activate gbias
conda install -y -c conda-forge rioxarray geopandas seaborn scipy earthengine-api openpyxl plotly python-kaleido dask-ml dask-jobqueue tqdm
```

### 3. Google Earth Engine Authentication
This project relies on the Google Earth Engine (GEE) Python API for downloading (and reducing) datasets from the GEE
data repository. The Google Cloud CLI tools are required for GEE authentication. Refer to the installation docs [here](https://cloud.google.com/sdk/docs/install-sdk). 

A GCloud project needs to be set up online (e.g., ```gee-gbias```), with the GEE API service enabled (https://console.cloud.google.com/). Then set a default project using ```gcloud config set project gee-gbias```. Additionally, you may need to run ```gcloud auth application-default set-quota-project gee-gbias``` if prompted by the GCloud CLI. 
After that, run ```earthengine authenticate```. The installation and authentication guide 
for the earth-engine Python API is available [here](https://developers.google.com/earth-engine/guides/python_install). 

### 4. Running the codes

Navigate to the `gridmetbias/` directory to run the analysis scripts:

```bash
cd gridmetbias/
```

#### Main Analysis
Run the primary correlation analysis for gridMET bias correction:
```bash
python corr_analysis_gridmet.py
```

#### Additional Analysis Scripts
The `scripts/` directory contains additional analysis and visualization tools:

| Script | Description |
|--------|-------------|
| `OpenET_flux_grouped_scatter_plots.py` | Generate grouped scatter plots for OpenET flux comparisons |
| `boxplots_stats.py` | Create boxplot visualizations with statistics |
| `conus_agweather_eto_analysis.py` | CONUS AgWeather ETo analysis |
| `conus_agweather_var_analysis.py` | CONUS AgWeather variable analysis |
| `data_formatting.py` | Data formatting utilities |
| `gen_map.py` | Map generation utilities |
| `site_analysis_gridmet.py` | Site-level gridMET analysis |
| `site_analysis_gridmet_openet.py` | Combined gridMET and OpenET site analysis |
| `site_analysis_openet.py` | Site-level OpenET analysis |
| `station_check.py` | Station data quality checks |
| `station_climate_plots.py` | Station climate visualization |
| `station_crop_plots.py` | Station crop type visualization |

To run any of these scripts:
```bash
cd gridmetbias/scripts/
python <script_name>.py
```
