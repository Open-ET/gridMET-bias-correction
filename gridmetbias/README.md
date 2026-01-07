# gridMET Bias Analysis Package

This package provides tools for analyzing biases in gridMET reference evapotranspiration (ETo) and related meteorological variables compared to ground-based weather station observations. The analysis supports research on improving gridded climate data accuracy for agricultural and hydrological applications.

## Overview

The gridMET dataset provides high-resolution (~4 km) daily surface meteorological data across the contiguous United States. This package quantifies systematic biases in gridMET variables by comparing them against quality-controlled weather station data, with analyses stratified by:

- Geographic region (East vs. West of 100th meridian)
- Köppen climate classification zones
- Irrigation density and agricultural land use
- Crop type from USDA Cropland Data Layer (CDL)

## Directory Structure

```
gridmetbias/
├── corr_analysis_gridmet.py    # Main analysis script
├── biaslibs/                   # Core library modules
│   ├── __init__.py
│   ├── biasops.py              # Bias analysis and correlation functions
│   └── geeops.py               # Google Earth Engine operations
└── scripts/                    # Visualization and analysis scripts
    └── README.md               # Detailed script documentation
```

---

## Main Script: `corr_analysis_gridmet.py`

**Author:** Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

This is the primary entry point for running the complete bias analysis workflow. It orchestrates multiple analysis tasks including correlation plots, gridMET bias comparisons, and bias distribution visualizations.

### Analyses Performed

1. **Correlation Matrix Plots** - Generates Pearson correlation matrices between monthly ET bias ratios and other meteorological variables:
   - All stations combined
   - East vs. West CONUS (split at 100th meridian)
   - By Köppen climate classification

2. **GridMET Bias Comparison** - Compares gridMET reference ET against flux tower observations with corrected and uncorrected versions

3. **Bias Distribution Plots** - Visualizes bias distributions grouped by irrigation density and crop type

### Input Data Requirements

| Data | Path | Description |
|------|------|-------------|
| Point bias data | `../Data/Point bias data/*.csv` | Station-level bias ratio summaries |
| Climate shapefile | `../Data/climateClass_poly_diss/climateClass_poly_diss.shp` | Köppen climate zones |
| Daily flux data | `../Data/flux_ET_dataset/daily_data_files/` | Daily station observations |
| Monthly flux data | `../Data/flux_ET_dataset/monthly_data_files/` | Monthly station observations |
| GridMET reference | `../Data/flux_data/openet_reference_et_summary_all_sites_bias_corr_paper.csv` | GridMET ETo data |
| Station metadata | `../Data/flux_ET_dataset/station_metadata.xlsx` | Station information |

### Output Directories

| Output | Path |
|--------|------|
| East/West correlation plots | `../Plots/East_vs_West/` |
| Climate correlation plots | `../Plots/Climate/` |
| GridMET comparison plots | `../Plots/GridMET_Plots/` |
| Bias distribution plots | `../Plots/Climate_IrrBias/` |

### Usage

```bash
cd gridmetbias
python corr_analysis_gridmet.py
```

---

## Library Module: `biaslibs/`

The `biaslibs` package contains reusable functions for bias analysis, correlation computations, and geospatial data extraction.

### `biasops.py` - Bias Operations

**Author:** Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

Core functions for bias ratio analysis and correlation plotting.

#### Key Functions

| Function | Description |
|----------|-------------|
| `correlation_matrix_with_pvalues()` | Computes Pearson correlation matrices with statistical significance (p-values) |
| `plot_bias_corr_matrix_all()` | Generates correlation heatmaps for all stations combined |
| `plot_bias_corr_matrix_lon()` | Creates East/West CONUS correlation comparisons |
| `plot_bias_corr_matrix_climate()` | Produces correlation matrices by Köppen climate zone |
| `gridmet_bias_comp_analysis()` | Performs comprehensive gridMET vs. station comparison analysis |
| `plot_irr_crop_bias_distributions()` | Visualizes bias distributions by irrigation density and crop type |

#### Variables Analyzed

- **Reference ET:** ETo, ETr (mm)
- **Temperature:** Tmin, Tmax (°C)
- **Humidity:** Vapor pressure (ea, kPa)
- **Wind:** Wind speed at 2m (u2, m/s)
- **Radiation:** Solar radiation (srad, W/m²)

### `geeops.py` - Google Earth Engine Operations

**Author:** Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)

Functions for extracting geospatial data from Google Earth Engine at station locations.

#### Key Functions

| Function | Description |
|----------|-------------|
| `get_irr_crop_data()` | Extracts irrigation density (IrrMapper, LANID) and crop type (CDL) data at ~4 km gridMET resolution |
| `categorize_irr_ag_fraction()` | Categorizes irrigation and agriculture fractions into low/medium/high classes |
| `fix_cdl_classes()` | Maps CDL crop codes to simplified categories (Corn, Cotton, Soybeans, Wheat, Alfalfa, Other) |

#### Data Sources Used

- **IrrMapper:** `UMT/Climate/IrrMapper_RF/v1_2` - Irrigation mapping for Western US
- **LANID v2:** LANID irrigation dataset (1997-2020)
- **USDA CDL:** `USDA/NASS/CDL` - Cropland Data Layer
- **NLCD:** Annual National Land Cover Database

#### Requirements

- Google Earth Engine account and authentication
- GCloud project ID (default: `ee-grid-obs-comp`)

---

## Scripts Directory

The `scripts/` folder contains specialized analysis and visualization scripts. See [scripts/README.md](scripts/README.md) for detailed documentation of each script, including:

- Data formatting and climate zone merging
- Box plot generation with summary statistics
- Station mapping
- CONUS-AgWeather QC analysis
- OpenET validation scatter plots
- Site-level gridMET and OpenET analysis
- Climate and crop-grouped visualizations

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
openpyxl
```

## Authors

- Dr. Sayantan Majumdar (sayantan.majumdar@dri.edu)
- Dr. John Volk (john.volk@dri.edu)
- Christian Dunkerly (christian.dunkerly@dri.edu)

## License

See [LICENSE](../LICENSE) for details.

## Citations
Volk, J. M., Dunkerly, C., Majumdar, S., Huntington, J. L., Minor, B. A., Kim, Y., Morton, C. G., ReVelle, P., Kilic, A., Melton, F., Allen, R. G., Pearson, C., Purdy, A. J., & Caldwell, T. G. (2026). 
Assessing and Correcting Bias in Gridded Reference Evapotranspiration over Agricultural Lands Across the Contiguous United States. _In prep. for Agricultural Water Management_.

Dunkerly, C., Volk, J. M., Majumdar, S.,  Huntington, J. L., Allen, R. G., Pearson, C., Kim, Y., Morton, C. G., Minor, B. A., ReVelle, P., Kilic, A., Melton, F., Purdy, A. J., & Caldwell, T. G. (2026). 
CONUS-AgWeather, a high-quality benchmark daily agricultural weather station dataset for evapotranspiration applications in the Contiguous United States. _In prep. for Nature Scientific Data. Zenodo_. https://doi.org/10.5281/zenodo.18122156
