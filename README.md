# gridMET_ETo
gridMET bias correction analysis

## Citations
Volk, J. M., Dunkerly, C., Majumdar, S., Huntington, J. L., Minor, B. A., Kim, Y., Morton, C. G., Kilic, A., Melton, F., Allen, R. G., Pearson, C., & Purdy, A. J. (2025). 
Assessing and Correcting Bias in Gridded Reference Evapotranspiration over Agricultural Lands Across the Contiguous United States. _Under review in Agricultural Water Management_.

Dunkerly, C., Volk, J. M., Majumdar, S.,  Huntington, J. L., Allen, R. G., Pearson, C., Kim, Y., Morton, C. G., Minor, B. A., Kilic, A., Melton, F., & Purdy, A. J. (2025). 
CONUS-AgWeather: A High-Quality Benchmark Daily Agricultural Weather Station Dataset for Meteorological Applications in the Contiguous United States. _Under review in Nature Scientific Data_.

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
conda install -y -c conda-forge rioxarray geopandas seaborn scipy earthengine-api openpyxl plotly python-kaleido dask-ml dask-jobqueue
```

### 3. Google Earth Engine Authentication
This project relies on the Google Earth Engine (GEE) Python API for downloading (and reducing) datasets from the GEE
data repository. The Google Cloud CLI tools are required for GEE authentication. Refer to the installation docs [here](https://cloud.google.com/sdk/docs/install-sdk). 

A GCloud project needs to be set up online (e.g., ```gee-gbias```), with the GEE API service enabled (https://console.cloud.google.com/). Then set a default project using ```gcloud config set project gee-gbias```. Additionally, you may need to run ```gcloud auth application-default set-quota-project gee-gbias``` if prompted by the GCloud CLI. 
After that, run ```earthengine authenticate```. The installation and authentication guide 
for the earth-engine Python API is available [here](https://developers.google.com/earth-engine/guides/python_install). 

### 4. Running the codes
```
cd gridmetbias/
python corr_analysis_gridmet.py

cd gridmetbias/scripts
python OpenET_flux_grouped_scatter_plots.py
```
