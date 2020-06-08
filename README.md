# A hydrology-aware DL model for runoff modeling
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.3856486-blue.svg)](https://doi.org/10.5281/zenodo.3856486)

The code demonstrates the Keras implementation of the hybrid DL model as proposed in paper "***Improving AI system awareness of geoscience knowledge: Symbiotic integration of physical approaches and deep learning***"  published in *Geophysical Research Letters*. [[link to paper]]()

Please refer to the file `License.txt` for the license governing this code.

If you use this repository in your work, please cite:

> **Jiang S., Zheng Y., and Solomatine D. (2020) Improving AI system awareness of geoscience knowledge: Symbiotic integration of physical approaches and deep learning. *Geophysical Research Letters* (accepted).**

If you have any questions or suggestions with the code or find a bug, please let us know. Contact Shijie Jiang at *jiangsj(at)mail.sustech.edu.cn*

------

## Quick Start

The code was tested with Python 3.6. To use this code, please do:

1. Clone the repo:

   ```shell
   git clone https://github.com/oreopie/physics-aware-dl.git
   cd physics-aware-dl
   ```

2. Install dependencies:

   ```shell
   pip install numpy==1.16.4 pandas scipy tensorflow==1.14 keras matplotlib jupyter
   ```

   Note that the latest version of `tensorflow` is `2.0`, while the **core NN layers (P-RNN)** is built under `tensorflow 1.x`. For this implementation, `tensorflow v1.14` is recommended.

3. Download CAMELS (Catchment Attributes and Meteorology for Large-sample Studies) data set  `CAMELS time series meteorology, observed flow, meta data (.zip) `  from [https://ral.ucar.edu/solutions/products/camels](https://ral.ucar.edu/solutions/products/camels). Unzip `basin_timeseries_v1p2_metForcing_obsFlow.zip` and reorganize the directory as follows,

   ```
   camels\
   |---basin_mean_forcing\
   |   |---daymet\
   |       |---01\
   |       |---...	
   |       |---18	\
   |---usgs_streamflow\
       |---01\
       |---...	
       |---18\
   ```

4. Start `Jupyter Notebook` and run the `demo_single.ipynb` locally.
