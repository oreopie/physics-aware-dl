# A hydrology-aware DL model for runoff modeling
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.3856486-blue.svg)](https://doi.org/10.5281/zenodo.3856486)
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Tips on the regional model](#tips-on-the-regional-model)

## Overview
The code demonstrates the Keras implementation of the hybrid DL model as proposed in paper "***Improving AI system awareness of geoscience knowledge: Symbiotic integration of physical approaches and deep learning***"  published in *Geophysical Research Letters*. [[link to the paper]](https://doi.org/10.1029/2020GL088229)

Please refer to the file `License.txt` for the license governing this code.

If you use this repository in your work, please cite:

> **Jiang S., Zheng Y., & Solomatine D.. (2020) Improving AI system awareness of geoscience knowledge: Symbiotic integration of physical approaches and deep learning. *Geophysical Research Letters*, 47. DOI: 10.1029/2020GL088229**

If you have any questions or suggestions with the code or find a bug, please let us know. You are welcome to [raise an issue here](https://github.com/oreopie/physics-aware-dl/issues) or contact Shijie Jiang at *jiangsj(at)mail.sustech.edu.cn*

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
   pip install numpy==1.16.4 pandas scipy tensorflow==1.14 keras==2.3.1 matplotlib jupyter scikit-learn
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
------

## Tips on the regional model

To implement the regional model (main pipeline + parameterization pipeline) as developed in the study, we provide `RegionalPRNNLayer`, `RegionalConv1Layer`, and `RegionalConv2Layer` classes in the `libs\hydrolayer.py`. Below is a demo to use the classes for buiding the regional model:

   ```python
from libs.hydrolayer import RegionalPRNNLayer, RegionalConv1Layer, RegionalConv2Layer, ScaleLayer

x_forcing = Input(shape=train_forcing[0].shape[], name='Input_forcing')
x_attrs   = Input(shape=train_attrs[0].shape, name='Input_attrs')
hydro = RegionalPRNNLayer(h_nodes=32, seed=200, name='RegionalPRNN')([x_forcing, x_attrs])

x_forcing_scaled = ScaleLayer(name='ScaleForcing')(x_forcing)
x_new = Concatenate(axis=-1, name='ConcatBW')([x_forcing_scaled, hydro])

conv1 = RegionalConv1Layer(h_nodes=8, seed=200, name='RegionalConv1')([x_new, x_attrs])
conv2 = RegionalConv2Layer(h_nodes=8, seed=200, name='RegionalConv2')([conv1, x_attrs])

model = Model([x_forcing, x_attrs], conv2)
   ```

Please note:
1. `x_forcing` represents the meteorological time sequences with a shape of `[sample size, sequence length, 5]`, where the first three variables should be *precipitation*, *mean temperature*, and *day length*.
2. `x_attrs` represents the catchment attributes with a shape of `[sample size, 27]`, where the attribute values must be scaled in advance (I recommend using the standardization).
3. Slightly different from the conceptual diagram in the paper, we merge the parameterization into respective layers directly in the implementation. You can also separate each merged layer into two layers by treating the generated `theta_p` and `theta_n` as explicit variables.
4. Please read carefully the help notes in each developed layer if you would like to adapt the architectures for your research.
