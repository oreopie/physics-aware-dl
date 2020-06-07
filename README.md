# A hydrology-aware DL model for runoff modeling
[![DOI](https://zenodo.org/badge/241771877.svg)](https://zenodo.org/badge/latestdoi/241771877)

The code demonstrates the Keras implementation of the hybrid DL model as proposed in paper "***Improving AI system awareness of geoscience knowledge: Symbiotic integration of physical approaches and deep learning***"  published in *Geophysical Research Letters*. [[link to paper]]()

Please refer to the file `License.txt` for the license governing this code.

If you use this repository in your work, please cite:

> **Jiang S., Zheng Y., and Solomatine D. (2020) Improving AI system awareness of geoscience knowledge: Symbiotic integration of physical approaches and deep learning. (accepted).**

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

## About the `PRNNLayer`

As the core NN layer for wrapping the representations of geosystem dynamics into the DL model,  the proposed `PRNNLayer` is implemented as a custom Keras layer following the instructions noted in the official document:  [writing your own keras layers](https://keras.io/layers/writing-your-own-keras-layers/).

The `PRNNLayer` is a nerual network wrapper for the [EXP-HYDRO model](https://github.com/sopanpatil/exp-hydro), which involves 3 input variables (***P***, ***T***,  and ***L_day***), 2 state variables (***S_0***  and  ***S_1***), 6 process variables (***P_s***, ***P_r***, ***M***, ***ET***, and ***Q***) and 6 parameters to be estimated (***T_min***, ***T_max***, ***D_f***, ***S_max***, ***f***, and ***Q_max***). The desired output variable is ***Q***.

### Usage

```python
hydro = PRNNLayer(name='Hydro')(x_input) 
# x_input denotes meteorological data [prcp, tmean, dayl] with a shape of (sample size, sequence length, 3)
# hydro denotes the calculated runoff with a shape of (sample size, sequence length, 1)
```

### (Pseudo)code

The (pseudo)code for the `PRNNLayer` is as following. The recursive part is implemented by `rnn`, which is used for iterateing over the time dimension of a tensor and provided by `tensorlfow` backend. See more details in [keras.backend.rnn](https://keras.io/backend/#rnn).

```python
from keras.layers import Layer
from keras import backend as K

class PRNNLayer(Layer):
    def __init__(self, **kwargs):
        super(PRNNLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create the 6 trainable parameters (based on the EXP-HYDRO model) for this layer:
        self.f, self.smax, self.qmax, self.ddf, self.tmax, self.tmin
        super(PRNNLayer, self).build(input_shape)
        
    def step_do(self, step_in, states):  # Define step function for the RNN
      	# 1. Load the current state variables in this step
        s0 = states[0][:, 0:1]  # Snow bucket
        s1 = states[0][:, 1:2]  # Soil bucket
        
        # 2. Load the current input variables in the step
      	p = step_in[:, 0:1]
        t = step_in[:, 1:2]
        pet = step_in[:, 2:3]
        
        # 3. Calculate current process variables based on the 2 state variables, 3 input
        #    variables, and 6 trainable paramters:
        _ps, _pr, _m, _et, _q
        
        # 4. Implement water balance equation to calculate states in the next step:
        next_s0 = s0 + _ps - _m
        next_s1 = s1 + _m - _et - _q
        step_out = K.concatenate([next_s0, next_s1], axis=1)
        
        return step_out
      
    def call(self, x):
        # 1. Load the input vector
        prcp = inputs[:, :, 0:1]
        tmean = inputs[:, :, 1:2]
        dayl = inputs[:, :, 2:3]
        
        # 2. Calculate PET using Hamon’s formulation
        pet = hamon_fun(dayl, tmean)

        # 3. Concatenate prcp, tmean, and pet into a new input
        new_inputs = K.concatenate((prcp, tmean, pet), axis=-1)

        # 4. Define 2 initial state variables at the beginning
        init_states = [K.zeros((K.shape(new_inputs)[0], 2))]

        # 5. Recursively calculate all state variables by using RNN
        _, outputs, _ = K.rnn(self.step_do, new_inputs, init_states)
        s0 = outputs[:, :, 0:1]
        s1 = outputs[:, :, 1:2]

        # 6. Calculate final process variables
        q = soil_bucket(s1, pet, self.f, self.smax, self.qmax)
        
        return q

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)
```
