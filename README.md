# A physics-aware DL model for hydrological modeling 

**The code will be released soon.**

The code demonstrates the Keras implementation of the hybrid DL model as proposed in paper "***Raising AI systems' awareness for geoscience knowledge: symbiotic integration of physical approaches and deep learning***" (which has been submitted to a journal for peer-reviewing).

Please refer to the file `License.txt` for the license governing this code.

If you have any questions or suggestions with the code or find a bug, please let us know. Contact Shijie Jiang at shijie.jiang(at)hotmail.com

------

## Quick Start

The code was tested with Python 3.6. To use this code, please do:

1. Clone the repo:

   ```sh
   git clone https://github.com/oreopie/physics-aware-dl.git
   cd physics-aware-dl
   ```

2. Install dependencies:

   ```sh
   pip install tensorflow keras matplotlib pandas
   ```

3. Start `Jupyter Notebook` and run the `demo_single_model` locally.

## Overview

- `libs\` -- Library for core NN layers (P-RNN) and other utilities (e.g., CAMELS data pre-processing and model evaluation).
- `demos\` -- The folder contains demo data for running `demo_single_model`.
- `data\` -- Download and unzip CAMELS datasets into this directory.

