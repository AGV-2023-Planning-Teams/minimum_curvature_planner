# Without access to a CUDA capable GPU, environment variable NUMBA_ENABLE_CUDASIM can be set to 1 to run the code on the CPU
import os
env_var_value = os.getenv('NUMBA_ENABLE_CUDASIM', '0')
if env_var_value == '1':
    import numpy as _cp
else:
    import cupy as _cp

cp = _cp