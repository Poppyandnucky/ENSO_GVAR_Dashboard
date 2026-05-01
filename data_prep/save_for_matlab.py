# data_prep/save_for_matlab.py
# Helper files used in the debugger to save files as needed

import xarray as xr
import scipy.io
import numpy as np

def save_for_matlab(da, file_name):
    numpy_array = da.values
    data_dict = {'X': numpy_array}
    data_dict['lat'] = da.coords['lat'].values
    data_dict['lon'] = da.coords['lon'].values
    data_dict['t'] = da.coords['time'].values
    scipy.io.savemat('file_name.mat', data_dict)

# Assume you have an xarray DataArray named 'da'
# Example: da = xr.DataArray(np.random.rand(2, 3), dims=("x", "y"), coords={"x": [1, 2], "y": [10, 20, 30]})

# Convert the DataArray to a NumPy array
# numpy_array = da.values

# Create a dictionary where keys will be the variable names in MATLAB
# and values are the NumPy arrays
# data_dict = {'var_name': numpy_array}

# Optional: You can also save coordinates if needed
# data_dict['coord_x'] = my_dataarray.coords['x'].values
# data_dict['coord_y'] = my_dataarray.coords['y'].values

# Save the dictionary to a .mat file
# scipy.io.savemat('output_data.mat', data_dict)
