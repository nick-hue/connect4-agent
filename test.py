
import h5py
import numpy as np


# Open the HDF5 file and read data
file = h5py.File('weights.h5', 'r')
print(file)
big_array = file['dense']['deep_model']['dense']  # Load the data into memory
print(big_array)


file.close()