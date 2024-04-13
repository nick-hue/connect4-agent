import numpy as np

# Example arrays for demonstration
hl1_w = np.array([1, 2, 3])
hl1_b = np.array([4, 5, 6])
hl2_w = np.array([7, 8, 9])
hl2_b = np.array([10, 11, 12])
hl3_w = np.array([13, 14, 15])
hl3_b = np.array([16, 17, 18])
hl4_w = np.array([19, 20, 21])
hl4_b = np.array([22, 23, 24])
ol_w = np.array([25, 26, 27])
ol_b = np.array([28, 29, 30])

# Tuple of tuples as you specified
tup = ((hl1_w, hl1_b),
       (hl2_w, hl2_b),
       (hl3_w, hl3_b),
       (hl4_w, hl4_b),
       (ol_w, ol_b))

# Create a dictionary to store arrays with unique keys
arrays_dict = {}
for i, (w, b) in enumerate(tup, start=1):
    arrays_dict[f'hl{str(i)}_w'] = w
    arrays_dict[f'hl{str(i)}_b'] = b

# Save arrays to an .npz file
np.savez('arrays.npz', **arrays_dict)

import numpy as np

# Load the .npz file
data = np.load('arrays.npz')

# Access arrays by their keys
hl1_w = data['hl1_w']
hl1_b = data['hl1_b']
hl2_w = data['hl2_w']
hl2_b = data['hl2_b']
hl3_w = data['hl3_w']
hl3_b = data['hl3_b']
hl4_w = data['hl4_w']
hl4_b = data['hl4_b']
ol_w = data['hl5_w']  # Assuming ol_w was saved with key 'hl5_w'
ol_b = data['hl5_b']  # Assuming ol_b was saved with key 'hl5_b'

# Use the loaded arrays as needed
print("Loaded hl1_w:", hl1_w)
print("Loaded hl1_b:", hl1_b)