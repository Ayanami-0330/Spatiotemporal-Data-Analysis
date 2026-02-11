import numpy as np

# dimensions of data
nx = 64
ny = 64
nt = 15000
shape = (nt, ny, nx, 2)

# load the data
vectors = np.load('vector_64.npy')  # shape (15000, 64, 64, 2)
print("Loaded data with shape:", vectors.shape)