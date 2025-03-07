import numpy as np

# 1. Create a 1D array
x1 = np.array([1, 2, 3, 4, 5], dtype='int32') # uint float  

x1_grid = x1.reshape((3,3)) # reshape to 3x3 grid
x1_grid = x1.reshape((3, -1)) # reshape to 3x3 grid with automatic column count

# 2. Create a 2D array
x2 = np.array([[1, 2, 3], 
               [4, 5, 6], 
               [7, 8, 9]])

x2.ndim # number of dimensions 
x2.shape # shape of the array
x2.size # number of elements
x2.dtype # data type of the array
x2.itemsize # size of each element in bytesx2

x2[0, :] # first row
x2[:, 0] # first column
x2[:2, 1:3] # first two rows, second and third columns
x2[0, 0] # first element
x2[0,0] = 99

x2_sub = x2[:2, :2] # subarray
x2_sub[0, 0] = 42 # original array is modified

x2_sub_copy = x2[:2, :2].copy() # copy of the subarray, not a view

x2_linear = x2.reshape(1, -1) # 1D array

# 3. Create a 3D array
x3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# 4. Create an array of zeros
np.zeros((2, 3)) # 2x3 array of zeros
np.zeros((2, 3, 2), dtype='int32') # 2x3x2 array of zeros

# 5. Create an array of ones
np.ones((2, 3)) # 2x3 array of ones

# 6. Create an array of a constant value 
np.full((2, 2), 99) # 2x2 array of 99

# 7. Create an array of random numbers
np.random.rand(4, 2) # 4x2 
np.random.random((2, 2)) # 2x2
np.random.randint(1, 7, size=(3, 3)) # 3x3 array of random integers from 1 to 6
np.random.normal(0, 1, (3, 3)) # min = 0, stdev = 1, 3x3 array of random numbers from normal distribution
np.random.seed(0) # seed for reproducibility

np.arange(1, 10, 2) # min, max, step [1, 3, 5, 7, 9]
np.linspace(0, 10, 5) # min, max, [0, 2.5, 5, 7.5, 10]
np.eye(3) # 3x3 identity matrix
np.empty(5) # 1D array of 5 uninitialized values, fastest

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
np.concatenate([x, y]) 

z = np.array([[7, 8, 9], 
              [10, 11, 12],
              [13, 14, 15]])

h = np.array([[16], 
              [17],
              [18]])

n = np.vstack([x, z]) # vertical stack
m = np.hstack([z, h]) # horizontal stack

np.split(x, [1, 2]) # split at indices 1 and 2

s = np.array([1,2,3,4,5,6,7,8,9])
s1, s2, s3 = np.split(s, [3, 6]) 

s1, s2 = np.vsplit(z, [2]) # vertical split
s1, s2 = np.hsplit(z, [2]) # horizontal split

import array
l = list(range(1, 10))
A = array.array('i', l)
