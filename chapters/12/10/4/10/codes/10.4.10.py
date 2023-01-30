import numpy as np

# Creating two vectors
# We have inserted elements of int type
arr1 = [1, -1, 3]
arr2 = [2, -7, 1]

# Display the vectors
print("Vector 1...\n", arr1)
print("\nVector 2...\n", arr2)

# To compute the cross product of two vectors, use the numpy.cross() method in Python Numpy
# The method returns c, the Vector cross product(s).
print("\nResult...\n",np.cross(arr1, arr2))
