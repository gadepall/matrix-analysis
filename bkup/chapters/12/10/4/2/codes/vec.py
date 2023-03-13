import numpy as np

# Creating two vectors
# We have inserted elements of int type
arr1 = np.array([3, 2, 2])
arr2 = np.array([1, 2, -2])
arr3 = np.array([0.083,0.083,0.083])
arr4 = np.array([8,-8,-4])

# Display the vectors
print("Vector 1...\n", arr1)
print("\nVector 2...\n", arr2)
print("\nVector 3...\n", arr3)
print("\nVector 4...\n", arr4)
# To compute the cross product of two vectors, use the numpy.cross() method in Python Numpy
# The method returns c, the Vector cross product(s).
print("\nResult...\n",np.cross(arr1, arr2))
print("\nResult...\n",np.cross(arr3, arr4))

