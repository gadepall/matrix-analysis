import numpy as np

a = np.array([1, 4, 2])
b = np.array([3, -2, 7])
c = np.array([2, -1, 4])

A = np.block([[a],[b],[c]])

B = np.array([[0],
              [0],
              [15]])

# Ax = B
X = np.linalg.solve(A,B)

print(X)
