import numpy as np

A = np.array([2,3,4])
B = np.array([-1,-2,1])
C = np.array([5,8,7])

mat = np.block([[A],[B],[C]])
print(mat)
print("The Rank of a Matrix: ", np.linalg.matrix_rank(mat))
