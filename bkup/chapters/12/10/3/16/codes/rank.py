import numpy as np

#Points
A = np.array([[1.0],[2.0],[7.0]])
B = np.array([[2.0],[6.0],[3.0]])
C = np.array([[3.0],[10.0],[-1.0]])

#Compute rank
r = np.linalg.matrix_rank(np.hstack((A,B,C)))

#Check collinearity
print("Points are", end=" ")
if r < 3:
    print("collinear")
else:
    print("non-collinear")
