
# Calculation of cross-product
import numpy as np

A = []
for i in range(3):
    A.append(float(input(f"Enter the a{i+1} for A : ")))
A = np.array(A)

B = []
for i in range(3):
    B.append(float(input(f"Enter the b{i+1} for B : ")))
B = np.array(B)

A23 = np.array([A[1],A[2]])
A31 = np.array([A[2],A[0]])
A12 = np.array([A[0],A[1]])
B23 = np.array([B[1],B[2]])
B31 = np.array([B[2],B[0]])
B12 = np.array([B[0],B[1]])

M = np.array([ [A23,B23],[A31,B31],[A12,B12] ])
D = np.linalg.det(M)
print(D)
