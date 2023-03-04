import numpy as np
import matplotlib.pyplot as plt
import os

#Points
A = np.array([[1],[1],[0]])
B = np.array([[2],[1],[-1]])

#Direction vectors
m1 = np.array([[2],[-1],[1]])
m2 = np.array([[3],[-5],[2]])

x = B-A
M = np.hstack((m1, m2))

#Perform svd
U, s, VT = np.linalg.svd(M)
Sig = np.zeros(M.shape)
Sig[:M.shape[1], :M.shape[1]] = np.diag(1/s)

#Get the optimal lambda
lam = VT.T@Sig.T@U.T@x

print(U)
print(s)
print(VT)
print(lam)