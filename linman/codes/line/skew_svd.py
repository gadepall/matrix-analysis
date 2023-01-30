#Code by GVV Sharma
#September 21, 2012
#Released under GNU/GPL

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA

#Creating matrices
m1 = np.array(([1,-1,1]))
m2 = np.array(([2,1,2]))
A = np.vstack((m1,m2)).T
b=np.array([1,-3,-2])

#Eigenvalue decompostion of A'A
Dv, Pv=LA.eig(A.T.dot(A))

#Eigenvalue decompostion of AA'
Du, Pu=LA.eig(A.dot(A.T))

#Singular values of A
Stemp=np.sqrt(Dv)

#QR Decomposition to get U and V
V, Rv=LA.qr(Pv)
U, Ru=LA.qr(Pu)

#SVD, A=USV
U_1, s, V_1=LA.svd(A)
print(V)
print(V_1)
print(U)
print(U_1)
print(s)
print(Stemp)
