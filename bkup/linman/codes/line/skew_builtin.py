#Code by GVV Sharma
#September 21, 2012
#Released under GNU/GPL

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy import linalg as LA

m1 = np.array(([1,-1,1]))
m2 = np.array(([2,1,2]))
A = np.vstack((m1,m2)).T
b=np.array([1,-3,-2])

#SVD, A=USV
U, s, V=LA.svd(A)

#Find size of A
mn=A.shape

#Creating the singular matrix
S = np.zeros(mn)
Sinv = S.T
S[:2,:2] = np.diag(s)

#Verifying the SVD, A=USV
#print(U@S@V)
print(U,s,V)

#Inverting s
sinv = 1./s
#Inverse transpose of S
Sinv[:2,:2] = np.diag(sinv)
#print(Sinv)

#Moore-Penrose Pseudoinverse
Aplus = V.T@Sinv@(U.T)

#Least squares solution
x_ls = Aplus@b
#
print(x_ls)
