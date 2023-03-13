import numpy as np
import sys
sys.path.insert(0,'/home/jaswanth/6th-Sem/EE2802/CoordGeo')
from line.funcs import *


A = np.array([2,-1,1])
B = np.array([1,-3,-5])
C = np.array([3,-4,-4])

mat = np.array([A,B,C])
print(mat)
print("The Rank of a Matrix: ", np.linalg.matrix_rank(mat))

# verifing whether the triangle is right angled at A
print((C-A)@(B-A))

# verifing whether the triangle is right angled at B
print((A-B)@(C-B))

# verifing whether the triangle is right angled at C
print((A-C)@(B-C))
