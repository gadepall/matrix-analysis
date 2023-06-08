import numpy as np
import math
#Points
A = np.array([1,-2,-8])
B = np.array([5,0,-2])
C = np.array([11,3,7])
d=B-A
e=C-B

#Compute rank
r = np.linalg.matrix_rank(np.block([[A],[B],[C]]))

#Check collinearity
print("Points are", end=" ")
if r < 3:
    print("collinear")
else:
    print("non-collinear")

one=np.array((d.T)@e)
q=np.linalg.norm(C-B)
two=int(q*q)
print("B divides AC in ratio ",one/two)
