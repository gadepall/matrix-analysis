import numpy as np
#For A
A = np.array([3,1,1])
r = (A.T@A)
print("A:",r)

#For B
B = np.array([1,2,3])
r = (B.T@B)
print("B:",r)

#For C
C = np.array([2,-1,-2])
r = (C.T@C)
print("C:",r)

