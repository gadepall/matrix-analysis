import numpy as np
#For A
A = np.array([2/7, 3/7, 6/7])
r = (A.T@A)
print("Unit of A:",r)

#For B
B = np.array([3/7, -6/7, 2/7])
r = (B.T@B)
print("Unit of B:",r)

#For C
C = np.array([6/7, 2/7, -3/7])
r = (C.T@C)
print("Unit of C:",r)

#Show that they are mutually perpendicular to eatch other

#For A
r = int(A.T@B)
print("Perpendicular of AB:",r)

#For B
r = int(B.T@C)
print("Perpendicular of BC:",r)

#For C
r = int(C.T@A)
print("Perpendicular of CA:",r)
