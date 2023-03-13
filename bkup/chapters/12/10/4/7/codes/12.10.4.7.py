import numpy as np

#define vectors
A = np.array([2, 3, 1])
B = np.array([1, 2, 1])
C= np.array([1,1,1])
S=B+C
print('LHS=',np.cross(A, S))
ab=np.cross(A, B)
ac=np.cross(A, C)
Z=ab+ac
print('RHS=',Z)
