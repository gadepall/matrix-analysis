import numpy as np

A = np.array([1, 1, 1])
B = np.array([1, 0, 0])
C = np.array([0, 1, 0])
D = np.array([0, 0, 1])
X=((A.T)*B)/(np.linalg.norm(A)*np.linalg.norm(B))
Y=((A.T)*C)/(np.linalg.norm(A)*np.linalg.norm(C))
Z=((A.T)*D)/(np.linalg.norm(A)*np.linalg.norm(D))
U=np.array([X,Y,Z])
print(U)

