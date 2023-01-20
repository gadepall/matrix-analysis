import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(0,'/home/pratik/CoordGeo')

from line.funcs import *

A = np.array([[3],[5],[-4]])
B = np.array([[-1],[1],[2]])
C = np.array([[-5],[-5],[-2]])

X = np.array([[1],[0],[0]])
Y = np.array([[0],[1],[0]])
Z = np.array([[0],[0],[1]])

Axes= np.block([X,Y,Z])


m = np.block([A-B,B-C,C-A])

cosines = np.zeros([3,3])

for i in range(3):
    for j in range(3):
        cosines[i][j] = (m[i].T @ Axes[j])/(np.linalg.norm(m[i])*np.linalg.norm(Axes[j]))

        
print(cosines)

fig = plt.figure()
ax =fig.add_subplot(projection='3d')

S1 = np.hstack((A,B))
S2 = np.hstack((B,C))
S3 = np.hstack((C,A))

ax.scatter(S1[0],S1[1],S1[2])
ax.plot(S1[0],S1[1],S1[2])

ax.scatter(S2[0],S2[1],S2[2])
ax.plot(S2[0],S2[1],S2[2])

ax.scatter(S3[0],S3[1],S3[2])
ax.plot(S3[0],S3[1],S3[2])

ax.text(A[0][0],A[1][0],A[2][0],'A')
ax.text(B[0][0],B[1][0],B[2][0],'B')
ax.text(C[0][0],C[1][0],C[2][0],'C')

plt.show()
