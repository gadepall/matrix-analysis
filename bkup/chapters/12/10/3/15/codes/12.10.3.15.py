import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(0,'/home/pratik/CoordGeo')

from line.funcs import *

A = np.array([[1],[2],[3]])
B = np.array([[-1],[0],[0]])
C = np.array([[0],[1],[2]])

M = np.hstack((A,B,C))

m1 = A -B
m2 = C -B

angle = np.arccos((m1 @ m2)/(np.linalg.norm(m1)*np.linalg.norm(m2)))

print(angle)

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(M[0], M[1], M[2])
plt.grid()

x_AB = line_gen(A,B)

plt.plot(x_AB[0],x_AB[1])

plt.show()
