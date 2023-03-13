import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


O = np.array([[0],[0],[0]])
A = np.array([[3],[3],[3]])
X = np.array([[4],[0],[0]])
Y = np.array([[0],[4],[0]])
Z = np.array([[0],[0],[4]])

vec = np.hstack((O,A))
X_ax = np.hstack((O,X))
Y_ax = np.hstack((O,Y))
Z_ax = np.hstack((O,Z))

ax.scatter(vec[0],vec[1],vec[2])
ax.plot(vec[0],vec[1],vec[2])

ax.plot(X_ax[0],X_ax[1],X_ax[2])

ax.plot(Y_ax[0],Y_ax[1],Y_ax[2])

ax.plot(Z_ax[0],Z_ax[1],Z_ax[2])

ax.text(A[0][0],A[1][0],A[2][0],'A (3,3,3)')
ax.text(O[0][0],O[1][0],O[2][0],'O (0,0,0)')

plt.grid()
plt.show()
