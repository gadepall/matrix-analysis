import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
## computation for c
n= np.array([[0],[1],[0]])
x= np.array([[0],[3],[0]])
c= n.T@x
print(c)
def plot_plane(y_intercept, color):
    x = np.linspace(-5, 5, 50)
    z = np.linspace(-5, 5, 50)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X) + y_intercept
    ax.plot_surface(X, Y, Z, color=color, alpha=0.5)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_plane(3, 'red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)
plt.show()

