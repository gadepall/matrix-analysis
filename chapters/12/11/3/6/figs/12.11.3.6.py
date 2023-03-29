import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


p1= np.array([1,1,-1])
p2= np.array([6,4,-5])
p3= np.array([-4,-2,3])


# Calculate the normal vector of the plane
v1 = p3 - p1
v2 = p2 - p1
normal = np.cross(v1, v2)

# Define the plane function
def plane(x, y):
    return (-normal[0] * x - normal[1] * y - np.dot(normal, p1)) * 1. / normal[2]

# Generate the meshgrid
xx, yy = np.meshgrid(range(-10, 11), range(-10, 11))

# Evaluate the plane function
z = plane(xx, yy)

# Plot the plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, z)
ax.scatter(p1[0], p1[1], p1[2], color='red')
ax.scatter(p2[0], p2[1], p2[2], color='red')
ax.scatter(p3[0], p3[1], p3[2], color='red')
plt.show()
