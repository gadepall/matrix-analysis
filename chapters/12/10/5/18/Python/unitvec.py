import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from numpy.linalg import norm
from mpl_toolkits.mplot3d import Axes3D

# Define the vector
e1  = np.array([1,0,0])
e2  = np.array([0,1,0])
e3  = np.array([0,0,1])

# calculate cross product of the vectors
a = np.cross(e2,e3)
b = np.cross(e1,e3)
c = np.cross(e1,e2)

#calculate dot product of the vectors
d = e1@a
e = e2@b
f = e3@c

#print result
print(d+e+f)


# Define the unit vectors
i_unit = np.array([1, 0, 0])
j_unit = np.array([0, 1, 0])
k_unit = np.array([0, 0, 1])


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the unit vectors
ax.quiver(0, 0, 0, *i_unit, color='r', label='i')
ax.quiver(0, 0, 0, *j_unit, color='g', label='j')
ax.quiver(0, 0, 0, *k_unit, color='b', label='k')
ax
# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set axis limits
ax.set_xlim([0, 2])
ax.set_ylim([2,0])
ax.set_zlim([0, 2])
# Set the aspect ratio to be equal
ax.set_aspect('equal')
# Add a legend
ax.legend()

# Show the plot
plt.show()



