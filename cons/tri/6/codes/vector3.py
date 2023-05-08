import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import matplotlib
import subprocess
import shlex
import math
import numpy as np
from numpy import sin, cos, pi, linspace


# Define the coordinates of the vertices
A = [2.3, 8]
B = [1.6, 5.6]
C = [3.9, 2]
D = [5.1, 7.8]

# Define the two lines to be drawn
line1 = [[3.63, 8.07], [3.63, 7.77]]  
line2 = [[2.17, 4.5], [2.32, 4.7]]  



# Zreate a new plot
fig, ax = plt.subplots()

# Plot the triangle with labels
ax.plot([A[0], B[0]], [A[1], B[1]],  label='AB')
ax.plot([B[0], C[0]], [B[1], C[1]],  label='BC')
ax.plot([A[0], C[0]], [A[1], C[1]],  label='CA')
ax.plot([B[0], D[0]], [B[1], D[1]],  label='BD')
ax.plot([A[0], D[0]], [A[1], D[1]],  label='DA')
ax.plot([C[0], D[0]], [C[1], D[1]],  label='DC')
plt.plot([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], 'r-')
plt.plot([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], 'r-')

center = [2.3,8]
radius = 0.5

start_angle = -100
end_angle = -5

angles = np.linspace(start_angle, end_angle, 100)
x_coords = center[0] + radius * np.cos(np.radians(angles))
y_coords = center[1] + radius * np.sin(np.radians(angles))

#fig, ax = plt.subplots()
ax.plot(x_coords, y_coords, 'k-', linewidth=2)

center = [1.6,5.6]
radius = 0.4

start_angle = -55
end_angle = 70

angles = np.linspace(start_angle, end_angle, 100)
x_coords = center[0] + radius * np.cos(np.radians(angles))
y_coords = center[1] + radius * np.sin(np.radians(angles))

#fig, ax = plt.subplots()
ax.plot(x_coords, y_coords, 'k-', linewidth=2)

# Label the vertices
ax.text(A[0], A[1], 'A', ha='center', va='center', fontweight='bold')
ax.text(B[0], B[1], 'B', ha='center', va='center', fontweight='bold')
ax.text(C[0], C[1], 'C', ha='center', va='center', fontweight='bold')
ax.text(D[0], D[1], 'D', ha='center', va='center', fontweight='bold')

# Set the plot limits and title
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_title('Quadrilateral ABCD')

# Xdd legend
ax.legend()
plt.grid(True)

# Show the plot
plt.savefig('/sdcard/arduino/report7/figs/graph.png')
subprocess.run(shlex.split("termux-open /sdcard/arduino/report7/figs/graph.png"))
plt.show()
