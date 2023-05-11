import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import shlex


# Define the coordinates of the vertices
A = [1, 1]
B = [4, 1]
C = [3, 4]

P = [6,1]
Q = [9,1]
R = [8,4]

# Define the two lines to be drawn
line1 = [[2.17, 2.55], [1.98, 2.7]]  # First line from A to B
line2 = [[2.59, 1.18], [2.59, 0.89]]  # Second line from B to C
line3 = [[2.21, 2.65], [2.03, 2.81]] #Third line from A to B
line4 = [[3.286, 2.683], [3.548, 2.792]]  # Fourth line from A to C
line5 = [[3.326, 2.602], [3.588, 2.711]]  # Fifth line from A to C
line6 = [[3.366, 2.494], [3.588, 2.602]] # Sixth line from A to C
line7 = [[7.58, 1.14], [7.58, 0.898]] # Sixth line from P to Q
line8 = [[6.934, 2.683], [7.135, 2.386]]
line9 = [[7.016, 2.765], [7.177, 2.495]]
line10 = [[8.286, 2.683], [8.548, 2.792]]
line11 = [[8.326, 2.602], [8.588, 2.711]]  
line12 = [[8.366, 2.494], [8.588, 2.602]]

# Create a new plot
fig, ax = plt.subplots()

# Plot the triangle with labels
ax.plot([A[0], B[0]], [A[1], B[1]], 'k', label='AB')
ax.plot([B[0], C[0]], [B[1], C[1]], 'k', label='BC')
ax.plot([C[0], A[0]], [C[1], A[1]], 'k', label='CA')
ax.plot([P[0], Q[0]], [P[1], Q[1]], 'k', label='PQ')
ax.plot([Q[0], R[0]], [Q[1], R[1]], 'k', label='QR')
ax.plot([R[0], P[0]], [R[1], P[1]], 'k', label='RP')
plt.plot([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], 'r-')  # First line
plt.plot([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], 'g-')  # Second line
plt.plot([line3[0][0], line3[1][0]], [line3[0][1], line3[1][1]], 'r-')  # Third line
plt.plot([line4[0][0], line4[1][0]], [line4[0][1], line4[1][1]], 'b-')  # First line
plt.plot([line5[0][0], line5[1][0]], [line5[0][1], line5[1][1]], 'b-')  # Second line
plt.plot([line6[0][0], line6[1][0]], [line6[0][1], line6[1][1]], 'b-')  # Third line
plt.plot([line7[0][0], line7[1][0]], [line7[0][1], line7[1][1]], 'g-')
plt.plot([line8[0][0], line8[1][0]], [line8[0][1], line8[1][1]], 'r-')
plt.plot([line9[0][0], line9[1][0]], [line9[0][1], line9[1][1]], 'r-')
plt.plot([line10[0][0], line10[1][0]], [line10[0][1], line10[1][1]], 'b-')
plt.plot([line11[0][0], line11[1][0]], [line11[0][1], line11[1][1]], 'b-')
plt.plot([line12[0][0], line12[1][0]], [line12[0][1], line12[1][1]], 'b-')


# Label the vertices
ax.text(A[0], A[1], 'A', ha='center', va='center', fontweight='bold')
ax.text(B[0], B[1], 'B', ha='center', va='center', fontweight='bold')
ax.text(C[0], C[1], 'C', ha='center', va='center', fontweight='bold')
ax.text(P[0], P[1], 'P', ha='center', va='center', fontweight='bold')
ax.text(Q[0], Q[1], 'Q', ha='center', va='center', fontweight='bold')
ax.text(R[0], R[1], 'R', ha='center', va='center', fontweight='bold')

# Set the plot limits and title
ax.set_xlim([0, 10])
ax.set_ylim([0, 6])
ax.set_title('')

# Add legend
ax.legend()
#ax.add_patch(angle_plot) 

# Show the plot
plt.savefig('/sdcard/arduino/report7/figs/graph.png')
subprocess.run(shlex.split("termux-open /sdcard/arduino/report7/figs/graph.png"))
plt.show()
