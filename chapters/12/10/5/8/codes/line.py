import numpy as np
import matplotlib.pyplot as plt
import os


#Points
A = np.array([[1.0],[-2.0],[-8.0]])
B = np.array([[5.0],[0.0],[-2.0]])
C = np.array([[11.0],[3.0],[7.0]])

M = np.hstack((A,B,C))

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(M[0], M[1], M[2])
ax.plot(M[0], M[1], M[2])
plt.grid()
plt.tight_layout()
plt.savefig('/sdcard/Download/vectors/figs/line_3d.png', dpi=600)
os.system('termux-open /sdcard/Download/vectors/figs/line_3d.png')
