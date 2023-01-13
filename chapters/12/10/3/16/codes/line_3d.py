import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use('_mpl-gallery')

#Points
A = np.array([[1.0],[2.0],[7.0]])
B = np.array([[2.0],[6.0],[3.0]])
C = np.array([[3.0],[10.0],[-1.0]])

M = np.hstack((A,B,C))

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.scatter(M[0], M[1], M[2])
ax.plot(M[0], M[1], M[2])
plt.grid()
plt.tight_layout()
plt.savefig('../figs/line_3d.png', dpi=600)
os.system('termux-open ../figs/line_3d.png')
