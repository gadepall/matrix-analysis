import numpy as np
import matplotlib.pyplot as plt
import os

#Points
A = np.array([[29.0],[8.0],[77.0]])/19
B = np.array([[20.0],[11.0],[86.0]])/19

#Direction vectors
m1 = np.array([[1.0],[-3.0],[2.0]])
m2 = np.array([[2.0],[3.0],[1.0]])

#Arrays for plotting
M = np.hstack((A-2*m1,A+2*m1))
N = np.hstack((B-2*m2,B+2*m2))
P = np.hstack((A,B))
# Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(M[0], M[1], M[2])
ax.plot(N[0], N[1], N[2])
ax.plot(P[0], P[1], P[2])
ax.scatter(M[0], M[1], M[2])
ax.scatter(N[0], N[1], N[2])
ax.scatter(P[0], P[1], P[2])
ax.text(A[0][0],A[1][0],A[2][0],'A')
ax.text(B[0][0],B[1][0],B[2][0],'B')
plt.legend(['L1','L2','Normal'])
ax.view_init(60,30)
plt.grid()
plt.tight_layout()
plt.savefig('../figs/skew.png', dpi=600)
os.system('termux-open ../figs/skew.png')
