import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1 = np.array([[-1],[-1],[-1]])
x2 = np.array([[3],[5],[7]])

#Direction vectors
m1 = np.array([[7],[-6],[1]])
m2 = np.array([[1],[-2],[1]])
mat = np.block([m1,m2,x2-x1])
print(mat)
print(np.linalg.matrix_rank(mat))

M = np.block([m1,m2])
print(M)
M1 = M.T @ M
M2 = M.T @(x2-x1)
l1,l2 = np.linalg.solve(M1,M2)

print(l1, -l2)
#Points
A = x1 + (l1)*m1
B = x2 + (-l2)*m2

print(A) 
print(B)
print(np.linalg.norm(A-B))

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
plt.show()
