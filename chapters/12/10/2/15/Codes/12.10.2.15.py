import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

P = np.array([[1],[2],[-1]])
Q = np.array([[-1],[1],[1]])
R = (2*P + Q)/3
print(R)

fig=plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(-4,2)
ax.set_ylim(-1,2)
ax.set_zlim(-1,4)

ax.scatter(P[0],P[1],P[2])
ax.text(P[0][0],P[1][0],P[2][0],'%s' %'P', size= 10, zorder=1 )
ax.scatter(Q[0],Q[1],Q[2])
ax.text(Q[0][0],Q[1][0],Q[2][0], '%s'% 'Q', size = 10, zorder=1)
ax.scatter(R[0],R[1],R[2])
ax.text(R[0][0],R[1][0],R[2][0],'%s' %'R_internal', size =10, zorder=1)

S = 2*Q -P

ax.scatter(S[0],S[1],S[2])
ax.text(S[0][0],S[1][0],S[2][0], '%s' % 'R_external', size =10, zorder =1 )
print(S)

ax.set_title("Figure 1")

plt.show()
