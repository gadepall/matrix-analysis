import numpy as np
import matplotlib.pyplot as plt
import os

#Points
A = np.array([[1.0],[2.0],[3.0]])
B = np.array([[4.0],[5.0],[6.0]])

#Direction vectors
m1 = np.array([[1.0],[-3.0],[2.0]])
m2 = np.array([[2.0],[3.0],[1.0]])

x = B-A
M = np.hstack((m1, m2))

#Perform svd
U, s, VT = np.linalg.svd(M)
Sig = np.zeros(M.shape)
Sig[:M.shape[1], :M.shape[1]] = np.diag(1/s)

#Get the optimal lambda
lam = VT.T@Sig.T@U.T@x

#Generate line points
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

#Arrays for plotting
Am = A + lam[0]*m1
Bm = B - lam[1]*m2
P = np.hstack((Am,Bm))
P1 = np.hstack((A-2*m1,A+2*m1))
P2 = np.hstack((B-2*m2,B+2*m2))
# Plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.text(Am[0][0],Am[1][0],Am[2][0],'A$_m$')
ax.text(Bm[0][0],Bm[1][0],Bm[2][0],'B$_m$')
ax.plot(P1[0], P1[1], P1[2])
ax.plot(P2[0], P2[1], P2[2])
ax.plot(P[0], P[1], P[2])
ax.scatter(Am[0],Am[1],Am[2])
ax.scatter(Bm[0],Bm[1],Bm[2])
plt.legend(['L1','L2','Normal'])
ax.view_init(60,30)
plt.grid()
plt.tight_layout()
plt.savefig('../figs/skew_svd.png', dpi=600)
os.system('termux-open ../figs/skew_svd.png')
