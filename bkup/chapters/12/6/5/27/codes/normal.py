import numpy as np
import matplotlib.pyplot as plt
import os

#Generating points on a parabola
def parab_gen(x,a):
	y = x**2/a
	return y

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

X = np.linspace(-4,4,10000)
a = 2
P = np.array([[0.0],[5.0]])
N1 = np.array([[0.0],[0.0]])
N2 = np.array([[2*np.sqrt(2)],[4.0]])
N3 = np.array([[-2*np.sqrt(2)],[4.0]])
L1 = line_gen(P,N1)
L2 = line_gen(P,N2)
L3 = line_gen(P,N3)
plt.plot(X,parab_gen(X,a))
plt.plot(L1[0],L1[1],'r')
plt.plot(L2[0],L3[1],'r')
plt.plot(L3[0],L2[1],'r')
plt.plot(P[0],P[1],'k.')
plt.plot(N1[0],N1[1],'k.')
plt.plot(N2[0],N2[1],'k.')
plt.plot(N3[0],N3[1],'k.')
plt.text(P[0],P[1],'P')
plt.text(N1[0],N1[1],'N$_1$')
plt.text(N2[0],N2[1],'N$_2$')
plt.text(N3[0],N3[1],'N$_3$')
plt.grid()
plt.tight_layout()
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.savefig('../figs/normal.png')
os.system('termux-open ../figs/normal.png')
