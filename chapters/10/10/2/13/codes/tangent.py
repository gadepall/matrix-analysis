import numpy as np
import matplotlib.pyplot as plt
import os

#Generating points on a circle
def circ_gen(O,r):
	len = 1000
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ

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

O = np.array([[0.0],[0.0]])
P = np.array([[-2.0],[0.0]])
A = np.array([[-1.0],[np.sqrt(3.0)]])/2
B = np.array([[-1.0],[-np.sqrt(3.0)]])/2
r = 1
C = circ_gen(O.T,r)
L1=line_gen(O,P)
L2=line_gen(P,A)
L3=line_gen(P,B)
L4=line_gen(O,B)
L5=line_gen(O,A)
plt.plot(C[0],C[1])
plt.plot(L1[0],L1[1],'r')
plt.plot(L2[0],L2[1],'r')
plt.plot(L3[0],L3[1],'r')
plt.plot(L4[0],L4[1],'r')
plt.plot(L5[0],L5[1],'r')
plt.plot(A[0],A[1],'k.')
plt.plot(B[0],B[1],'k.')
plt.plot(O[0],O[1],'k.')
plt.plot(P[0],P[1],'k.')
plt.text(A[0]+1e-3,A[1]+1e-3,'A')
plt.text(B[0]+1e-3,B[1]+1e-3,'B')
plt.text(O[0]+1e-3,O[1]+1e-3,'O')
plt.text(P[0]+4e-3,P[1]+1e-3,'P')
plt.grid()
plt.tight_layout()
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.savefig('../figs/tangent.png')
os.system('termux-open ../figs/tangent.png')
