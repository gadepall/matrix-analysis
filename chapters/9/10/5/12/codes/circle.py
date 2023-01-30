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

#Convert a polar point in cartesian coordinates
def polar_to_rect(r,theta):
    return np.array([r*np.cos(np.radians(theta)),r*np.sin(np.radians(theta))])

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

O = np.array([0.0,0.0])
r = 1
C = circ_gen(O,r)
P1=polar_to_rect(r,30)
P2=polar_to_rect(r,150)
P3=polar_to_rect(r,210)
P4=polar_to_rect(r,330)
L1=line_gen(P1,P2)
L2=line_gen(P2,P3)
L3=line_gen(P3,P4)
L4=line_gen(P4,P1)
plt.plot(C[0],C[1])
plt.plot(L1[0],L1[1],'r')
plt.plot(L2[0],L2[1],'r')
plt.plot(L3[0],L3[1],'r')
plt.plot(L4[0],L4[1],'r')
plt.plot(P1[0],P1[1],'k.')
plt.plot(P2[0],P2[1],'k.')
plt.plot(P3[0],P3[1],'k.')
plt.plot(P4[0],P4[1],'k.')
plt.text(P1[0]+1e-3,P1[1]+1e-3,'P$_1$')
plt.text(P2[0]+1e-3,P2[1]+1e-3,'P$_2$')
plt.text(P3[0]+1e-3,P3[1]+1e-3,'P$_3$')
plt.text(P4[0]+4e-3,P4[1]+1e-3,'P$_4$')
plt.grid()
plt.tight_layout()
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.savefig('../figs/circle.png')
os.system('termux-open ../figs/circle.png')
