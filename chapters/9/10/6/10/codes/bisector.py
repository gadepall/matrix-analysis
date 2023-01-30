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

O=np.array([0.0,0.0])
r=1
S=circ_gen(O,r)
b=100
c=200
A=polar_to_rect(r,0)
B=polar_to_rect(r,b)
C=polar_to_rect(r,c)
D=polar_to_rect(r,(b+c)/2)
L1=line_gen(A,B)
L2=line_gen(B,C)
L3=line_gen(A,C)
L4=line_gen(O,D)
L5=line_gen(A,D)
plt.plot(L4[0],L4[1],'r')
plt.plot(L5[0],L5[1],'g')
plt.plot(S[0],S[1])
plt.plot(L1[0],L1[1],'b')
plt.plot(L2[0],L2[1],'b')
plt.plot(L3[0],L3[1],'b')
plt.plot(A[0],A[1],'k.')
plt.plot(B[0],B[1],'k.')
plt.plot(C[0],C[1],'k.')
plt.plot(D[0],D[1],'k.')
plt.plot(O[0],O[1],'k.')
plt.text(A[0]+2e-2,A[1]+2e-2,'A')
plt.text(B[0]+2e-2,B[1]+2e-2,'B')
plt.text(C[0]-8e-2,C[1]-2e-2,'C')
plt.text(D[0]-2e-2,D[1]+2e-2,'D')
plt.text(O[0]+2e-2,O[1]+2e-2,'O')
plt.grid()
plt.tight_layout()
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.savefig('../figs/bisector.png')
os.system('termux-open ../figs/bisector.png')
