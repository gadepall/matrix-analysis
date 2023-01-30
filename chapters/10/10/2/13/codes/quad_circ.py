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

#intersection of two lines
def line_intersect(n1,c1,n2,c2):
  n=np.vstack((n1.T,n2.T))
  p = np.array([[c1],[c2]])
  #intersection
  p=np.linalg.inv(n)@p
  return p

O = np.array([[0.0],[0.0]])
A = np.array([[-2.0],[0.0]])
C = np.array([[1.0],[1.0]])
r = 1
n1 = np.array([[1.0],[np.sqrt(3)]])
n2 = np.array([[1.0],[-np.sqrt(3)]])
m1 = np.array([[2.0],[0.0]])
m2 = np.array([[0.0],[2.0]])
H = -n1/np.linalg.norm(n1)
F = m1/np.linalg.norm(m1)
G = m2/np.linalg.norm(m2)
E = -n2/np.linalg.norm(n2)
B = line_intersect(n1,np.sum(A.T@n1),m1,np.sum(C.T@m1))
D = line_intersect(n2,np.sum(A.T@n2),m2,np.sum(C.T@m2))
#Plot
P = circ_gen(O.T,r)
L1=line_gen(O,A)
L2=line_gen(O,B)
L3=line_gen(O,C)
L4=line_gen(O,D)
AB=line_gen(A,B)
BC=line_gen(B,C)
CD=line_gen(C,D)
DA=line_gen(D,A)
plt.plot(P[0],P[1])
plt.plot(L1[0],L1[1],'r')
plt.plot(L2[0],L2[1],'r')
plt.plot(L3[0],L3[1],'r')
plt.plot(L4[0],L4[1],'r')
plt.plot(AB[0],AB[1],'g')
plt.plot(BC[0],BC[1],'g')
plt.plot(CD[0],CD[1],'g')
plt.plot(DA[0],DA[1],'g')
plt.plot(A[0],A[1],'k.')
plt.plot(B[0],B[1],'k.')
plt.plot(C[0],C[1],'k.')
plt.plot(D[0],D[1],'k.')
plt.plot(E[0],E[1],'k.')
plt.plot(F[0],F[1],'k.')
plt.plot(G[0],G[1],'k.')
plt.plot(H[0],H[1],'k.')
plt.plot(O[0],O[1],'k.')
plt.text(A[0]+1e-3,A[1]+1e-3,'A')
plt.text(B[0]+1e-3,B[1]+1e-3,'B')
plt.text(C[0]+1e-3,C[1]+1e-3,'C')
plt.text(D[0]+1e-3,D[1]+1e-3,'D')
plt.text(E[0]+1e-3,E[1]+1e-3,'E')
plt.text(F[0]+1e-3,F[1]+1e-3,'F')
plt.text(G[0]+1e-3,G[1]+1e-3,'G')
plt.text(H[0]+1e-3,H[1]+1e-3,'H')
plt.text(O[0]+1e-3,O[1]+1e-3,'O')
plt.grid()
plt.tight_layout()
ax = plt.gca()
ax.set_aspect('equal', 'box')
plt.savefig('../figs/quad_circ.png')
os.system('termux-open ../figs/quad_circ.png')
