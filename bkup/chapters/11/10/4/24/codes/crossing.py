import numpy as np
import matplotlib.pyplot as plt
import os

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

#Return unit vector
def unit_vec(A):
    return A/np.linalg.norm(A)

#Orthogonal matrix
omat = np.array([[0.0, 1.0],[-1.0, 0.0]])

#intersection of two lines
def line_intersect(n1,c1,n2,c2):
  n=np.vstack((n1.T,n2.T))
  p = np.array([[c1],[c2]])
  #intersection
  p=np.linalg.inv(n)@p
  return p

#Intersection of two lines
def perp_foot(n,cn,P):
  m = omat@n
  cm = (m.T@P)[0][0]
  return line_intersect(n,cn,m,cm)

#Lines
n1 = np.array([[2.0],[-3.0]])
n2 = np.array([[3.0],[4.0]])
n3 = np.array([[6.0],[-7.0]])
c1 = -4.0
c2 = 5.0
c3 = -8.0

#Intersection point
A = line_intersect(n1,c1,n2,c2)
#Foot of perpendicular
F = perp_foot(n3,c3,A)

#Line plots
u1 = 0.5*unit_vec(omat@n1)
u2 = 0.5*unit_vec(omat@n2)
u3 = 0.5*unit_vec(omat@n3)
u4 = 0.5*unit_vec(n3)
L1 = line_gen(A-u1,A+u1)
L2 = line_gen(A-u2,A+u2)
L3 = line_gen(F-u3,F+u3)
L4 = line_gen(A-u4,F+u4)
plt.plot(L1[0],L1[1])
plt.plot(L2[0],L2[1])
plt.plot(L3[0],L3[1])
plt.plot(L4[0],L4[1])
plt.plot(A[0][0],A[1][0],'k.')
plt.text(A[0][0]-0.03,A[1][0],'A')
plt.plot(F[0][0],F[1][0],'k.')
plt.text(F[0][0]+0.03,F[1][0],'F')
plt.grid()
plt.tight_layout()
plt.legend(['$2x - 3y = -4$', '$3x + 4y = 5$', '$6x - 7y = -8$', '$7x + 6y = \\frac{125}{17}$'])
plt.savefig('../figs/crossing.png')
os.system('termux-open ../figs/crossing.png')
