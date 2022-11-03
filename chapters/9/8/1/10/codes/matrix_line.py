import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import subprocess
import shlex
def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB

def dir_vec(A,B):
   return B-A
def norm_vec(A,B):
   return np.matmul(omat, dir_vec(A,B))
b = 4
theta = np.pi/3
r = 3
#given points
A = np.array(([0,0]))
B = np.array(([b,0]))
D = np.array(([r*np.cos(theta),r*np.sin(theta)]))
C = B+D 
m = B-D

P = B - (m@(B)/LA.norm(m)**2)*m
Q = B + (m@(C-B)/LA.norm(m)**2)*m

m_1 = A-P
m_2 = P-B
m_3 = A-B
n_1 = C-Q
n_2 = Q-D
n_3 = C-D

f1 = np.linalg.norm(B-P)
f2 = np.linalg.norm(A-B)
e1 = np.linalg.norm(Q-D)
e2 = np.linalg.norm(C-D)
dp1 = np.dot(m_2/f1,m_3/f2)
dp2 = np.dot(n_2/e1,n_3/e2)
an1 = np.arccos(dp1)
an2 = np.arccos(dp2)
#lines generation
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)
x_BD = line_gen(D,B)
x_AP = line_gen(A,P)
x_CQ = line_gen(C,Q)
#plotting all the lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.plot(x_DA[0,:],x_DA[1,:],label='$DA$')
plt.plot(x_BD[0,:],x_BD[1,:],label='$BD$')
plt.plot(x_AP[0,:],x_AP[1,:],label='$AP$')
plt.plot(x_CQ[0,:],x_CQ[1,:],label='$CQ$')

l1 = np.linalg.norm(A-P)
l2 = np.linalg.norm(Q-C)
if (round(l1,4) == round(l2,4)) and (round(an1) == round(an2)) and (abs(round(m_1@m_2)) == abs(round(n_1@n_2))):
   print(" (i) AP = CQ")
   print(" (ii) Triangle APB is congurrent to Triangle DQC")
#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,P,Q)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','P','Q']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x_axis$')
plt.ylabel('$y_axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.title('Parallelogram')
#if using termux
#plt.savefig('/sdcard/Linearalgebra/par.pdf')
#subprocess.run(shlex.split("termux-open '/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-2.pdf'")) 
#else
plt.show()
