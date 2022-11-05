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
def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return np.array(([x/z, y/z]))

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
C = B+D-A
E=(A+B)/2
F=(C+D)/2
#P = get_intersect(A,F,B,D)
P=((2*D+B)/3)
#print(P)
#Q = get_intersect(E,C,B,D)
Q=((2*B+D)/3)
print(P,Q)
m_2 = P-B
m_3 = A-B
n_1 = C-Q
n_2 = Q-D
n_3 = C-D

x = np.array([B-D])
a = np.array([B-Q])
b = np.array([Q-P])
c = np.array([P-D])
#print(x,a,b,c)
#lines generation
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)
x_BD = line_gen(D,B)
x_AP = line_gen(A,P)
x_CQ = line_gen(C,Q)
x_PF = line_gen(P,F)
x_QE = line_gen(Q,E)
if((x==a+b+c).all()):
    print("BQ=PQ=PD")
##x_QE=line_gen(Q,E)
#plotting all the lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.plot(x_DA[0,:],x_DA[1,:],label='$DA$')
plt.plot(x_BD[0,:],x_BD[1,:],label='$BD$')
plt.plot(x_AP[0,:],x_AP[1,:],label='$AP$')
plt.plot(x_CQ[0,:],x_CQ[1,:],label='$CQ$')
plt.plot(x_PF[0,:],x_PF[1,:],label='$PF$')
plt.plot(x_QE[0,:],x_QE[1,:],label='$QE$')

l1 = np.linalg.norm(A-P)
l2 = np.linalg.norm(Q-C)
#if (length(BD)==BQ+PQ+PD):
#   print("BQ=PQ=PD")
#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,P,Q,E,F)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','P','Q','E','F']
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
plt.savefig('/home/apiiit-rkv/Desktop/line.pdf')
#subprocess.run(shlex.split("termux-open '/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-2.pdf'")) 
#else
plt.show()
