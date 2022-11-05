
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import subprocess
import shlex
import math
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

def line_intersect(n1,A1,n2,A2):
  N=np.vstack((n1,n2))
  print(type(N))
  p = np.zeros(2)
  p[0] = n1@A1
  p[1] = n2@A2
  #Intersection
  P=np.linalg.inv(N)@p
  P = np.array(([P]))
  return P
r = 5
theta = np.pi/3


#Given points
A = np.array(([r*np.cos(theta),r*np.sin(theta)]))
B = np.array(([0,0]))
C = np.array(([r,0]))
D = np.array(([r/2,0]))
E = np.array(([r/2*np.cos(theta),-r/2*np.sin(theta)]))
#F = get_intersect(A,E,B,D)
F= np.array(([2/3*r*np.cos(theta),0]))
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_AC = line_gen(A,C)
x_BE = line_gen(B,E)
x_ED = line_gen(E,D)
x_AE = line_gen(A,E)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_AC[0,:],x_AC[1,:],label='$AC$')
plt.plot(x_BE[0,:],x_BE[1,:],label='$BE$')
plt.plot(x_ED[0,:],x_ED[1,:],label='$ED$')
plt.plot(x_AE[0,:],x_AE[1,:],label='$AE$')
v1=A-E
v2=A-B
l_BD=np.linalg.norm(B-D)
A_BDE=(math.sqrt(3)/4)*(l_BD**2)
l_BC=np.linalg.norm(B-C)
A_ABC=(math.sqrt(3)/4)*(l_BC**2)
if(A_BDE == A_ABC/4):
   print("ar(BDE)=1/4*ar(ABC)")
v1=A-E
v2=A-B
A_BAE=0.5*np.linalg.norm((np.cross(v1,v2)))
if(round(A_BDE) == round(A_BAE/2)):
   print("ar(BDE)=1/2*ar(BAE)")
v3=B-C
v4=B-E
A_BEC=0.5*np.linalg.norm((np.cross(v3,v4)))
if(round(A_ABC) == round(2*A_BEC)):
   print("ar(ABC)=2*ar(BEC)")
v5=B-F
v6=B-E
A_BFE=0.5*np.linalg.norm((np.cross(v5,v6)))
v7=A-D
v8=A-F
A_AFD=0.5*np.linalg.norm((np.cross(v7,v8)))
if(round(A_BFE) == round(A_AFD)):
   print("ar(BFE)=ar(AFD)")
v9=F-E
v10=F-D
A_FED=0.5*np.linalg.norm((np.cross(v9,v10)))
if(round(A_BFE) == round(2*A_FED)):
   print("ar(BFE)=2*ar(FED)")
v11=A-F
v12=A-C
A_AFC=0.5*np.linalg.norm((np.cross(v11,v12)))
if(round(A_FED) == round(1/8*A_AFC)):
   print("ar(FED)=1/8*ar(AFC)")
#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,E,F)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F']
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
plt.title('triangle')
#if using termux
plt.savefig('/home/user/vamsi/line_a2/par.pdf')
#subprocess.run(shlex.split("termux-open '/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-2.pdf'")) 
#else
plt.show()
