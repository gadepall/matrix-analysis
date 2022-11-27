
import numpy as np
import mpmath as mp 
import matplotlib.pyplot as plt 
from numpy import linalg as LA

import sys
sys.path.insert(0,'/sdcard/fwc/matrices/CoordGeo')

import subprocess
import shlex

#equation of curve
x=np.linspace(-1,5,100)
y=(x-2)**2

#direction vector of tangent
M=np.array(([1,2]))
omat=np.array(([0,1],[-1,0]))
#normal vector of tangent
n=omat@(M.T)
#parameters of curve
v=np.array(([1,0],[0,0]))
U=np.array(([-2,-0.5]))
F=4
#points joining chord
P=np.array(([2,0]))
Q=np.array(([4,4]))

#finding point of contact

p1=np.array(([0,1]))   #eigen vector
k=(p1@(U.T))/(p1@(n.T)) #finding kappa
a = np.block([[(U+k*n).T],[v]])
b = np.block([[-F],[(k*n-U).reshape(-1,1)]])
R = LA.lstsq(a,b,rcond=None)[0]      #point of contact
R = R.flatten()
print(R)

def line_gen(X,Y):
  len =10
  dim = X.shape[0]
  x_XY = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = X + lam_1[i]*(Y-X)
    x_XY[:,i]= temp1.T
  return x_XY

def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB

#generating lines
x_C=line_dir_pt(M,Q,-4,2)
x_R=line_dir_pt(M,R,-2,4)
#plotting
plt.plot(x,y,label='$Parabola$')
plt.plot(x_C[0,:],x_C[1,:],label='$Chord$')
plt.plot(x_R[0,:],x_R[1,:],label='$Tangent$')

#Labelling
tri_coords = np.vstack((P,Q,R)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','Q','R']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-2,4), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axis('equal')
plt.legend(loc='best')

plt.savefig('/sdcard/fwc/matrices/conics/figs/main.pdf')
subprocess.run(shlex.split("termux-open /sdcard/fwc/matrices/conics/figs/main.pdf"))



