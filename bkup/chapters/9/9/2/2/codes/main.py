import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/sdcard/fwc/matrices/CoordGeo')

import subprocess
import shlex

l=10
j=5
theta=np.pi/2.4
k=1
A=np.array(([0,0]))
B=l*np.array(([np.cos(0),np.sin(0)]))
D=j*np.array(([np.cos(theta),np.sin(theta)]))
C=B+D

E=(k*A+B)/(k+1)
F=(k*B+C)/(k+1)
G=(k*C+D)/(k+1)
H=(k*D+A)/(k+1)

ab=np.array([10,0])
ad=np.array([1.25,4.8])
P=np.cross(ab,ad)
print('ar(ABCD)=',P)

ef=np.array([5.625,2.4])
eh=np.array([-4.375,2.4])
Q=np.cross(ef,eh)
print('ar(EFGH)=',Q)

def line_gen(X,Y):
  len =10
  dim = X.shape[0]
  x_XY = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = X + lam_1[i]*(Y-X)
    x_XY[:,i]= temp1.T
  return x_XY

x_AB=line_gen(A,B)
x_BC=line_gen(B,C)
x_CD=line_gen(C,D)
x_DA=line_gen(D,A)
x_EF=line_gen(E,F)
x_FG=line_gen(F,G)
x_GH=line_gen(G,H)
x_HE=line_gen(H,E)

plt.plot(x_AB[0,:],x_AB[1,:],'-r')
plt.plot(x_BC[0,:],x_BC[1,:],'-r')
plt.plot(x_CD[0,:],x_CD[1,:],'-r')
plt.plot(x_DA[0,:],x_DA[1,:],'-r')
plt.plot(x_EF[0,:],x_EF[1,:],'-g')
plt.plot(x_FG[0,:],x_FG[1,:],'-g')
plt.plot(x_GH[0,:],x_GH[1,:],'-g')
plt.plot(x_HE[0,:],x_HE[1,:],'-g')

tri_coords = np.vstack((A,B,C,D,E,F,G,H)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F','G','H']
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

plt.savefig('/sdcard/fwc/matrices/lines/figs/main.pdf')
subprocess.run(shlex.split("termux-open /sdcard/fwc/matrices/lines/figs/main.pdf"))

