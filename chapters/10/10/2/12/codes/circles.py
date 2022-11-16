import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/IITH-FWC-main/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

def tangent_from(V,u,f,p):
    sigma = (V@p+u)@(V@p+u).T - (p.T@V@p + 2*u.T@p + f)*V 
    lamda, gamma = LA.eigh(sigma)
    n1 = gamma@np.array([np.sqrt(np.abs(lamda[0])), -np.sqrt(np.abs(lamda[1]))])
    n2 = gamma@np.array([np.sqrt(np.abs(lamda[0])), np.sqrt(np.abs(lamda[1]))])
    return n1,n2

def line_intersection(n1,A1,n2,A2):
    N = np.vstack((n1,n2))
    p = np.zeros(2)
    p[0] = n1@A1
    p[1] = n2@A2
    A = np.linalg.inv(N)@p
    return A
#Input parameters
r = 4
bd = 8
cd = 6
cen = np.array([0,0])
O = cen.reshape(2,1)
D = np.array([0,-4])
D = D.reshape(2,1)
h1 = np.array([D[0]+bd,D[1]])
h1 = h1.reshape(2,1)
h2 = np.array([D[0]-cd,D[1]])
h2 = h2.reshape(2,1)
f = LA.norm(O)**2-r**2
V = np.array([[1,0],[0,1]])
u = -np.array([O[0],O[1]]).reshape(2,1)

n1,n2 = tangent_from(V,u,f,h1)
n1 = n1.reshape(2,1)
n2 = n2.reshape(2,1)
m1 = omat@n1
m2 = omat@n2
T1 = h1 - ((m1.T@(V@h1))/(m1.T@V@m1))*m1
T2 = h1 - ((m2.T@(V@h1))/(m2.T@V@m2))*m2

n3,n4 = tangent_from(V,u,f,h2)
n3 = n3.reshape(2,1)
n4 = n4.reshape(2,1)
m3 = omat@n3
m4 = omat@n4
T3 = h2 - ((m3.T@(V@h2+u))/(m3.T@V@m3))*m3
T4 = h2 - ((m4.T@(V@h2+u))/(m4.T@V@m4))*m4

n5 = n1.reshape(1,2)
n6 = n4.reshape(1,2)

A = line_intersection(n5,T1,n6,T4)
A = A.reshape(2,1)

AB = LA.norm(A-h1)
print("length of AB is",AB)
AC = LA.norm(A-h2)
print("length of AC is",AC)


A = A.reshape(2,)
h2 = h2.reshape(2,)
h1 = h1.reshape(2,)
D = D.reshape(2,)
T4 = T4.reshape(2,)
T1 = T1.reshape(2,)

##Generating all lines
x_h1A = line_gen(h1,A)
x_h2A = line_gen(h2,A)
x_h1h2 = line_gen(h1,h2)
x_cenD = line_gen(cen,D)
x_circ= circ_gen(cen,r)
#Plotting all lines
plt.plot(x_h1A[0,:],x_h1A[1,:])#label=('$Diameter$')
plt.plot(x_h2A[0,:],x_h2A[1,:])#label=('$Diameter$')
plt.plot(x_h1h2[0,:],x_h1h2[1,:])#label=('$Diameter$')
plt.plot(x_cenD[0,:],x_cenD[1,:])
#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((A,h2,h1,D,cen)).T
tri_coords = tri_coords.reshape(2,-1)
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','C','B','D','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/Download/IITH-FWC-main/matrices/circles/circle.png')
subprocess.run(shlex.split("termux-open /sdcard/Download/IITH-FWC-main/matrices/circles/circle.png"))
#else
plt.show()

