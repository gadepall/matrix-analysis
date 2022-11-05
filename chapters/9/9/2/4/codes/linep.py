import numpy as np
import math as m
import random as r
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                        #for path to external scripts
sys.path.insert(0,'/sdcard/FWC/Matrices/Line/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *

#if using termux
import subprocess
import shlex
#end if


#input parameters
a=4                     #length of base (AB)
b=2                     #length of side (AD)
theta = np.pi/3         #angle between AB & AD
A = np.array([0,0])     #Vertex A
B = np.array([a,0])
C = b*np.array([(a/b)+(np.cos(theta)),(np.sin(theta))])
D = b*np.array([(np.cos(theta)),(np.sin(theta))])

P = b*np.array([(np.cos(r.uniform(0,theta))),(np.sin(r.uniform(0,theta)))])


A1 = LA.norm(np.cross(A-D,A-P))/2   #Area of triangle APD
A2 = LA.norm(np.cross(B-C,P-B))/2   #Area of triangle PBC

A3 = LA.norm(np.cross(A-B,B-P))/2   #Area of triangle APB
A4 = LA.norm(np.cross(C-D,D-P))/2   #Area of triangle PCD

print(A1+A2)

APa = LA.norm(np.cross(A-D,A-B))    #Area of || ABCD
print(APa)

if (A3+A4 == APa/2):
    print('Areas are equal')
else:
    print('Areas are not equal')

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)

x_PA = line_gen(P,A)
x_PB = line_gen(P,B)
x_PC = line_gen(P,C)
x_PD = line_gen(P,D)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_BC[0,:],x_BC[1,:])
plt.plot(x_CD[0,:],x_CD[1,:])
plt.plot(x_DA[0,:],x_DA[1,:])

plt.plot(x_PA[0,:],x_PA[1,:])
plt.plot(x_PB[0,:],x_PB[1,:])
plt.plot(x_PC[0,:],x_PC[1,:])
plt.plot(x_PD[0,:],x_PD[1,:])


#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,P)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','P']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt,                                   # this is the text
                 (tri_coords[0,i], tri_coords[1,i]),    # this is the point to label
                 textcoords="offset points",            # how to position the text
                 xytext=(5,-15),                      # distance from text to points (x,y)
                 ha='left')                           # horizontal alignment can be left, right or center

plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
#plt.legend(loc='best')
plt.grid()
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/FWC/Matrices/Line/linep.pdf')
#subprocess.run(shlex.split("termux-open '/sdcard/FWC/Matrices/Line/linep.pdf'")) 
#else
#plt.show()
