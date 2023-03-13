
#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/mat lab/Circle/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
r1 = 5
r2 = 3

A = np.array(([0,0]))
e1 = np.array(([1,0]))
B   4 * e1


#Distance between A and B
#d = np.linalg.norm(A-B)
#print(d)

#Centre and point 
O1 = A #Centre1
O2 = B #Centre2
P = np.array(([4,3]))
P1 = np.array(([4,-3]))
d = np.linalg.norm(P-P1)
print(d)
theta = mp.asin(r/d)

Q1 = r*mp.cot(theta)*np.array(([mp.cos(theta),mp.sin(theta)]))
Q2 = r*mp.cot(theta)*np.array(([mp.cos(theta),-mp.sin(theta)]))


##Generating all lines
xAB = line_gen(P,P1)
xAP = line_gen(A,P)
xAP1 = line_gen(A,P1)
xPP1 = line_gen(P,P1)
#Generating the circle
x_circ= circ_gen(O1,r1)
y_circ= circ_gen(O2,r2)
#Plotting all lines
plt.plot(xAB[0,:],xAB[1,:],label='$Tangent1$')
plt.plot(xAP[0,:],xAP[1,:],label='$Tangent2$')
plt.plot(xAP1[0,:],xAP1[1,:],label='$Tangent2$')
plt.plot(xPP1[0,:],xPP1[1,:],label='$Tangent1$')
#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
plt.plot(y_circ[0,:],y_circ[1,:],label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((A,B,O1,O2,P,P1)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','O1','O2','P','P1']
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
plt.savefig('/sdcard/mat lab/Circle/CoordGeo/figs1.pdf')
subprocess.run(shlex.split("termux-open /sdcrad/mat lab/Circle/CoordGeo/figs1.pdf"))
#else
#plt.show();
