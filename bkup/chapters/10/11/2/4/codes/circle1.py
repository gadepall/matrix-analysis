#Code by Anusha Jella
#Sep 23, 2022
#License
#To construct a circle and two tangents to it from a point outside which are inclined each other at 60 degrees


#Python libraries for math and graphics
import numpy as np
import math
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/anu/anusha1/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
r = 5
e1 = np.array(([1,0]))

#Centre and point P 
P = np.array(([0,0]))#Point P
theta=np.pi/6
a=np.sin(theta)
d=r/a
O = d*e1
#print(mp.sin(theta1))
Q1 = r*mp.cot(theta)*np.array(([mp.cos(theta),mp.sin(theta)]))
Q2 = r*mp.cot(theta)*np.array(([mp.cos(theta),-mp.sin(theta)]))
#a=np.sin(theta)
#P1 = (r/a)*np.array(([np.cos(theta),np.sin(theta)]))
print(r/a,P)
v1=P-Q1
v2=P-Q2
v11=v1@v2
v22=np.linalg.norm(v1)*np.linalg.norm(v2)
angle =mp.acos((v11/v22))
print(round(math.degrees(angle)))
##Generating all lines
xPQ1 = line_gen(P,Q1)
xPQ2 = line_gen(P,Q2)

##Generating the circle
x_circ= circ_gen(O,r)

#Plotting all lines
plt.plot(xPQ1[0,:],xPQ1[1,:],label='$Tangent1$')
plt.plot(xPQ2[0,:],xPQ2[1,:],label='$Tangent2$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((P,Q1,Q2,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','Q1','Q2','O']
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
plt.savefig('/home/anu/anusha1/latex_asgn/circle1.pdf')
#subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/cbse-10-13.pdf"))
#else
plt.show()

