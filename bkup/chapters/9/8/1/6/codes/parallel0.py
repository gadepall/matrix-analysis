import math
import numpy as np
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

A = np.array(([0,0]))
B = np.array(([4,0]))
C = np.array(([6,6]))

D = C-B
g = (B-A)@(C-A)
a = np.linalg.norm((B-A))
b = np.linalg.norm((C-A))
theta1 = np.arccos(g/(a*b))
f = (D-C)@(A-C)
c = np.linalg.norm((D-C))
theta3 = np.arccos(f/(c*b))
if(theta1 == theta3):
    print("angles are equal")
##Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)
x_AC = line_gen(A,C)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Diameter$')
plt.plot(x_BC[0,:],x_BC[1,:])#,label='$Diameter$')
plt.plot(x_CD[0,:],x_CD[1,:])#,label='$Diameter$')
plt.plot(x_DA[0,:],x_DA[1,:])#,label='$Diameter$')
plt.plot(x_AC[0,:],x_AC[1,:])#,label='$Diameter$')

#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')
plt.show()
plt.savefig('/sdcard/Download/IITH-FWC-main/matrices/lines/parallel.png')
subprocess.run(shlex.split("termux-open /sdcard/Download/IITH-FWC-main/matrices/lines/parallel.png"))
