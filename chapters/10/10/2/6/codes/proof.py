import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Linearalgebra/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
r=3
#Centre and point 
X = np.array(([0,0]))
A = np.array(([0,-3]))
B = np.array(([-4,-3]))
C = np.array(([-2.4,-1.8]))
D = np.array(([-5,-3]))
E = np.array(([3,-3]))
B=[]
for i in range(0,2):
    e=int(input('Enter the coordinates other than A in the tangent drawn to the circle:'))
    B.append(e)
dis=np.linalg.norm(B)
dis1=np.linalg.norm(A)
if(dis1<dis):
    print("The angle made with tangent and radius at point of contact is 90");


##Generating all lines
xXA = line_gen(X,A)
xXB = line_gen(X,B)
xDE = line_gen(D,E)
xXC = line_gen(X,C)
##Generating the circle
x_circ= circ_gen(X,r)

#Plotting all lines
plt.plot(xXA[0,:],xXA[1,:],label='$radius$')
plt.plot(xXB[0,:],xXB[1,:],label='$XB$')
plt.plot(xDE[0,:],xDE[1,:],label='$tangent$')
plt.plot(xXC[0,:],xXC[1,:],label='$radius$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((A,B,X,D,E,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','X','D','E','C']
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
print("Radius of circle is 3cm")
#if using termux
plt.savefig('/sdcard/Linearalgebra/proof.pdf')
#subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/cbse-10-13.pdf"))
#else
#plt.show()
