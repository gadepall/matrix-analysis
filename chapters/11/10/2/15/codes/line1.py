#Code by GVV Sharma (works on termux)
#March 1, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To draw a quadrilateral circumscribing a circle


#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/user/Desktop/matrices/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#input parameters
A = np.array(([-2,9]))
O = np.array(([0,0])) 


#directional vector
n=O-A
m=omat@n

##Generating all lines
x_AO = line_gen(A,O)
k1=-1
k2=1
x_CD = line_dir_pt(m,A,k1,k2)


#Plotting all lines
plt.plot(x_AO[0,:],x_AO[1,:])#,label='$Diameter$')
plt.plot(x_CD[0,:],x_CD[1,:])#,label='$Line equation$')

#print the equation
print(n[0],'*','x',n[1],'*','y','=',(n[0]*A[0]+n[1]*A[1]),sep="")



#Labeling the coordinates
tri_coords = np.vstack((A,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','O']
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
#plt.savefig('/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-14.pdf')
#subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/cbse-10-14.pdf"))
#else
plt.show()
