#python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys
sys.path.insert(0,'/home/sireesha/Desktop/CoordGeo')

#local imports

from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#input parameters
A=np.array(([4,4]))
B=np.array(([3,5]))
C=np.array(([-1,-1]))
if (A-B)@(B-C)==0:
   print("AB is perpendicular to BC and hence the triangle is right angled")
elif (B-C)@(C-A)==0:
   print("BC  is perpendicular to CA and hence the triangle is right angled")
elif (C-A)@(A-B)==0:
   print("CA is perpendicular to AB and hence the triabgle is right angled")



#generating all lines
X_AB=line_gen(A,B)
X_BC=line_gen(B,C)
X_CA=line_gen(C,A)

#plotting all the lines
plt.plot(X_AB[0,:],X_AB[1,:],label='$A$')
plt.plot(X_BC[0,:],X_BC[1,:],label='$B$')
plt.plot(X_CA[0,:],X_CA[1,:],label='$C$')

#Labelling the coordinates
tri_coords=np.vstack((A,B,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C']
for i, txt in enumerate(vert_labels):
   plt.annotate(txt, #this is the text
                (tri_coords[0,i],tri_coords[1,i]),#this is the point to label
                textcoords="offset points",#how to position the text
                xytext=(0,10),#distance from text to points(x,y)
                ha='center')#horizontal alignment can be left ,right or center

plt.xlabel('$x$')
plt.ylabel('$y&')
plt.legend(loc='best')
plt.grid()# minor
plt.axis('equal')  


#if using termux
#plt.savefig('#path to save fig')
#subprocess.run(shlex.split( "termux-open 'path "))
#else
plt.show()  
