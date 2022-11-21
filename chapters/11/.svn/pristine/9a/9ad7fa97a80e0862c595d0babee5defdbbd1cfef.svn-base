#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys      #for path to external scrip>
sys.path.insert(0,'/home/dell/matrix/CoordGeo')  #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#input parameters
A=np.array(([1.732,1],[1,1.732]))
B=np.array(([1,1]))
e1=np.array(([1,0]))
n1=A[0,:]
n2=A[1,:]
c1=B[0]
c2=B[1]

#solution vector
x=LA.solve(A,B)

#Direction vectors
m1=omat@n1
m2=omat@n2

#points on lines
x1=c1/(n1@e1)
A1=x1*e1
x2=c2/(n2@e1)
A2=x2*e1
num=m1@m2.T
den=LA.norm(m1)*LA.norm(m2)
theta1=np.arccos(num/den)
theta=(theta1*180)/np.pi
print("The angle between given lines is",np.round(theta))

#Generating all lines
k1=-1
k2=2
x_AB = line_dir_pt(m1,A1,k1,k2)
x_CD = line_dir_pt(m2,A2,k1,k2)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Diameter$')
plt.plot(x_CD[0,:],x_CD[1,:])#,label='$Diameter$')

#Labeling the coordinates
tri_coords = np.vstack((x,A1,A2)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['x','P','Q']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
    (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') #horizontal alignment can be left,right,center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('/sdcard/matrix/CoordGeo/line1.pdf')
plt.show()
