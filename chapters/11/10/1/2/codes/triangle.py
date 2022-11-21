



import sys                                          #for path to external scripts
sys.path.insert(0,'/home/manoj/Documents/CoordGeo')         #path to my scripts

#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from sympy import symbols,Eq,solve
#if using termux
import subprocess
import shlex
#end if

#Input parameters
a=4
e1=np.array([1,0])
e2=np.array([0,1])
theta=60*np.pi/180
x=(a/2)*np.tan(theta)
A=x*e1
B=(a/2)*e2
C=-(a/2)*e2
print(A,B,C)
#A = np.array(([r*np.cos(theta), r*np.sin(theta)]))
#Generating all lines
x_AB = line_gen(A,B)
x_BC= line_gen(B,C)
x_CA = line_gen(C,A)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_BC[0,:],x_BC[1,:])#,label='$Diameter$')
plt.plot(x_CA[0,:],x_CA[1,:])#,label='$Diameter$')

#
#
#Labeling the coordinates
tri_coords = np.vstack((A,B,C,)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A(x,y)','B(0,a)','C(0,-a)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(18,20), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')
#plt.axis([-4,4.5,-3,3])


plt.savefig('/home/manoj/git/FWC/Matrix/line/triangle.png')

plt.savefig('/home/manoj/git/FWC/Matrix/line/code-py/triangle.pdf')
plt.show()
