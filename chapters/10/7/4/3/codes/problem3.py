
#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/sat/CoordGeo')

#local imports
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Given points on the circle
P = np.array(([6,-6]))
Q = np.array(([3,-7]))
R = np.array(([3,3]))

#Matrix points to solve the system of equations
b = np.array(([-58,-72,-18]))
A = np.block([[6,-14,1],[12,-12,1],[6,6,1]])
#Solution vector
O = LA.solve(A,b)
#Centre and radius calculations
c = -np.array((O[0],O[1]))
r = math.sqrt(LA.norm(c)**2 - O[2])

##Generating the circle
x_circ= circ_gen(c,r)

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((c,P,Q,R)).T
plt.scatter(tri_coords[0,:],tri_coords[1,:])

vert_labels = ['c','P','Q','R']
for i, txt in enumerate(vert_labels):
    label = "{}({:.0f},{:.0f})".format(txt, tri_coords[0,i],tri_coords[1,i]) #Form label as A(x,y)
    plt.annotate(label, # this is the text
            (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('../figs/problem3.pdf')
#else
plt.show()






