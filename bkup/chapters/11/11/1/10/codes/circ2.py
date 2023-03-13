import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import sys               #for path to external scripts
sys.path.insert(0,'/sdcard/Download/parv/CoordGeo')

#local imports
from conics.funcs import circ_gen
from line.funcs import *

#if using termux
import subprocess
import shlex

A = np.array([4,1])
B = np.array([6,5])

#line parameters
M = np.array([4,1])
c1 = 16

#Entering equations inmatrix form
A1 = np.array([[-4,-1,0],[12,10,1],[8,2,1]])
b = np.array([16,-61,-17])

S = LA.solve(A1,b)


#Centre and radius calculations
c = -np.array((S[0],S[1]))
r = math.sqrt(LA.norm(c)**2 - S[2])

print("The centre is : ", c)
print("The radius is : " ,r)

P = np.array(([3/2,10]))
Q = np.array(([9/2,-2]))

#generating line
x_PQ = line_gen(P,Q)
plt.plot(x_PQ[0,:],x_PQ[1,:],label='{}X={}'.format(M,c1))

#generating circle
x_circ= circ_gen(c,r)

#plotting
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')

tri_coords = np.vstack((c,A,B)).T
plt.scatter(tri_coords[0,:],tri_coords[1,:])

vert_labels = ['c','A','B']
for i, txt in enumerate(vert_labels):
    label = "{}({:.0f},{:.0f})".format(txt, tri_coords[0,i],tri_coords[1,i]) #Form label as A(x,y)
    plt.annotate(label, # this is the text
            (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')
plt.savefig('/sdcard/Download/latexfiles/circle/figs/circ2.png')
plt.show()
