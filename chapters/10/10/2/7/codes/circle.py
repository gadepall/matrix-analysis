

#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0, '/sdcard/Download/10/codes/CoordGeo')        #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if
#Input parameters
a=3
p=5
lamda = 0.5
theta = np.arccos(a/p)
A = 5*np.array(([-np.cos((90-theta)),np.sin((90-theta))]))
B = 5*np.array(([-np.cos((270+theta)),-np.sin((270+theta))]))
P = A + lamda*(B-A)  
O = np.array(([0,0]))
x_circ_1= circ_gen(O,a)
x_circ_2= circ_gen(O,p)
#Plotting the circle
plt.plot(x_circ_1[0,:],x_circ_1[1,:])
plt.plot(x_circ_2[0,:],x_circ_2[1,:])


##Generating all lines
x_OA = line_gen(O,A)
x_AB = line_gen(A,B)
x_OP = line_gen(O,P)


#Plotting all lines
plt.plot(x_OA[0,:],x_OA[1,:])
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_OP[0,:],x_OP[1,:])


#Labeling the coordinates
tri_coords = np.vstack((O,A,B,P)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','A','B','P']
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
plt.savefig('/sdcard/Download/FWC/trunk/circle_assignment/fig.pdf')
subprocess.run(shlex.split("termux-open '/sdcard/Download/FWC/trunk/circle_assignment/fig.pdf'")) 
