#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys, os                                          #for path to external scripts
script_dir = os.path.dirname(__file__)
#lib_relative = '../../../CoordGeo'
#fig_relative = '../figs/fig1.pdf'
#sys.path.insert(0,os.path.join(script_dir, lib_relative))
sys.path.insert(0,'/sdcard/github/cbse-papers/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
k = 1
c = 5
a = 4
theta = np.arctan(c/a)
A = np.sqrt(c**2+a**2)*np.array(([np.cos(theta),np.sin(theta)]))
B = np.array(([0,0]))
e1 = np.array(([1,0]))
C = a*e1
M = (k*A+B)/(k+1)
D = (k*A+C)/(k+1)


##Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(C,B)
x_CA = line_gen(A,C)
x_MN = line_gen(M,D)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_BC[0,:],x_BC[1,:])
plt.plot(x_CA[0,:],x_CA[1,:])
plt.plot(x_MN[0,:],x_MN[1,:])


#Labeling the coordinates
tri_coords = np.vstack((A,B,C,M,D)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','M','D']
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

#if using termux
plt.savefig('/sdcard/github/matrix-analysis/chapters/9/8/2/7/figs/fig1.pdf')
subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/fig1.pdf"))
#else
#plt.show()






