#Code by GVV Sharma
#August 13, 2020
#released under GNU GPL
#Checking if given points are collinear

import numpy as np
import matplotlib.pyplot as plt

import sys                                          #for path to external scripts
sys.path.insert(0, '/storage/emulated/0/tlc/school/ncert/linman/codes/CoordGeo')        #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#setting up plot
fig1 = plt.figure(1)
fig2 = plt.figure(2)
#ax1 = fig.add_subplot(111, aspect='equal')
ax1 = fig1.gca(aspect='equal')
ax2 = fig2.gca(aspect='equal')

#Input values for ax1
C = np.array(([3,0]))
B = np.array(([-2,-2]))
A = np.array(([8,2]))

#Input values for ax2
P = np.array(([3,2]))
Q = np.array(([-2,-3]))
R = np.array(([2,3]))


#Generating line 1
x_AB=line_gen(A,B)

#Generating line 2

#Plotting Line 1 and Line 2
ax1.grid()
plt.axis('equal')

ax2.plot(x_AB[0,:],x_AB[1,:],label='$3x-y=4$')
ax2.grid()
ax2.legend(loc='best')

#Labeling the coordinates in Fig 1
tri_coords = np.vstack((P,Q,R)).T
ax1.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','Q', 'R']
for i, txt in enumerate(vert_labels):
    ax1.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.axis('equal')

#Labeling the coordinates in Fig 2
tri_coords = np.vstack((A,B,C)).T
ax2.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B', 'C']
for i, txt in enumerate(vert_labels):
    ax2.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')

#Saving figures
#if using termux
fig2.savefig('./figs/line/points_collinear.pdf')
fig2.savefig('./figs/line/points_collinear.png')

fig1.savefig('./figs/line/points_triangle.pdf')
fig1.savefig('./figs/line/points_triangle.png')
subprocess.run(shlex.split("termux-open ./figs/line/points_collinear.pdf"))
subprocess.run(shlex.split("termux-open ./figs/line/points_triangle.pdf"))
#else
#plt.show()









