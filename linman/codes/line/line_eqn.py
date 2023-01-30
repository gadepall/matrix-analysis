#Code by GVV Sharma
#August 13, 2020
#released under GNU GPL
#Drawing lines between two points
#Drawing line given slope

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
slope = -4
P = np.array(([-2,3]))
m = np.array(([1,slope]))

#Input values for ax2
A = np.array(([1,-1]))
B = np.array(([3,5]))


#Generating line 1
k1 = 0.1
k2 = -0.15
x_m=line_dir_pt(m,P,k1,k2)

#Generating line 2
x_AB=line_gen(A,B)

#Plotting Line 1 and Line 2
ax1.plot(x_m[0,:],x_m[1,:],label='$4x-y=-5$')
ax1.grid()
ax1.legend(loc='best')
ax2.plot(x_AB[0,:],x_AB[1,:],label='$3x-y=4$')
ax2.grid()
ax2.legend(loc='best')

#Labeling the coordinates in Fig 1
tri_coords = P.T
ax1.scatter(tri_coords[0], tri_coords[1])
vert_labels = ['P']
for i, txt in enumerate(vert_labels):
    ax1.annotate(txt, # this is the text
                 (tri_coords[0], tri_coords[1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

#plt.xlabel('$x$')
#plt.ylabel('$y$')
#plt.legend(loc='best')
#plt.grid() # minor

#Labeling the coordinates in Fig 2
tri_coords = np.vstack((A,B)).T
ax2.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['$x_1$','$x_2$']
for i, txt in enumerate(vert_labels):
    ax2.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.axis('equal')

#Saving figures
#if using termux
fig1.savefig('./figs/line/line_slope.pdf')
fig1.savefig('./figs/line/line_slope.png')

fig2.savefig('./figs/line/line_points.pdf')
fig2.savefig('./figs/line/line_points.png')
#subprocess.run(shlex.split("termux-open ./figs/line/line_slope.pdf"))
subprocess.run(shlex.split("termux-open ./figs/line/line_points.pdf"))
#else
#plt.show()









