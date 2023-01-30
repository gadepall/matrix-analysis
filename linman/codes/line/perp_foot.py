#Code by GVV Sharma
#August 13, 2020
#released under GNU GPL
#Drawing the foot of the perpendicular from
#a point to a line

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

#Input values
n = np.array(([3,-4]))
P = np.array(([3,-5]))
cn = 26
#Foot of the perpendicular
x_0 = perp_foot(n,cn,P)
print(x_0,66/25,-113/25)

#Generating all lines
k1 = 0
k2 = -0.15
x_n=line_dir_pt(n,P,k1,k2)
k1 = 0.05
k2 = -0.05
x_m = line_dir_pt(omat@n,x_0,k1,k2)

#Plotting all lines
plt.plot(x_n[0,:],x_n[1,:],label='$4x+3y=-3$')
plt.plot(x_m[0,:],x_m[1,:],label='$3x-4y=26$')


#Labeling the coordinates
tri_coords = np.vstack((x_0,P)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['$x_0$','P']
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
plt.savefig('./figs/line/perp_foot.pdf')
plt.savefig('./figs/line/perp_foot.png')
subprocess.run(shlex.split("termux-open ./figs/line/perp_foot.pdf"))
#else
#plt.show()







