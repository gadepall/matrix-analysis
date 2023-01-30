#Code by GVV Sharma
#July 15, 2020
#released under GNU GPL
#Drawing a ellipse

import numpy as np
import matplotlib.pyplot as plt

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if

#Standard ellipse
a = 2
b = 1
x = ellipse_gen(a,b)

#Vertices
A1 = np.array([a,0])
A2 = -A1
B1 = np.array([0,b])
B2 = -B1

#Plotting the ellipse
plt.plot(x[0,:],x[1,:],label='Standard Ellipse')

#Labeling the coordinates
ellipse_coords = np.vstack((A1,A2,B1,B2)).T
plt.scatter(ellipse_coords[0,:], ellipse_coords[1,:])
vert_labels = ['$A_1$','$A_2$','$B_1$', '$B_2$']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (ellipse_coords[0,i], ellipse_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('./figs/ellipse.pdf')
plt.savefig('./figs/ellipse.eps')
subprocess.run(shlex.split("termux-open ./figs/ellipse.pdf"))
#else
#plt.show()
#
#
#
#
#
#
#
