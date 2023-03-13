#Code by GVV Sharma
#July 15, 2020
#released under GNU GPL
#Drawing a standard hyperbola

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

#Standard hyperbola
len = 100
y = np.linspace(-5,5,len)
x = hyper_gen(y)

#Hyperbola vertices
V1 = np.array([1,0])
V2 = -V1


#Plotting the hyperbola
plt.plot(x,y,label='Standard hyperbola')
plt.plot(-x,y)

#Labeling the coordinates
hyper_coords = np.vstack((V1,V2)).T
plt.scatter(hyper_coords[0,:], hyper_coords[1,:])
vert_labels = ['$V_1$','$V_2$']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (hyper_coords[0,i], hyper_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('./figs/hyper.pdf')
plt.savefig('./figs/hyper.eps')
subprocess.run(shlex.split("termux-open ./figs/hyper.pdf"))
#else
#plt.show()
