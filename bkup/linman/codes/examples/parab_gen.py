#Code by GVV Sharma
#July 15, 2020
#released under GNU GPL
#Drawing the standard parabola

import numpy as np
import matplotlib.pyplot as plt

#local imports
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if

simlen = 100
#Standard parabola
x = np.linspace(-1,1,simlen)
y = parab_gen(x)

#Parabola points
#Standard Parabola Vertex
O = np.array([0,0])

#Focus
F= np.array([0,1/4])

#Point on the directrix
D = -F

#Plotting the parabola
plt.plot(x,y,label='Standard Parabola')

#Plotting the directrix
plt.plot(x,D[1]*np.ones(simlen),label='Directrix')

#Labeling the coordinates
parab_coords = np.vstack((O,F, D)).T
plt.scatter(parab_coords[0,:], parab_coords[1,:])
vert_labels = ['O','F','D']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (parab_coords[0,i], parab_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
#
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('./figs/parabola.pdf')
plt.savefig('./figs/parabola.eps')
subprocess.run(shlex.split("termux-open ./figs/parabola.pdf"))
#else
#plt.show()
