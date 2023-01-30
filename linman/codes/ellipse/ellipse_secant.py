#Program to plot  an ellipse  given two points
#Code by GVV Sharma
#August 9, 2020
#Released under GNU GPL

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0, '/storage/emulated/0/tlc/school/ncert/linman/codes/CoordGeo')        #path to my scripts


#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if

#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
len = 100
y = np.linspace(-5,5,len)

#Given Points
p = np.array(([4,3]))
q = np.array(([-1,4]))
P = np.diag(p)
Q = np.diag(q)
c = np.array(([0,0]))

dA = np.block([[p@P],[q@Q]])
db = np.array(([1,1]))

#Ellipse parameters
d = LA.solve(dA,db)
a = np.sqrt(1/d[0])
b = np.sqrt(1/d[1])
xStandardEllipse = ellipse_gen(a,b)

#Major and Minor Axes
MajorStandard = np.array(([a,0]))
MinorStandard = np.array(([0,b]))
#

#Plotting the standard ellipse
plt.plot(xStandardEllipse[0,:],xStandardEllipse[1,:],label='Standard ellipse')


#Labeling the coordinates
tri_coords = np.vstack((p,q,c)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['$a$','$b$','$c$']
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
plt.savefig('./figs/ellipse/ellipse_secant.pdf')
plt.savefig('./figs/ellipse/ellipse_secant.png')
subprocess.run(shlex.split("termux-open ./figs/ellipse/ellipse_secant.pdf"))
#else
#plt.show()







