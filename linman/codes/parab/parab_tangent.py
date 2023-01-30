#Program to plot  the tangent of a parabola
#Code by GVV Sharma
#Released under GNU GPL
#August 10, 2020

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
y = np.linspace(0,4,len)

#parab parameters
V = np.array(([0,0],[0,1]))
u = np.array(([-2,1]))
f = 4
p = np.array(([1,0]))
foc = 2*np.abs(p@u)
print(foc)

#Generating the Standard parabola
x = parab_gen(y,foc)

#Affine Parameters
R =  np.array(([0,1],[1,0]))
cA = np.vstack((u-2*p,V))
cb = np.vstack((-f,(-(2*p+u)).reshape(-1,1)))
c = LA.lstsq(cA,cb,rcond=None)[0]
c = c.flatten()

xStandardparab = np.vstack((x,y))
xActualparab = xStandardparab + c[:,np.newaxis]

#Tangent Analysis
m_0 = 2/3
m = np.array(([1,m_0]))
n = omat@m
kappa = p@u/(p@n)

qA = np.vstack((u+kappa*n,V))
qb = np.vstack((-f,(kappa*n-u).reshape(-1,1)))
q = LA.lstsq(qA,qb,rcond=None)[0]
q = q.flatten()
O = np.array([0,0])
print(c,q)

#Generating the tangent
k1 = 2
k2 = -2
x_AB = line_dir_pt(m,q,k1,k2)
#
#Labeling the coordinates
parab_coords = np.vstack((q,c)).T
plt.scatter(parab_coords[0,:], parab_coords[1,:])
vert_labels = ['$q$','$c$']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (parab_coords[0,i], parab_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='Tangent')

#Plotting the actual parabola
#plt.plot(xStandardparab[0,:],xStandardparab[1,:],label='Parabola',color='r')
plt.plot(xActualparab[0,:],xActualparab[1,:],label='Parabola',color='r')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('./figs/parab/parab_tangent.pdf')
plt.savefig('./figs/parab/parab_tangent.png')
subprocess.run(shlex.split("termux-open ./figs/parab/parab_tangent.pdf"))
##else
##plt.show()
