#Code by GVV Sharma (works on termux)
#October 4, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To find the tangents of a circle from an external point


#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
#sys.path.insert(0,'/storage/emulated/0/github/cbse-papers/CoordGeo')         #path to my scripts
sys.path.insert(0,'/sdcard/github/cbse-papers/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Constants

I = np.eye(2)
e1 = I[:,0].reshape(2,1)
e2 = I[:,1].reshape(2,1)
R = np.array(([0,-1],[1,0]))
h = np.array(([4],[5]))
u = -2*np.array(([2],[1]))
f = -11

#kap
f0 = u.T@u - f

#Quadratic
coeff = [41, -10, -495]
mu = np.roots(coeff)#numerical

n1 = 1/4*(e1 - mu[0] * R@h)
n2 = 1/4*(e1 - mu[0] * R@h)

kap1 = np.sqrt(f0/LA.norm(n1)**2)
kap2 = np.sqrt(f0/LA.norm(n2)**2)

qn11 = kap1*n1-u
qn12 = -kap1*n1-u
qn21 = kap2*n2-u
qn22 = -kap2*n2-u

#output

print(mu)
#print(e1)

##Input parameters
#A = np.array(([-6,3]))
#B = np.array(([6,4]))
#
#Centre and radius
O = -u
r = f0

print(r)
##
###Generating all lines
#x_AB = line_gen(A,B)
#
#Generating the circle
x_circ= circ_gen(O,r)

##Plotting all lines
#plt.plot(x_AB[0,:],x_AB[1,:],label='$Diameter$')
#
#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
#
#
##Labeling the coordinates
#tri_coords = np.vstack((A,B,O)).T
#plt.scatter(tri_coords[0,:], tri_coords[1,:])
#vert_labels = ['A','B','O']
#for i, txt in enumerate(vert_labels):
#    plt.annotate(txt, # this is the text
#                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
#                 textcoords="offset points", # how to position the text
#                 xytext=(0,10), # distance from text to points (x,y)
#                 ha='center') # horizontal alignment can be left, right or center
#
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-3.pdf')
subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/cbse-10-3.pdf"))
##else
##plt.show()
#
#
#
#
#
#
#
