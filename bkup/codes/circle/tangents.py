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
from conics.funcs import *
#from conics.funcs import circ_gen

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
u = -np.array(([2],[1]))
f = -11

#kap
f0 = u.T@u - f

#Quadratic
coeff = [41, -10, -495]
mu = np.roots(coeff)#numerical

m1 = R@h
u1 = np.zeros((2,1))
f1 = -f0
q1 = e1/(e1.T@h)
V1 = I
[mu1,mu2] = inter_pt(m1.T,q1.T,V1,u1.T,f1)

n1 = 1/4*(e1 + mu[0] * R@h)
n2 = 1/4*(e1 + mu[1] * R@h)

kap1 = np.sqrt(f0/LA.norm(n1)**2)
kap2 = np.sqrt(f0/LA.norm(n2)**2)

q11 = kap1*n1-u
q12 = -kap1*n1-u
q21 = kap2*n2-u
q22 = -kap2*n2-u

#print(q11,q12,q21,q22)
#print(n2.T@h)

#output

#print(mu)
#print(e1)

##Input parameters
#A = np.array(([-6,3]))
#B = np.array(([6,4]))
#
#Centre and radius
O = -u
r = np.sqrt(f0)

#print(r)
##
###Generating all lines
#x_AB = line_gen(A,B)
#
#Generating the circle
x_circ= circ_gen(O.T,r)

##Plotting all lines
#plt.plot(x_AB[0,:],x_AB[1,:],label='$Diameter$')
#
#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
#
#
#Labeling the coordinates
tri_coords = np.vstack((q11.T,q12.T,q21.T,q22.T,O.T,h.T)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['q11','q12','q21','q22','O','h']
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
plt.savefig('/sdcard/github/matrix-analysis/figs/circle/tangest.pdf')
subprocess.run(shlex.split("termux-open /sdcard/github/matrix-analysis/figs/circle/tangest.pdf"))
