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
#x = np.linspace(1,5,len)
y = np.linspace(-1,2,len)

#parab parameters
V = np.array(([1,0],[0,0]))
u = -np.array(([2,1/2]))
f = 4
p = np.array(([0,1]))
foc = np.abs(p@u)/2

#Eigenvalues and eigenvectors
#D_vec,P = LA.eig(V)
#D = np.diag(D_vec)
#print(D,P)
P = np.array(([0,1],[1,0]))
print(P)
#Generating the Standard parabola
x = parab_gen(y,foc)
#y =(x-2)**2
xStandardparab = np.vstack((x,y))
#
#Affine Parameters
R =  np.array(([0,1],[1,0]))
cA = np.vstack((u-2*p,V))
cb = np.vstack((-f,(-(2*p+u)).reshape(-1,1)))
c = LA.lstsq(cA,cb,rcond=None)[0]
c = c.flatten()
print(c)
xStandardparab = np.vstack((x,y))
xActualparab = P@xStandardparab + c[:,np.newaxis]
#
#Tangent Analysis
sec1 = np.array(([4,4]))
sec2 = np.array(([2,0]))
m = sec1-sec2
n = omat@m
#print(n)
kappa = p@u/(p@n)

qA = np.vstack((u+kappa*n,V))
qb = np.vstack((-f,(kappa*n-u).reshape(-1,1)))
q = LA.lstsq(qA,qb,rcond=None)[0]
q = q.flatten()
#O = np.array([0,0])
#print(p@u,p@n)
#print(kappa,qA,qb)

#Generating the tangent
k1 = 2
k2 = -2
x_AB = line_dir_pt(m,q,k1,k2)
#
#Generating the secant
x_sec12 = line_dir_pt(m,sec1,k1,k2)
#Labeling the coordinates
parab_coords = np.vstack((q,sec1,sec2)).T
plt.scatter(parab_coords[0,:], parab_coords[1,:])
vert_labels = ['$q$','$A$','$B$']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (parab_coords[0,i], parab_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='Tangent')

#Plotting all lines
plt.plot(x_sec12[0,:],x_sec12[1,:],label='Secant')

#Plotting the actual parabola
#plt.plot(xStandardparab[0,:],xStandardparab[1,:],label='Parabola',color='r')
plt.plot(xActualparab[0,:],xActualparab[1,:],label='Parabola',color='r')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('./figs/parab/parab_tangent_secant.pdf')
plt.savefig('./figs/parab/parab_tangent_secant.png')
subprocess.run(shlex.split("termux-open ./figs/parab/parab_tangent_secant.pdf"))
##else
##plt.show()
