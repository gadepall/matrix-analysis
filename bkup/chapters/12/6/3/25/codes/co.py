import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
from pylab import *
from sympy import *

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/srinath/lines/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from conics.funcs import *


#if using termux
import subprocess
import shlex
#end if

#for parabola
V = np.array([[0,0],[0,1]])
u = np.array(([-1.5,0]))
f = 2

C = np.array(([0,0]))
r = np.array(([1,2])) #h
s = np.array(([2,1])) #q=h+lam(m)

def affine_transform(P,c,x):
    return P@x + c

#Transformation 
lamda,P = LA.eigh(V)
if(lamda[1] == 0):  # If eigen value negative, present at start of lamda 
    lamda = np.flip(lamda)
    P = np.flip(P,axis=1)
    
eta = u@P[:,0]
a = np.vstack((u.T + eta*P[:,0].T, V))
b = np.hstack((-f, eta*P[:,0]-u)) 
center = LA.lstsq(a,b,rcond=None)[0]
O = center 
n = np.sqrt(lamda[1])*P[:,0]
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)
F = np.array(([0,0.5]))
fl = LA.norm(F)

#pmeters to generate parabola
num_points = 1700
delta = 50*np.abs(fl)/10
p_y = np.linspace(-2*np.abs(fl)-delta,2*np.abs(fl)+delta,num_points)
a = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y


y = np.linspace(-5,5,100)
x = (y**2+2)/3
plt.plot(x,y)

x1 = (2*y-5)/4
plt.plot(x1,y)

x2 = (24*y+23)/48
plt.plot(x2,y)

p_x = parab_gen(p_y,a)
p_std = np.vstack((p_x,p_y)).T

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T
plt.plot(p[0,:], p[1,:])

#Computation
x = Symbol('x')
# m is the vector perpendicular to normal chord ie m^tx = c
m2 = Matrix([-x, 1])
omat = Matrix(([0, -1], [1, 0]))
print('omat:',omat)
# n is the vector along the normal chord ie h + kn = x 
n2 = omat*m2
# Conic parameters
V2 = Matrix(([1,0],[0,0]))
print('v:',V)
u2 = Matrix([0, -2])
f2 = Matrix([-3])
# Point from which normal drawn
h2 = Matrix([1,2])
# Equation solving
eq1 = n2.T*((V2*h2 + u2)*(V2*h2 + u2).T - (h2.T*V2*h2 + 2*u2.T*h2 + f2)[0,0]*V2)*n2
eq2 = (m2.T*V2*n2)**2
eq3 = ((V2*h2 + u2).T*(n2*(m2.T*V2*n2) - m2*(n2.T*V2*n2)))**2
eq = eq1[0,0]*eq2[0,0] - eq3[0,0]
print(expand(eq))
print(solveset(eq, x))

#GeneratingLine
#x_RS = line_gen(r,s) #Normal
#plt.plot(x_RS[0,:],x_RS[1,:],label='$Normal$')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')



#Labeling the coordinates
#tri_coords = np.vstack((C)).T
#tri_coords = x.T
#plt.scatter(tri_coords[0], tri_coords[1])
#vert_labels = ['']
#for i, txt in enumerate(vert_labels):
 #   plt.annotate(txt, # this is the text
  #               (tri_coords[0], tri_coords[1]), # this is the point to label
   #              textcoords="offset points", # how to position the text
    #             xytext=(0,10), # distance from text to points (x,y)
     #            ha='center') # horizontal alignment can be left, right or center
plt.show()

