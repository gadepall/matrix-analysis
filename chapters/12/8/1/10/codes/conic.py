#Python libraries for math and graphics
import numpy as np
import sympy as sy
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import quad

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/IITH/Assignment-1/MATRICES/CoordGeo')

#local imports
from line.funcs import *
from triangle.funcs import *
#from conics.funcs import circ_gen
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if

def affine_transform(P,c,x):
    return P@x + c

I =  np.eye(2)
e1 =  I[:,0]
#Input parameters

#line parameters
n =  np.array(([1,-4])) #normal vector
m =  omat@n #direction vector
print(m)
c = -2
q = c/(n@e1)*e1 #x-intercept
print(q)

#circle parameters
V = np.array(([1,0],[0,0]))
u = np.array(([0,-2]))
f = 0

#Points of intersection of line with curve
x1,x2 = inter_pt(m,q,V,u,f)
print(x1,x2)

#for computing stright line
lamda,P = LA.eigh(V)
#print(P)
#print("lamda is",lamda)
if(lamda[1] == 0):  # If eigen value negative, present at start of lamda
    lamda = np.flip(lamda) # e value
    P = np.flip(P,axis=1)   #e vectors in col
#print(P)

#for parabola
lamda,P = LA.eigh(V)
if(lamda[1] == 0):  # If eigen value negative, present at start of lamda
    lamda = np.flip(lamda) # e value
    P = np.flip(P,axis=1)   #e vectors in col

#
eta = u@P[:,0]
a = np.vstack((u.T + eta*P[:,0].T, V))
b = np.hstack((-f, eta*P[:,0]-u))
center = LA.lstsq(a,b,rcond=None)[0]
O = np.array([0,0])
n = np.sqrt(lamda[1])*P[:,0]
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)
F = np.array(([0,0.7]))
fl = LA.norm(F)

#pmeters to generate parabola
num_points = 1700
delta = 20*np.abs(fl)/10
p_y = np.linspace(-2*np.abs(fl)-delta,2*np.abs(fl)+delta,num_points)
a = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y


##Generating all shapes
p_x = parab_gen(p_y,a)
p_std = np.vstack((p_x,p_y)).T

#x_circ= circ_gen(O,r)

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T
plt.plot(p[0,:], p[1,:])


#Plotting the circle
#plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')


#area
def f(x):
    return (x+2)/4

def g(x):
    return (x**2)/4

x = sy.Symbol('x')
area=sy.integrate(f(x) - g(x), (x, x2.T@e1,x1.T@e1))
print(area)

#Generating  line
x_R1 = line_gen(x1,x2);


# use set_position
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')

#Plotting all line
plt.plot(x_R1[0,:],x_R1[1,:],label='$line$')

#Labeling the coordinates
tri_coords = np.vstack((x1,x2,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['x1','x2','O']
for i, txt in enumerate(vert_labels):
       plt.annotate(txt,      # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points",   # how to position the text
                 xytext=(0,10),     # distance from text to points (x,y)
                 ha='center')     # horizontal alignment can be left, right or center


#plt.fill_between(x1,x2,O,color='green', alpha=.2)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid(True) # minor
plt.axis('equal')
#plt.savefig('conics1.png')
#plt.show()

plt.savefig('/sdcard/IITH/Assignment-1/MATRICES/Conic/conicplot.pdf')
subprocess.run(shlex.split("termux-open /sdcard/IITH/Assignment-1/MATRICES/Conic/conicplot.pdf"))
