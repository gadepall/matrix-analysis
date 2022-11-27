import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
from pylab import *

import sys, os                                         #for path to external scripts
sys.path.insert(0,'/home/krishna/Krishna/python/codes/CoordGeo') 


#local imports
#local imports
from line.funcs import *
from triangle.funcs import *
#from conics.funcs import circ_gen
from conics.funcs import *


#if using termux
import subprocess
import shlex
#end if

#for parabola
V = np.array([[0,0],[0,1]])
u = np.array(([-2,0]))
f = 0
q=np.array(([1,2]))
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
#c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)
F = np.array(([0,0.5]))
fl = LA.norm(F)

#pmeters to generate parabola
num_points = 1700
delta = 50*np.abs(fl)/10
p_y = np.linspace(-2*np.abs(fl)-delta,2*np.abs(fl)+delta,num_points)
a = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y


p_x = parab_gen(p_y,a)



#pmeters to generate parabola
num_points = 1700
delta = 50*np.abs(fl)/10
p_y = np.linspace(-2*np.abs(fl)-delta,2*np.abs(fl)+delta,num_points)
a = -2*eta/lamda[1]   # y^2 = ax => y'Dy = (-2eta)e1'y


#p_x = parab_gen(p_y,a)


#Affine transformation
plt.plot(p_x, p_y)

y = np.linspace(-5,5,100)
x = (y**2)/4
plt.plot(x,y,label='Parabola')
#tangent
x1 = y-1
plt.plot(x1,y,label='Tangent')
#Normal
x2 = -y+3
plt.plot(x2,y,label='Normal')


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')
plt.legend(loc='best') 
#if using termux
plt.savefig('/home/krishna/Krishna/python/figs/conic.pdf')
#subprocess.run(shlex.split("termux-open "/home/krishna/Krishna/python/figs/conic.pdf'))
#else
plt.show()

