import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
from pylab import *
from sympy import *

import sys                                          #for path to external scripts

#sys.path.insert(0,'/home/aluru-ajay99/CoordGeo')         #path to my scripts


sys.path.insert(0, '/sdcard/github/cbse-papers/CoordGeo')        #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from conics.funcs import *


#if using termux
import subprocess
import shlex
#end if
#for line

n2=np.array(([-2,1]))
n3=np.array(([12,36]))

c2=3
c3=227
e1=np.array(([1,0]))

x3 = c2/(n2@e1) #X-intercept
A3 =  x3*e1
x4 = c3/(n3@e1) #X-intercept
A4 =  x4*e1
#Direction vector

m3=omat@n2
m4=omat@n3
#Generating all lines
k1 = -10
k2 = 8

EF = line_dir_pt(m3,A3,k1,k2)
GH = line_dir_pt(m4,A4,k1,k2)

#for parabola
V = np.array([[1,0],[0,0]])
u = np.array(([-1,-0.5]))
f = 7

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
print(center)
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


p_x = parab_gen(p_y,a)
p_std = np.vstack((p_x,p_y)).T

##Affine transformation
p = np.array([affine_transform(P,center,p_std[i,:]) for i in range(0,num_points)]).T
plt.plot(p[0,:], p[1,:])



#Plotting all lines
plt.plot(EF[0,:],EF[1,:],label='$line2$')
plt.plot(GH[0,:],GH[1,:],label='$line1$')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')
plt.legend(loc='best')

plt.xlim(-5,5)
plt.ylim(0,18)

#if using termux
#plt.savefig(os.path.join(script_dir, fig_relative))
#subprocess.run(shlex.split("termux-open "+os.path.join(script_dir, fig_relative)))
#else
plt.show()
#plt.savefig('/sdcard/Ajay/matrix/conic/conicfig.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/Ajay/matrix/conic/conicfig.pdf"))



