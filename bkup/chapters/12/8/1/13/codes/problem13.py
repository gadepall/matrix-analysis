#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import *
from scipy.integrate import quad

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/sat/CoordGeo')
#local imports
from conics.funcs import *
from line.funcs import *
#if using termux
import subprocess
import shlex
#end if


#Points of intersection of a conic section with a line
m=np.array([1,0]);#direction vector
h= np.array([0,3]);
V=np.array([[0,0],[0,1]]);
u=np.array([-2,0]);
f=0;

g_of_h = h.T@V@h + 2*u.T@h+f

mu = -g_of_h/(2* ( m.T@(V@h + u)) )
a0 = h + np.array(([mu,0])) # Point of intersection with conic 
y1 = 3
#  plotting parabola
y = np.linspace(-5, 5, 100)
x = (y ** 2) / 4
plt.plot(x, y, label='Parabola')
plt.fill_between(x,y1,y,where= (0< y)&(y < 3), color = 'cyan', label = '$Area$')
def integrand1(y):
    return (y**2)/4
A1,err=quad(integrand1, 0,3)

print('the area between x=2 and x=4 bounded by the parabola is ',A1);
#Generating  line
p1=np.array([0,3]);
p2=np.array([6,3]);
p3=np.array([0,0]);
p4=np.array([0,7]);

x_R1 = line_gen(p1,p2);
x_R2 = line_gen(p3,p4);

plt.plot(x_R1[0,:],x_R1[1,:],label='$y=3$')
plt.plot(x_R2[0,:],x_R2[1,:],label='$x=0$')

plt.scatter(a0[0],a0[1])
plt.annotate('$a0$',      # this is the text
             (a0[0], a0[1]), # this is the point to label
                textcoords="offset points",   # how to position the text
                 xytext=(0,10),     # distance from text to points (x,y)
                 ha='center')     # horizontal alignment can be left, right or center
plt.legend(loc='best')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor

#if using termux
plt.savefig('../figs/problem13.pdf')
#else
plt.show()
