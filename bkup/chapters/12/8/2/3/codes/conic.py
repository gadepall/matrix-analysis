#Python libraries for math and graphics
import numpy as np
import math 
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.integrate import quad
import mpmath as mp

import sys, os                                          #for path to external scripts
sys.path.insert(0,'/home/sinkona/Documents/CoordGeo') 

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if

#Points of Intersection
V=np.array([[1,0],[0,0]])
u=np.array([0,-1/2])
e1=np.array([1,0])
e2=np.array([0,1])
f=2
a1=e2.T@V@e2
b1=2*u.T@e2
c1=f
a2=e2.T@V@e2
b2=3*e2.T@V@e1+3*e1.T@V@e2+2*u.T@e2
c2=9*e1.T@V@e1+6*u.T@e1+2

y1=np.roots([a1,b1,c1])
y2=np.roots([a2,b2,c2])

x1=0 #x=0
x2=3 #x=3
x3=3 
y3=3
x4=0
y4=0

q1=np.array([x1,y1]) #intersection of line x=0 with given parabola y=x*2+2
q2=np.array([x2,y2]) #intersection of line x=3 with given parabola y=x*2+2
q3=np.array([x3,y3]) #intersection of line x=0 with line y=x
q4=np.array([x4,y4]) #intersection of line x=3 with line y=x

#Input parameters for parabola
x = np.linspace(-4, 4, 1000)
y = (x ** 2) + 2
plt.plot(x, y, label='y=$x^2$+2')

#Input parameters for lines
y1=np.linspace(-3,13,1000)  #x=0
x1=np.zeros(len(y1))
plt.plot(x1,y1,label='x=0')

y2=np.linspace(-3,13,1000)  #x=3
x2=3*np.ones(len(y1))
plt.plot(x2,y2,label='x=3')

x3=np.linspace(-4,4,1000)   #x=y
y3=x3
plt.plot(x3,y3,label='x=y')

#Labeling the coordinates
plot_coords = np.vstack((q1, q2, q3, q4)).T
plt.scatter(plot_coords[0,:], plot_coords[1,:])
vert_labels = ['A[0,2]', 'B[3,11]', 'C[3,3]', 'O[0,0]']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (plot_coords[0,i], plot_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(15,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

#Finding the area 
def integrand1(x):
   return ((x**2)+2)
a1,err=quad(integrand1, 0, 3)
def integrand1(x):
   return (x)
a2,err=quad(integrand1, 0, 3)
A=a1-a2
print("Area of the bounded region",A)

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid(True) # minor
plt.legend(loc='upper left')
plt.axis([-6,4,-5,15])
plt.show()

#if using termux
#plt.savefig('/sdcard/Download/fwc-main/Assignment1/conics/conic1.pdf')
#subprocess.run(shlex.split("termux-open '/sdcard/Download/fwc-main/Assignment1/conics/conic1.pdf'"))    
#else
