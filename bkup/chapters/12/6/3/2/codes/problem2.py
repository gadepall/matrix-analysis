#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import *

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/sat/CoordGeo')
#local imports
#from conics.funcs import parab_gen
#if using termux
import subprocess
import shlex
#end if

def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB

def func(x):
    return (x-1)/(x-2)

#Input parameters
V = np.array([[0,0.5],[0.5,0]])
u = np.array(([-0.5,-1]))
f = 1


num_points = 500
p_x = np.linspace(-2,11,num_points)
p_y = func(p_x) 

#Point of contact
q = np.array(([10,func(10)]))
n = V@q + u
m = np.array(([1, -n[0]/n[1]]))
c = u@q+f
print("Slope of the Tangent is ", m)

tangent_y = m[1]*p_x - c/4

#Plotting all shapes
plt.plot(p_x,p_y,label ='$y=(x-1)/(x-2)$')
plt.plot(p_x,tangent_y,label ='$Tangent$')

plt.scatter(q[0], q[1])
label = "{}({:.0f},{:.1f})".format('A', q[0],q[1]) #Form label as A(x,y)
plt.annotate(label, # this is the text
            (q[0], q[1]), # this is the point to label
            textcoords="offset points", # how to position the text
            xytext=(-18,5), # distance from text to points (x,y)
            ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best', fontsize = 'small')
plt.grid() # minor
plt.title('Tangent to a Hyperbola')
#if using termux
plt.savefig('../figs/problem2.pdf')
plt.show()
