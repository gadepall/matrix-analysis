#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys
lib_path = '/sdcard/fwc/matrices/CoordGeo'
sys.path.insert(0,lib_path)

#local imports
from line.funcs import *
from triangle.funcs import *

#if using termux
import subprocess
import shlex
#end if

#Generating points on a standard hyperbola 
def hyper_gen(y, a, b):
    x = np.sqrt(1+(y**2)/(b**2))*a
    return x

# Function to generate normal for hyberbola at P given array y
def normal_gen(y, a, b, P):
    x = (P[0]/(a**2))*(a**2 + b**2 - ((b**2)/P[1])*y)
    return x

# Function to generate tangent for hyberbola at P given array y
def tangent_gen(y, a, b, P):
    x = ((a**2)/P[0])*(1+(P[1]/(b**2))*y)
    return x

a = 2
b = 3
P = np.array([2*np.sqrt(2), 3])

# Plotting 4 sections of hyperbola individually
y = np.linspace(0.01,5,100)
x = hyper_gen(y, a, b)
plt.plot(x, y, color='blue')

y = np.linspace(-5,-0.01,100)
x = hyper_gen(y, a, b)
plt.plot(x, y, color='blue')

y = np.linspace(0.01,5,100)
x = -hyper_gen(y, a, b)
plt.plot(x, y, color='blue')

y = np.linspace(-5,-0.01,100)
x = -hyper_gen(y, a, b)
plt.plot(x, y, color='blue')

# Plotting Normal
y = np.linspace(-2, 4, 100)
x = normal_gen(y, a, b, P)
plt.plot(x, y, color='red')

# Plotting Tangent
y = np.linspace(-4, 4, 100)
x = tangent_gen(y, a, b, P)
plt.plot(x, y, color='green')

# Plotting point P
plt.plot(P[0], P[1], marker='o', markersize=7)

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')

plt.show()
plt.savefig('/sdcard/fwc/matrices/CoordGeo/conic_assignment/hyperbola.pdf')
