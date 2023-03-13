from math import *
import numpy as np
import sympy as sp
from numpy import poly1d
import matplotlib.pyplot as plt
from numpy import linalg as LA
tan_theta=1/3
cos_theta=3/sqrt(1**2+3**2)
print("Given cos theta = ",cos_theta)
def slope(m):
    m1=np.array([1,m])
    m2=np.array([1,2*m])
    num=m1.T@m2
    den=LA.norm(m1)*LA.norm(m2)
    cos=num/den
    return cos
print("##########    CASE-1    ##########")
print("cos from the function = ",slope(-1/2))
print("##########    CASE-2    ##########")
print("cos from the function = ",slope(1/2))
print("##########    CASE-3    ##########")
print("cos from the function = ",slope(-1))
print("##########    CASE-4    ##########")
print("cos from the function = ",slope(1))
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
#plotting part	
import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/DCIM/Assignment-4/CoordGeo')         #path to my scripts
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from math import *
#if using termux
import subprocess
import shlex
from math import *
#end if	

#Input parameters
A = np.array(([1,1],[2,1]))
B = np.array(([4,7]))
e1 = np.array(([1,0]))
n1 = A[0,:]
n2 = A[1,:]
c1 = B[0]
c2 = B[1]

l = LA.solve(A,B)
#r = 7
#Direction vectors
m1 = omat@n1
m2 = omat@n2

#Points on the lines
x1 = c1/(n1@e1)
A1 =  x1*e1
x2 = c2/(n2@e1)
A2 =  x2*e1
#x_circ= circ_gen(O,r)
#Generating all lines
k1=-5
k2=5
x_AB = line_dir_pt(m1,A1,k1,k2)
x_CD = line_dir_pt(m2,A2,k1,k2)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='x + y = 4')
plt.plot(x_CD[0,:],x_CD[1,:],label='2x + y = 7')

#Labeling the coordinates
#tri_coords = np.vstack((x)).T
tri_coords = l.T
#plt.scatter(tri_coords[0,:], tri_coords[1,:])
plt.scatter(tri_coords[0], tri_coords[1])
vert_labels = ['(3,1)']
#plt.plot(x_circ[0,:],x_circ[1,:],label='x^2+y^2−2x+2y=47')
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0], tri_coords[1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

	
