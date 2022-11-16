#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys, os                                          #for path to external scripts
script_dir = os.path.dirname(__file__)
lib_relative = '/sdcard/Download/CoordGeo'
sys.path.insert(0,os.path.join(script_dir, lib_relative))

#local imports
from line.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

# Returns normal vectors of tangents from external point p to conic (V,u,f)
def get_tangent_from_pt(V,u,f,p):
    sigma = (V@p+u)@(V@p+u).T - (p.T@V@p + 2*u.T@p + f)*V
    lamda, gamma = LA.eigh(sigma)
    n1 = gamma@np.array([np.sqrt(np.abs(lamda[0])), np.sqrt(np.abs(lamda[1]))])
    n1 = n1.reshape(2,1)
    n2 = gamma@np.array([np.sqrt(np.abs(lamda[0])), -np.sqrt(np.abs(lamda[1]))])
    n2 = n2.reshape(2,1)
    return n1,n2

# Returns the vector among n2_a and n2_b not parallel to n1
def get_non_parallel_vector(n1, n2_a, n2_b):
    if(LA.norm(np.cross(n1.reshape(2,), n2_a.reshape(2,))) < 0.0001):
        return n2_b
    else:
        return n2_a

# Input parameters
O = np.array([np.sqrt(2),0])
r = 1
d = np.sqrt(2) - 1
R = r + d
theta = 270*(np.pi/180)
delta1 = 2
delta2 = 3

h = np.array([R*np.cos(theta), R*np.sin(theta)]) + O
h = h.reshape(2,1)
V = np.array([[1, 0],[0, 1]])
u = -np.array([O[0], O[1]]).reshape(2,1)
f = LA.norm(O)**2-r**2
P1 = h

n1, n2 = get_tangent_from_pt(V,u,f,h)
m1 = omat@n1
m2 = omat@n2
T1 = h - ((m1.T@(V@h+u))/(m1.T@V@m1))*m1
T2 = h - ((m2.T@(V@h+u))/(m2.T@V@m2))*m2
m1 = T1 - h
dist = LA.norm(T1 - h)
k1 = dist*(1+delta1)  
P2 = h + (k1/LA.norm(m1))*m1    

n3_a, n3_b = get_tangent_from_pt(V,u,f,P2)
n3 = get_non_parallel_vector(n1, n3_a, n3_b)
PE_A = np.vstack((n2.T, n3.T))
PE_c = np.array([n2.T@P1, n3.T@P2]).reshape(2,1)
PE = LA.inv(PE_A)@PE_c

m2 = PE - T2
dist = LA.norm(PE - T2)
k2 = dist/(1+delta2) 
P3 = T2 + (k2/LA.norm(m2))*m2

n4_a, n4_b = get_tangent_from_pt(V,u,f,P3)
n4 = get_non_parallel_vector(n2, n4_a, n4_b)
P4_A = np.vstack((n3.T, n4.T))
P4_c = np.array([n3.T@P2, n4.T@P3]).reshape(2,1)
P4 = LA.inv(P4_A)@P4_c

P1 = P1.reshape(2,)
P2 = P2.reshape(2,)
P3 = P3.reshape(2,)
P4 = P4.reshape(2,)

#Generating all lines
x_P1P2 = line_gen(P1,P2)
x_P1P3 = line_gen(P1,P3)
x_P3P4 = line_gen(P3,P4)
x_P4P2 = line_gen(P4,P2)

##Generating the circle
x_circ= circ_gen(O,r)

#Plotting all lines
plt.plot(x_P1P2[0,:],x_P1P2[1,:])
plt.plot(x_P1P3[0,:],x_P1P3[1,:])
plt.plot(x_P3P4[0,:],x_P3P4[1,:])
plt.plot(x_P4P2[0,:],x_P4P2[1,:])
plt.plot(x_circ[0,:],x_circ[1,:])



#Labeling the coordinates
tri_coords = np.vstack((P1,P2,P3,P4,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','D','C','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
print("figure generated")
plt.savefig('/sdcard/Download/Matrices/circle/circle.png')
#subprocess.run(shlex.split("termux-open /storage/emulated/0/github/school/ncert-vectors/defs/figs/cbse-10-14.pdf"))
#else
#plt.show()
