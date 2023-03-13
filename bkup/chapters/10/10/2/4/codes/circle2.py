
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


import sys, os
script_dir = os.path.dirname(__file__)
lib_relative = '../../coord/circle'
sys.path.insert(0,os.path.join(script_dir, lib_relative))

#local imports 
from funcir import *
##set center and radius of circle
r = 10 
O = np.array(([0,0]))
t = 45
#generate circle:
x_circ = circ_gen(O,r)

#plot circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
A=np.array(([r*(np.cos(t)),r*(np.sin(t))]))
B= O-A
l1 = -A[0]/A[1]
l2 = -B[0]/B[1]
m1 = np.array(([1,l1]))
m2 = np.array(([1,l2]))
#generate tangents
c = r
e1 = np.array(([1,0]))
n1 = A/LA.norm(A)
n2 = B/LA.norm(B)
#n1 = np.array(([1,0]))
#n2 = np.array(([-1,0]))
omat = np.array([[0,1],[-1,0]])
m1 = omat@n1
m2 = omat@n2
x1 = c/(n1@e1)
A1 = x1*e1
x2 = c/(n2@e1)
A2 = x2*e1
k1 = -40
k2 = 10
x_line_1 = line_dir_pt(m1,A1,k1,k2)
x_line_2 = line_dir_pt(m2,A2,k1,k2)
plt.plot(x_line_1[0,:],x_line_1[1,:])
plt.plot(x_line_2[0,:],x_line_2[1,:])

#generate diameter
x_AB = line_gen(A,B)
#plot diameter
plt.plot(x_AB[0,:],x_AB[1,:])

#Labeling the coordinates
cir_coords = np.vstack((A,B,O)).T
plt.scatter(cir_coords[0,:], cir_coords[1,:])
vert_labels = ['A','B','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (cir_coords[0,i], cir_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#plt.savefig('../figs/plot_cir.png')
o = np.array(([0,0]))
a = 10
#r = 10
product = np.cross(m1,m2) 
print(m1)
print(m2)
print('cross product of the tangent directional vectors is', product)

