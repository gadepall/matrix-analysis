#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import sys       #for path to external scripts
sys.path.insert(0,'/sdcard/Download/parv/CoordGeo')

#local imports
from conics.funcs import circ_gen


#if using termux
import subprocess
import shlex

def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB

A = np.array([-2,2])
B = np.array([4,2])
C = np.array([-2,-2])
D = np.array([4,-2])


c = np.array([1,0])
r = 2
n = np.array([0,1])
u = np.array([-1,0])



#point of contact
P1 = r*n - u
P2 = -r*n - u

##Generating the circle
x_circ= circ_gen(c,r)

#generating lines
x_AB = line_gen(A,B)
x_CD = line_gen(C,D)


#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
#Labeling the coordinates
tri_coords = np.vstack((c))
plt.scatter(tri_coords[0],tri_coords[1])


#Labeling the coordinates
tri_coords = np.vstack((P1,P2,c)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P(1,2)','Q(1,-2)','C(1,0)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center                
plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
print("The point of contact are")
print(P1,P2)
plt.savefig('/sdcard/Download/latexfiles/tangent/figs/tan1.png')
plt.show()
