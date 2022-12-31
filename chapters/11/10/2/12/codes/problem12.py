import numpy as np
import math
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
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

#Given points
P = np.array(([5,0]))
Q = np.array(([0,5]))
A = np.array(([2,3]))

x_PQ = line_gen(P,Q)


#Plotting all lines
plt.plot(x_PQ[0,:],x_PQ[1,:],label='$PQ$')

#Labeling the coordinates
tri_coords = np.vstack((P,Q,A)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','Q','A']
for i, txt in enumerate(vert_labels):
    label = "{}({},{})".format(txt, tri_coords[0,i],tri_coords[1,i]) #Form label as A(x,y)
    plt.annotate(label, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(15,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.title('Line')
#if using termux
plt.savefig('../figs/problem12.pdf')
#else
#plt.show()
