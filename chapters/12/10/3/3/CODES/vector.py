import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sys     #for path to external scripts





#Two aray vectors are given 
A = np.array(([1, -1]))
B = np.array(([ 1, 1]))
C = np.array(([0,0]))

#Formula of an 
C = ((np.dot(A,B))/(np.linalg.norm(B)**2)) *B

print("C=", C)




def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB

  
x_CA = line_gen(C,A)
x_CB = line_gen(C,B)



#Plotting all lines
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_CB[0,:],x_CB[1,:],label='$CB$')



#Labeling the coordinates
tri_coords = np.vstack((A,B,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A=(1,-1)','B=(1,1)','C=(0,0)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x_axis$')
plt.ylabel('$y_axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.title('trapezium')
 
plt.show()
