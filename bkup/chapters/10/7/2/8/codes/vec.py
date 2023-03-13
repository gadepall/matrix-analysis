import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sys     #for path to external scripts





#Two aray vectors are given 
A = np.array(([-2, -2]))
B = np.array(([ 2, -4]))
n =3/4
P = (([-0.28,-2.85]))
#Formula of an 
P = (A+n*B)/(1+n)

print('P=',P)

def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB

  
x_AP = line_gen(A,P)
x_PB = line_gen(P,B)



#Plotting all lines
plt.plot(x_AP[0,:],x_AP[1,:],label='$AP$')
plt.plot(x_PB[0,:],x_PB[1,:],label='$PB$')



#Labeling the coordinates
tri_coords = np.vstack((A,P,B)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A''(-2,-2)','P''(-0.28,-2.85)','B''(2,-4)']
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
plt.title('trapezium')

plt.show()
