import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sys     #for path to external scripts





#Two aray vectors are given 
A = np.array(([4, -6]))
B = np.array(([ 3, -2]))
C = np.array(([5, 2]))
#Formula of an 
D = (B+C)/2

print("D=", D)

#Two aray vectors are given 
A = np.array(([4, -6]))
B = np.array(([ 3, -2]))
C = np.array(([5, 2]))
#Formula of an 
P = (1/2)*np.linalg.norm((np.cross(A-B,A-D)))

print("P=", P)

Q=(1/2)*np.linalg.norm((np.cross(A-C,A-D)))

print("Q=", Q)

def line_gen(Q,P):
   len =10
   dim = Q.shape[0]
   x_QP = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = Q + lam_1[i]*(P-Q)
     x_QP[:,i]= temp1.T
   return x_QP

  
x_AB = line_gen(A,B)
x_AD = line_gen(A,D)
x_BC = line_gen(B,C)
x_AC = line_gen(A,C)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_AC[0,:],x_AC[1,:],label='$AC$')


#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A''(4,-6)','B''(3,-2)','C''(5,2)','D''(4,0)']
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
