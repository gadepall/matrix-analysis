import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sys     #for path to external scripts





#Two aray vectors are given 
Q = np.array(([4, -1]))
P = np.array(([ -2, -3]))
n =1/2
R = (([2,-1.6]))
#Formula of an 
R = (Q+n*P)/(1+n)

print("R=", R)

#Two array vectors are given
Q = np.array(([4,-1]))
P = np.array(([-2,-3]))
n = 2
S = np.array(([0,-2.3]))
#Formula of an
S = (Q+n*P)/(1+n)

print('S=',S)

def line_gen(Q,P):
   len =10
   dim = Q.shape[0]
   x_QP = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = Q + lam_1[i]*(P-Q)
     x_QP[:,i]= temp1.T
   return x_QP

  
x_QR = line_gen(Q,R)
x_RS = line_gen(R,S)
x_SP = line_gen(S,P)


#Plotting all lines
plt.plot(x_QR[0,:],x_QR[1,:],label='$QR$')
plt.plot(x_RS[0,:],x_RS[1,:],label='$RS$')
plt.plot(x_SP[0,:],x_SP[1,:],label='$SP$')


#Labeling the coordinates
tri_coords = np.vstack((Q,R,S,P)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['Q''(4,-1)','R''(2,-1.6)','S''(0,-2.3)','P''(-2,-3)']
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
