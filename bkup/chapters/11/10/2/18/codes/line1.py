import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import subprocess
import shlex


####plotting part####

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
P = np.array(([8,0]))
Q = np.array(([0,6]))
R = np.array(([(P[0]+Q[0])/2,(P[1]+Q[1])/2]))


x_PQ = line_gen(P,Q)
x_PR = line_gen(P,R)
x_QR = line_gen(Q,R)

#Plotting all lines
plt.plot(x_PQ[0,:],x_PQ[1,:],label='$AB$')
plt.plot(x_PR[0,:],x_PR[1,:],label='$AP$')
plt.plot(x_QR[0,:],x_QR[1,:],label='$BP$')

#Labeling the coordinates
tri_coords = np.vstack((P,Q,R)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A(x,0)','B(0,y)','P(a,b)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')
#plt.title('equation of straight line')
plt.savefig('/sdcard/download/latexfiles/line/figs/line1.png')  
plt.show()
