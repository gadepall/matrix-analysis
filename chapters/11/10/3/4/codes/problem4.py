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

#points to plot line 12(x+6) = 5(y-2)
Q = np.array(([-6,2]))
R = np.array(([0,82/5]))

c = -82

#directional vector for the line
m = np.array(([5,12]))
#Normal vector to the above line
n = np.array(([12,-5]))
#Point from where the distance to be calculated
P = np.array(([-1,1]))

#Distance from P to line QR
d = (abs(np.dot(n,P)-c))/np.linalg.norm(n)
print("Distance from P(-1,1) to the line is ", d)

#Foot of the perpendicular
mn_matr = np.block([[m],[n]])
b = np.array(([np.dot(m,P),c]))
A = np.linalg.solve(mn_matr.T, b) # A is foot of perpendiular


x_QR = line_gen(Q,R)
x_AP = line_gen(A,P)

#Plotting all lines
plt.plot(x_QR[0,:],x_QR[1,:],label='$QR$') # Given line
plt.plot(x_AP[0,:],x_AP[1,:],label='$AP$') # Perpendicular line from P to Foot of perpendicular 

#Labeling the coordinates
tri_coords = np.vstack((Q,R,P,A)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['Q','R','P','A']
for i, txt in enumerate(vert_labels):
    label = "{}({:.2f},{:.2f})".format(txt, tri_coords[0,i],tri_coords[1,i]) #Form label as A(x,y)
    plt.annotate(label, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(25,3), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.title('Line')
#if using termux
plt.savefig('../figs/problem4.pdf')
#else
#plt.show()
