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
A = np.array(([1,5]))
B = np.array(([2,3]))
C = np.array(([-2,-11]))

#Form the matrix with transposes of A,B,C
mat = np.block([[A],[B], [C]])

rankOfMatrix = np.linalg.matrix_rank(mat)

if(rankOfMatrix == 1):
    print("The rank of the  matrix is ", rankOfMatrix, ". Hence the points are collinear")
else:
    print("The rank of the matrix is ", rankOfMatrix, ". Hence the points are not collinear")



x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_AC = line_gen(A,C)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_AC[0,:],x_AC[1,:],label='$AC$')


#Labeling the coordinates
tri_coords = np.vstack((A,B,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C']
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
plt.title('Triangle')
#if using termux
plt.savefig('../figs/problem3.pdf')
#else
#plt.show()
