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
A = np.array(([-1,-1]))
B = np.array(([-1,4]))
C = np.array(([5,4]))
D = np.array(([5,-1]))

#mid points
P = (A+B)/2
Q = (B+C)/2
R = (C+D)/2
S = (D+A)/2

#We know that the figure formed by joining mid points of a quadrilateral is a parallelogram.
#To establish, if it is a rectangle, we will compute the dot product of any 2 adjacent sides
#if the dot product is zero, then it is rectangle

if ( np.dot(Q-P, R-Q) == 0):   #Check, if any 2 adjacent sides are orthogonal
    if (np.dot(R-P, S-Q) == 0): #If diagonals are orthogonal, then it is square
        print("PQRS is a Square")
    else:                      #If diagonals are not orthogonal, then it is rectangle
        print("PQRS is a rectangle")
else:                            
    if (np.dot(R-P, S-Q) == 0):   #if diagonals are orthogonal, then it is rhombus
        print("PQRS is a Rhombus")
    else:
        print("PQRS is a parallelogram") # Else it is parallelogram

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)

x_PQ = line_gen(P,Q)
x_QR = line_gen(Q,R)
x_RS = line_gen(R,S)
x_SP = line_gen(S,P)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.plot(x_DA[0,:],x_DA[1,:],label='$AD$')

plt.plot(x_PQ[0,:],x_PQ[1,:],label='$PQ$')
plt.plot(x_QR[0,:],x_QR[1,:],label='$QR$')
plt.plot(x_RS[0,:],x_RS[1,:],label='$RS$')
plt.plot(x_SP[0,:],x_SP[1,:],label='$PS$')

#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,P,Q,R,S)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C', 'D','P','Q','R','S']
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
plt.title('Rectangle')
#if using termux
plt.savefig('../figs/problem1.pdf')
#subprocess.run(shlex.split("termux-open '/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-2.pdf'")) 
#else
plt.show()
