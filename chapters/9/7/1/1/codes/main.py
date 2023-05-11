import numpy as np 
import matplotlib.pyplot as plt 
from numpy import linalg as LA 
import math 
import sys

A = np.loadtxt('a.dat',dtype='float')
B = np.loadtxt('b.dat',dtype='float')
C = np.loadtxt('c.dat',dtype='float')
D = np.loadtxt('d.dat',dtype='float')

def line_gen(A,B):
    len=10
    dim = A.shape[0]
    x_AB = np.zeros((dim,len))
    lam_1 = np.linspace(0,1,len)
    for i in range(len):
        temp1 = A + lam_1[i]*(B-A)
        x_AB[:,i] = temp1.T
    return x_AB

x_AB = line_gen(A,B)
x_BD = line_gen(B,D)
x_AD = line_gen(A,D)
x_CA = line_gen(C,A)
x_BC = line_gen(B,C)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BD[0,:],x_BD[1,:],label='$BD$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')

line1 = [[2.6, 1.1], [2.3, 1.6]]  # First line from X to Y
line2 = [[2.6, -1.1], [2.3, -1.6]]  # Second line from Y to Z
plt.plot([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], 'r-')  # First line
plt.plot([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], 'g-')  # Second line

#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$xaxis$')
plt.ylabel('$yaxis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.title('Quadrilateral ABCD')
plt.savefig('/sdcard/arduino/Vector/figs/fig.png')
plt.show()


