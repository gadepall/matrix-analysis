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

def dir_vec(A,B):
   return B-A
def norm_vec(A,B):
   return np.matmul(omat, dir_vec(A,B))
#Given points
A = np.array(([0,0]))
B = np.array(([4,0]))
d=int(input('Enter the length of one side of paralelogram: '))  #length of one side in parallelogram
r=int(input('Enter the other side: ')) #second length 
theta=math.pi/6 #angle between two sides
D=np.array(([r*mp.cos(theta),r*mp.sin(theta)]))
C=D+B
O=(A+C)/2  #mid point of parallelogram
print("area of AOB")
f=np.linalg.norm(np.cross(O-A,O-B))
print(f/2)
print("area of BOC")
f=np.linalg.norm(np.cross(B-O,C-O))
print(f/2)
print("area of COD")
f=np.linalg.norm(np.cross(C-O,D-O))
print(f/2)
print("area of DOA")
f=np.linalg.norm(np.cross(A-O,D-O))
print(f/2)
print("Hence the areas of triangles in a parallelogram formed by the diagnols are equal")
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_AC = line_gen(A,C)
x_AD = line_gen(A,D)
x_BD = line_gen(B,D)
x_AO = line_gen(A,O)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.plot(x_AC[0,:],x_AC[1,:],label='$AC$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
plt.plot(x_BD[0,:],x_BD[1,:],label='$BD$')
plt.plot(x_AO[0,:],x_AO[1,:],label='$AO$')


#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','O']
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
plt.title('parallelogram')
#if using termux
plt.savefig('/sdcard/Linearalgebra/par.pdf')
#subprocess.run(shlex.split("termux-open '/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-2.pdf'")) 
#else
#plt.show()
