import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

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
#if using termux
import subprocess
import shlex
#end if


#Given points


i=-2
j=-0.5
k=1
x=-1
A = np.array([i,4])
B = np.array([3,6])
D = np.array([j,k])
C = D+B-A
m= B-A
n= D-A
P = B-x*m
Q = B+x*n
R = P+Q-B



##Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)
x_BP = line_gen(B,P)
x_BQ = line_gen(B,Q)
x_QR = line_gen(Q,R)
x_PR = line_gen(P,R)
x_AQ = line_gen(A,Q)
x_CP = line_gen(C,P)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_BC[0,:],x_BC[1,:])
plt.plot(x_CD[0,:],x_CD[1,:])
plt.plot(x_DA[0,:],x_DA[1,:])
plt.plot(x_BP[0,:],x_BP[1,:])
plt.plot(x_BQ[0,:],x_BQ[1,:])
plt.plot(x_QR[0,:],x_QR[1,:])
plt.plot(x_PR[0,:],x_PR[1,:])
plt.plot(x_AQ[0,:],x_AQ[1,:])
plt.plot(x_CP[0,:],x_CP[1,:])


#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,P,Q,R)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','P','Q','R']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/Download/chinna/matrix/l1.pdf')
subprocess.run(shlex.split("termux-open '/sdcard/Download/chinna/matrix/l1.pdf'")) 
#else
#plt.show()
