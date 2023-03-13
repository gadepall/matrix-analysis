import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import subprocess
import shlex
import math
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


#input parameters
A = np.array(([-3,0]))
O = np.array(([0,0])) 
B=np.array(([0,-6]))
omat = np.array(([0,-1],[1,0]))
slope = -2
m = np.array(([1,slope]))
#directional vector
n = omat@m
c = n@A
print(n,c)
print(n[0],'*','x','+',n[1],'*','y','=',(n[0]*A[0]+n[1]*A[1]),sep="")

x_AB = line_gen(A,B)



#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')

#Labeling the coordinates
tri_coords = np.vstack((A,B,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','O']
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
plt.title('A line passing through a point')
#if using termux
#plt.savefig('/home/user/vamsi/line_a2/par.pdf')
#subprocess.run(shlex.split("termux-open '/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-2.pdf'")) 
#else
plt.show()
