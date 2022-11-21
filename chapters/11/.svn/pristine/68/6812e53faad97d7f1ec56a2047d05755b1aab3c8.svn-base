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
#Given points
A = np.array(([7,6]))
B = np.array(([3,4]))
e_2 = np.array(([0,1])) #standard basis vector

#Diection vector
n = A-B

#Computations
c = (np.linalg.norm(A)**2- np.linalg.norm(B)**2)/2

x = c/(n@e_2)

#Output
P = x*e_2
#print(x,P)
A1=np.array(([0,1],[4,2]))
A2=np.array([0,30])
X=np.linalg.solve(A1,A2)
print(X)
#Generating all lines
x_AB = line_gen(A,B)
x_PA = line_gen(X,A);
x_PB = line_gen(X,B)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_PA[0,:],x_PA[1,:],label='$PA$')
plt.plot(x_PB[0,:],x_PB[1,:],label='$PB$')


#labeling the coordinates
tri_coords = np.vstack((A,B,X)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','P']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt,(tri_coords[0,i], tri_coords[1,i]),textcoords="offset points",xytext=(0,10),ha='center') 

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')

#if using termux
#plt.savefig('/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-2.pdf')
#subprocess.run(shlex.split("termux-open '/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-2.pdf'")) 
#else
plt.show()
