import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#Orthogonal matrix
omat = np.array([[0,1],[-1,0]]) 
 
def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB
   
def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB 
  
def line_intersect(n1,A1,n2,A2):
  N=np.vstack((n1,n2))
  print(type(N))
  p = np.zeros(2)
  p[0] = n1@A1
  p[1] = n2@A2
  #Intersection
  P=np.linalg.inv(N)@p
  return P  

def dir_vec(A,B):
   return B-A
 

def norm_vec(A,B):
   return np.matmul(omat, dir_vec(A,B))

#if using termux
import subprocess
import shlex
#end if


#input parameters
R=np.array([2,5])   #along the line
p=np.array([-3,5])   #perpendicular to the line
Q=np.array([-3,6])
m=Q-R
n=omat@m
a1=m.T@p
a2=n.T@Q
c=np.array((m,n))
d=np.array(([a1,a2]))
e1=np.array(([1,0]))
n1=c[0,:]
n2=c[1,:]
c1=d[0]
c2=d[1]
#solution vector
x=LA.solve(c,d)

#direction vectors
m1=omat@n1
m2=omat@n2

#points on the lines
x1=c1/n1@e1
A1=x1*e1
x2=c2/n2@e1
A2=x1*e1
print(x,x1,x2)
##Generating all lines
k1=0
k2=1
x_AB = line_dir_pt(m1,A1,k1,k2)
x_CD = line_dir_pt(m2,A2,k1,k2)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_CD[0,:],x_CD[1,:])
#Labeling the coordinates
tri_coords = x.T
plt.scatter(tri_coords[0], tri_coords[1])
vert_labels = ['x']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0], tri_coords[1]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-2.pdf')
#subprocess.run(shlex.split("termux-open '/storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-2.pdf'")) 
#else
plt.show()`
