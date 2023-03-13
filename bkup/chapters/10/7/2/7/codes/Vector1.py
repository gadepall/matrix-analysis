#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import numpy as np
#from params import *
def dir_vec(A,B):
  return B-A

def norm_vec(A,B):
  return np.matmul(omat, dir_vec(A,B))

#Generate line points
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


#Intersection of two lines
def line_intersect(n1,A1,n2,A2):
  N=np.vstack((n1,n2))
  print(type(N))
  p = np.zeros(2)
  p[0] = n1@A1
  p[1] = n2@A2
  #Intersection
  P=np.linalg.inv(N)@p
  return P

#Foot of the Perpendicular
def perp_foot(n,cn,P):
  m = omat@n
  N=np.block([[n],[m]])
  p = np.zeros(2)
  p[0] = cn
  p[1] = m@P
  #Intersection
  x_0=np.linalg.inv(N)@p
  return x_0

#Reflection
def reflect(n,c,P):

  D = P+2*(c-n@P)/(LA.norm(n)**2)*n
  return 
#Generating points on a circle
def circ_gen(O,r):
 len = 50
 theta = np.linspace(0,2*np.pi,len)
 x_circ = np.zeros((2,len))
 x_circ[0,:] = r*np.cos(theta)
 x_circ[1,:] = r*np.sin(theta)
 x_circ = (x_circ.T + O).T
 return x_circ


#Input parameters
r1 = np.sqrt(50)       #radius   #distance of points from center
C = np.array(([2,-3]))
B = np.array(([1,4]))

A=2*C-B
print(A)

###Generating the circle
x_circ3= circ_gen(C,r1)


#
x_AC = line_gen(A,C)
plt.plot(x_AC[0,:],x_AC[1,:])
x_CB = line_gen(C,B)
plt.plot(x_CB[0,:],x_CB[1,:])


#
#Plotting the circle
plt.plot(x_circ3[0,:],x_circ3[1,:],label='$Circle$')
#Labeling the coordinates
tri_coords = np.vstack((A,C,B)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A(3,-10)','C(2,-3)','B(1,4)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(3,-10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('/sdcard/Download/matrices/Circle/circle1.pdf')
#subprocess.run(shlex.split("termux-open '/sdcard/Download/matrices/Circle/circle1.pdf'"))
#else
plt.show()
