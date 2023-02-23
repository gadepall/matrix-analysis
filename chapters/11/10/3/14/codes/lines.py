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
   
   
def perp_foot(n,cn,P):
    omat=np.array(([0,1],[-1,0]))
    m = omat@n
    N=np.block([[n],[m]])
    p = np.zeros(2)
    p[0] = cn
    p[1] = m@P
    x_0=np.linalg.inv(N)@p
    return x_0

def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB



def compute_point_on_line(a,b,c):
    y=c/b
    p=np.array([0,y])
    return p




#enter the coeffcients of line

A=compute_point_on_line(3,-4,16);


n=np.array([3,-4])
omat=np.array(([0,1],[-1,0]));
m=omat@n;


P = np.array([-1, 3])
c = 16
point=perp_foot(n,c,P)
print(point)

x_QR = line_dir_pt(m,A,-3,3);
x_AP = line_gen(point,P)

#Plotting all lines
plt.plot(x_QR[0,:],x_QR[1,:],label='$3x-4y-16=0$') # Given line
plt.plot(x_AP[0,:],x_AP[1,:],label='$Foot of Perpendicular$') # Perpendicular line from P to Foot of perpendicular 
tri_coords = np.vstack((P,point)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','A']
for i, txt in enumerate(vert_labels):
    label = "{}({:.2f},{:.2f})".format(txt, tri_coords[0,i],tri_coords[1,i]) #Form label as A(x,y)
    plt.annotate(label, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(25,3), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
#Labeling the coordinates

plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.title('Line')
plt.show()

