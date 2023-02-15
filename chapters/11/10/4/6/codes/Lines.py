import matplotlib.pyplot as plt
import numpy as np
import math as ma
from matplotlib import pyplot as plt, patches
import math
import subprocess
import shlex
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
n1 = np.array([1, -7])
n2 = np.array([3, 1])
A1 = np.array([-5, 0])
A2 = np.array([0, 0])
P= line_intersect(n1,A1,n2,A2)                                                          
Q=P
m = np.array([0, 1])
z=np.array(([0,1],[-1,0]))                           
m1=z@n1                                     
print(m1)
m2=z@n2
k1=-5
k2=5
x_AB =line_dir_pt(m1,A1,k1,k2)
x_CD = line_dir_pt(m2,A2,k1,k2)
x_EF= line_dir_pt(m,Q,k1,k2)
plt.plot(x_AB[0,:],x_AB[1,:],label='Line1')
plt.plot(x_CD[0,:],x_CD[1,:],label='Line2')
plt.plot(x_EF[0,:],x_EF[1,:],label='Parallel Line')
##Generating the line 

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('/sdcard/matrix/code/fig.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/matrix/code/fig.pdf"))
#else
plt.savefig('/sdcard/download/fwcassgn/trunk/fwcassgn/trunk/naveed/line.png')
plt.show()
