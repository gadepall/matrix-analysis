
#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import math as ma
import matplotlib.pyplot as plt
from numpy import linalg as LA


#if using termux
import subprocess
import shlex
#end if

def line_dir_pt(m,P,k1,k2):
  len = 10
  dim = P.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = P + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB       


#Input parameters
P=  np.array(([-4,3]))


#Direction vector
m=np.array(([1,1/2]))                                                              
z=np.array(([0,1],[-1,0]))                           
n=z@m                                     

##Generating the line 
k1=-8
k2=3
x_mP = line_dir_pt(m,P,k1,k2)


#Plotting all lines
plt.plot(x_mP[0,:],x_mP[1,:],label='Line equation')

#Labeling the coordinates
tri_coords = np.vstack((P,)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()               

