#python libaries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys
sys.path.insert(0,'/home/susi/Documents/CoordGeo')
#local imports
from line.funcs import *
#from triangle.funcs import *
#from conics.funcs import circ_gen
import math
import subprocess
import shlex

####plotting part####

def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB
#Given points
P = np.array([0,3])
Q = np.array([4,6])

#Direction vector
m=P-Q
z=np.array(([0,1],[-1,0]))
n=np.matmul(z,m)

#points on line
A=np.array(([1,3.75]))

print("The direction vecor is:",m)
#print the equation
print(n[0],'','x','+',n[1],'','y','=',(n[0]*A[0]+n[1]*A[1]),sep="")
x_PQ = line_gen(P,Q)

#Plotting all lines
plt.plot(x_PQ[0,:],x_PQ[1,:],label='$PQ$')

#Labeling the coordinates
tri_coords = np.vstack((P,Q)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P(0,3)','Q(4,6)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$xaxis$')
plt.ylabel('$yaxis$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')
plt.title('equation of straight line')
plt.savefig('/sdcard/download/iith/python/Assignment-4/figure.pdf')  
subprocess.run(shlex.split("termux-open /sdcard/download/iith/python/Assignment-4/figure.pdf"))
#plt.show()
