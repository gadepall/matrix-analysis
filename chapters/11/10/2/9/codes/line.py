
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                    

#local imports
#from line.funcs import *
#from triangle.funcs import *
#from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
def line_gen(A,B):
    len =8
    dim = A.shape[0]
    x_AB = np.zeros((dim,len))
    lam_1 = np.linspace(0,1,len)
    for i in range(len):
      temp1 = A + lam_1[i]*(B-A)
      x_AB[:,i]= temp1.T
    return x_AB

P = np.array(([2,1]))
R = np.array(([ 4,5]))
Q = np.array(([-2,3]))
#A = np.array(([0,2]))


m = n = 1
A = ((m*P) + (n*Q))/(1+m)
print(A)

x_PR = line_gen(P,R)
x_RQ = line_gen(R,Q)
x_QP = line_gen(Q,P) 
x_RA = line_gen(R,A)
#Plotting all lines
plt.plot(x_PR[0,:],x_PR[1,:],color='blue')
plt.plot(x_RQ[0,:],x_RQ[1,:],color='red')
plt.plot(x_QP[0,:],x_QP[1,:],color='purple')
plt.plot(x_RA[0,:],x_RA[1,:],label='$AR$',linestyle="--")
#plt.plot(x_BQ[0,:],x_BQ[1,:],label='$BQ$',linestyle="

#Labeling the coordinates
tri_coords = np.block([[P],[Q],[R],[A]]).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','Q','R','A(0,2)']
for i, txt in enumerate(vert_labels):
     plt.annotate(txt, # this is the text
                  (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                  textcoords="offset points", # how to position the text
                  xytext=(0,5), # distance from text to points (x,y)
                  ha='center') # horizontal alignment can be left, right or center
 

plt.xlabel('$ X $')
plt.ylabel('$ Y $')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.title('Triangle PQR with median A')

#if using termux
plt.savefig('/home/adarsh/lines/fig.pdf')
subprocess.run(shlex.split("termux-open/sdcard/adarsh/lines/fig.pdf"))
#plt.show()
