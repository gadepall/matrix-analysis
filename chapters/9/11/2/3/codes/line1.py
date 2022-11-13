import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/ganga/matrix/CoordGeo') 

#local imports
from line.funcs import *
from triangle.funcs import *
#from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
#PQR
lambda_1=2
beta=6
theta=np.pi/3
A=np.array([[1,-1],[lambda_1,lambda_1+2*beta*np.cos(theta)]])
B=np.array(([2,18]))
#X=np.linalg.inv(A)@ B

e1 = np.array(([1,0]))
n1 = A[0,:]
n2 = A[1,:]
c1 = B[0]
c2 = B[1]


#Solution vector
x = LA.solve(A,B)

r=int(x[0])

P = r*np.array(([np.cos(theta),np.sin(theta)]))
Q = np.array(([0,0]))
R = np.array(([6,0]))
#Distance between P&R and P&Q
d1 = np.linalg.norm(P-R)
d2 = np.linalg.norm(P-Q)
d3 = d2-d1
#print(x[0])
#print(x[1])
print(x[0]-x[1])

##Generating all lines
x_PQ = line_gen(P,Q)
x_QR = line_gen(Q,R)
x_RP = line_gen(R,P)


#Plotting all lines
plt.plot(x_PQ[0,:],x_PQ[1,:],label='$r$')
plt.plot(x_QR[0,:],x_QR[1,:],label='$p$')
plt.plot(x_RP[0,:],x_RP[1,:],label='$q$')

#Labeling the coordinates
tri_coords = np.vstack((P,Q,R)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','Q','R']

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

#if using termux
plt.savefig('/home/ganga/matrix/figs/line1.pdf')
#subprocess.run(shlex.split("termux-open/sdcard/dowload/matrix/figs/line1.pdf"))
#else
plt.show()





