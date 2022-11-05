#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/fwc/matrices/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
import subprocess
import shlex
#Input parameters
k = 1
c = 8
a = 12
p2 =4
theta = np.pi/3
A = c*np.array(([np.cos(theta),np.sin(theta)]))
R = np.array(([0,0]))
e1 = np.array(([1,0]))
e2 = np.array(([0,1]))
B =(p2*e1)+A
P = (a*e1)
D = (k*A+R)/(k+1)
C = (k*P+B)/(k+1)

##Generating all lines
x_AR = line_gen(A,R)
x_RP = line_gen(R,P)
x_PB = line_gen(P,B)
x_BA = line_gen(B,A)
x_DB = line_gen(D,B)
x_AC = line_gen(A,C)
x_DP = line_gen(D,P)
x_RC = line_gen(R,C)
x_CD = line_gen(C,D)


#Plotting all lines
plt.plot(x_AR[0,:],x_AR[1,:])#,label='$Diameter$')
plt.plot(x_RP[0,:],x_RP[1,:])#,label='$Diameter$')
plt.plot(x_PB[0,:],x_PB[1,:])#,label='$Diameter$')
plt.plot(x_BA[0,:],x_BA[1,:])#,label='$Diameter$')
plt.plot(x_DB[0,:],x_DB[1,:])#,label='$Diameter$')
plt.plot(x_AC[0,:],x_AC[1,:])#,label='$Diameter$')
plt.plot(x_DP[0,:],x_DP[1,:])#,label='$Diameter$')
plt.plot(x_RC[0,:],x_RC[1,:])#,label='$Diameter$')
plt.plot(x_CD[0,:],x_CD[1,:])#,label='$Diameter$')


#Labeling the coordinates
tri_coords = np.vstack((A,R,P,B,D,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','R','P','B','D','C']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.savefig('/sdcard/fwc/matrices/CoordGeo/matrix.pdf')
#plt.show()
