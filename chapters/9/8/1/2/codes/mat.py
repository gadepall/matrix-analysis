#Code by GVV Sharma (works on termux)
#February 16, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To solve a system of linear equations 


#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/user/Documents/hha/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
#from conics.funcs import circ_gen

#if using termux
#iimport subprocess
#import shlex
#end if

#Input parameters
#Input parameters
theta1 = np.pi/3
theta2 = 2*np.pi/3
O = np.array(([0,0]))
r =2
A = r*np.array(([np.cos(theta1),-np.sin(theta1)]))
C =r*np.array(([np.cos(theta2),np.sin(theta2)]))
#C = r*np.array(([np.cos(theta1),np.sin(theta1)]))
D = r*np.array(([-np.cos(theta2),np.sin(theta2)]))
B=A+C-D
v1 = A-B
v2 = B-C
theta = np.arccos((v1).T@ (v2))/ (LA.norm(v1)*LA.norm(v2))
print (theta) 
#Generating all line

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_AD = line_gen(A,D)
x_OA = line_gen(O,A)
x_OB = line_gen(O,B)
x_OD = line_gen(O,D)
x_OC = line_gen(O,C)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_OC[0,:],x_OC[1,:],label='$OC$')
plt.plot(x_OB[0,:],x_OB[1,:],label='$OB$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.plot(x_OA[0,:],x_OA[1,:],label='$OA$')
plt.plot(x_OD[0,:],x_OD[1,:],label='$OD$')

#Labeling the coordinates
plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')
plt.plot(D[0], D[1], 'o')
plt.text(D[0] * (1 + 0.03), D[1] * (1 - 0.1) , 'D')
plt.plot(O[0], O[1], 'o')
plt.text(O[0] * (1 + 0.1), O[1] * (1 - 0.1) , 'O')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/home/user/Documents/Assignments/assg_4/fig.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/ramesh/maths/fig.pdf"))
#else
plt.show()
