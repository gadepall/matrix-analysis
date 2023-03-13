#Code by GVV Sharma
#November 12, 2019
#released under GNU GPL
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from coeffs import *

#if using termux
import subprocess
import shlex
#end if

#creating x,y for 3D plotting
xx, yy = np.meshgrid(range(10), range(10))
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
#ax = fig.add_subplot(111,projection='3d',aspect='equal')

#Vertices of 3D triangle
A = np.array([1,2,3]).reshape((3,1))
B = np.array([-1,-2,-1]).reshape((3,1))
C = np.array([2,3,2]).reshape((3,1))
D = np.array([4,7,6]).reshape((3,1))

#Testing orthogonality
#print((A-C).T@(B-C))
#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_AD = line_gen(A,D)


#plotting line
plt.plot(x_AB[0,:],x_AB[1,:],x_AB[2,:],label="AB")
plt.plot(x_BC[0,:],x_BC[1,:],x_BC[2,:],label="BC")
plt.plot(x_CD[0,:],x_CD[1,:],x_CD[2,:],label="CD")
plt.plot(x_AD[0,:],x_AD[1,:],x_AD[2,:],label="AD")

#plotting point
ax.scatter(A[0],A[1],A[2],'o')
ax.scatter(B[0],B[1],B[2],'o')
ax.scatter(C[0],C[1],C[2],'o')
ax.scatter(D[0],D[1],D[2],'o')
ax.text(1,2,3, "A", color='red')
ax.text(-1,-2,-1, "B", color='red')
ax.text(2,3,2, "C", color='red')
ax.text(4,7,6, "D", color='red')

#show plot
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
#if using termux
plt.savefig('./triangle/figs/quad_3d.pdf')
plt.savefig('./triangle/figs/quad_3d.eps')
subprocess.run(shlex.split("termux-open ./triangle/figs/quad_3d.pdf"))
#else
#plt.show()

	





