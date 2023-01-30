from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from coeffs import *
import numpy as np

#if using termux
import subprocess
import shlex
#end if

#creating x,y for 3D plotting
xx, yy = np.meshgrid([-10,10], range(10))
#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
#ax = fig.add_subplot(111,projection='3d',aspect='equal')

#defining lines : x(k) = A + k*l

#defining planes:  n.T * x = c 
n1 = np.array([2,3,-1]).reshape((3,1))
A =  np.array([5,2,-4]).reshape((3,1))
c1 = 20

k1=-4
k2=4
#generating points in line 

x_OA = line_dir_pt(n1,A,k1,k2)


#plotting line
plt.plot(x_OA[0,:],x_OA[1,:],x_OA[2,:],label="Normal")

#corresponding z for planes
z1 = (c1-n1[0]*xx-n1[1]*yy)/(n1[2])

#plotting planes
Plane=ax.plot_surface(xx, yy, z1,label="Plane", color='r',alpha=0.5)
Plane._facecolors2d=Plane._facecolors3d
Plane._edgecolors2d=Plane._edgecolors3d
#plotting point
ax.scatter(A[0],A[1],A[2],'o')

#show plot
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best');plt.grid()
#if using termux
plt.savefig('./line/figs/plane_3d.pdf')
plt.savefig('./line/figs/plane_3d.eps')
subprocess.run(shlex.split("termux-open ./line/figs/plane_3d.pdf"))
#else
#plt.show()

	

