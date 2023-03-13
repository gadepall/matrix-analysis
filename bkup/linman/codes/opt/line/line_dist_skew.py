#Code by GVV Sharma
#November 18, 2019
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



#Plane points
A1 = np.array([1,1,0])
A2 = np.array([2,1,-1])
m1 =np.array([2,-1,1])
m2 =np.array([3,-5,2])
k1 = -4
k2=4
x2_dist_skew = line_dir_pt(m2,A2,k1,k2)
x1_dist_skew = line_dir_pt(m1,A1,k1,k2)

#Plotting all lines
plt.plot(x1_dist_skew[0,:],x1_dist_skew[1,:], x1_dist_skew[2,:],label='$L_1$')
plt.plot(x2_dist_skew[0,:],x2_dist_skew[1,:], x2_dist_skew[2,:], label='$L_2$')
#plotting points
ax.scatter(A1[0],A1[1],A1[2],'o')
ax.scatter(A2[0],A2[1],A2[2],'o')
ax.text(1,1,0, "A1", color='red')
ax.text(2,1,-1, "A2", color='green')


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
#plt.axis('equal')
#if using termux
plt.savefig('./line/figs/line_dist_skew.pdf')
plt.savefig('./line/figs/line_dist_skew.eps')
subprocess.run(shlex.split("termux-open ./line/figs/line_dist_skew.pdf"))
#else
#plt.show()








