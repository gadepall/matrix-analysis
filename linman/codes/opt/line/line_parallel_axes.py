#Code by GVV Sharma
#November 14, 2019
#released under GNU GPL
import numpy as np
import matplotlib.pyplot as plt
from coeffs import *


#if using termux
import subprocess
import shlex
#end if



#Inputs
A = np.array([-2,3]) 
m = np.array([1,0]) 
n = np.array([0,1]) 
k1=-4
k2=0
#Generating all lines
x_ax = line_dir_pt(m,A,k1,k2)
x_ay = line_dir_pt(n,A,k1,k2)

#Plotting all lines
plt.plot(x_ax[0,:],x_ax[1,:],label='$\\parallel x$')
plt.plot(x_ay[0,:],x_ay[1,:],label='$\\parallel y$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
#if using termux
plt.savefig('./line/figs/line_parallel_axes.pdf')
plt.savefig('./line/figs/line_parallel_axes.eps')
subprocess.run(shlex.split("termux-open ./line/figs/line_parallel_axes.pdf"))
#else
#plt.show()







