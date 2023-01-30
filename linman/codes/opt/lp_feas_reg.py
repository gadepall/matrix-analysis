#Code by GVV Sharma
#Jan 1, 2019
#released under GNU GPL
#Get the feasible region for a linear program

import numpy as np
import matplotlib.pyplot as plt
from coeffs import *

#if using termux
import subprocess
import shlex
#end if

#Constraints
p = 2
A = np.array(([1,1],[3,1]))
I = np.identity(p) 

A_I = np.vstack((A,I))
b = np.array([50,90])
z = np.zeros(2)
b_z = np.hstack((b,z))
k = np.array(([-100,1],[-100,1],[-100,1],[-2,60]) )
m,n = A.shape
z=mult_line(A_I,b_z,k,m+p)

#Plots
for i in range(m+p):
	plt.plot(z[i,0,:],z[i,1,:],label= i)

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('./figs/lp_feas_reg.pdf')
plt.savefig('./figs/lp_feas_reg.eps')
subprocess.run(shlex.split("termux-open ./figs/lp_feas_reg.pdf"))
#else
#plt.show()

