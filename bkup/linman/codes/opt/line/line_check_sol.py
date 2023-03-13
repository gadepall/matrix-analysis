#Code by GVV Sharma
#Dec 16, 2019
#released under GNU GPL
#Check whether a system of linear equations has a solution

import numpy as np
import matplotlib.pyplot as plt
from coeffs import *

#if using termux
import subprocess
import shlex
#end if



#Lines
n1 = np.array([1,2]) 
n2 = np.array([2,4]) 
c =  np.array([4,12]) 

#Intercepts
A1,B1 =  line_icepts(n1,c[0])
A2,B2 =  line_icepts(n2,c[1])


#Matrix Ranks
N=np.vstack((n1,n2))
M =np.vstack((N.T, c)).T
#M =np.vstack((n1,n2, c))
rank_N = np.linalg.matrix_rank(N)
rank_M = np.linalg.matrix_rank(M)
m,n = np.shape(N)
print(rank_M, rank_N, M)

#Checking for solution
if rank_N==rank_M:
	if rank_N == m:
		print("Unique Solution Exists:",np.linalg.inv(N)@c)
	else:
		print("Infinite Number of Solutions")
else:
	print("No solution")

##Generating all lines
k1 = 0
k2 = 1

x_A1B1 =  line_dir_pt(n1,A1,k1,k2)
#k1 = -4
#k2 = 4
x_A2B2 =   line_dir_pt(n2,A2,k1,k2)
#
#Plotting all lines
plt.plot(x_A1B1[0,:],x_A1B1[1,:],label='line 1')
plt.plot(x_A2B2[0,:],x_A2B2[1,:],label='line 2')
#
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('./line/figs/line_check_sol.pdf')
plt.savefig('./line/figs/line_check_sol.eps')
subprocess.run(shlex.split("termux-open ./line/figs/line_check_sol.pdf"))
#else
#plt.show()

