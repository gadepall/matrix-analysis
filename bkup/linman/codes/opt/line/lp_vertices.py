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

for i in range(m+p):
	plt.plot(z[i,0,:],z[i,1,:],label= i)
#plt.plot(x_2[0,:],x_2[1,:],label='line 2')
#plt.plot(x_3[0,:],x_3[1,:],label='line 3')
#plt.plot(x_4[0,:],x_4[1,:],label='line 4')
#
##Lines
##n1 = np.array([1,2]) 
##n2 = np.array([2,4]) 
##c =  np.array([4,12]) 
##
###Intercepts
##A1,B1 =  line_icepts(n1,c[0])
##A2,B2 =  line_icepts(n2,c[1])
##
##
###Matrix Ranks
##N=np.vstack((n1,n2))
##M =np.vstack((N.T, c)).T
###M =np.vstack((n1,n2, c))
##rank_N = np.linalg.matrix_rank(N)
##rank_M = np.linalg.matrix_rank(M)
##m,n = np.shape(N)
##print(rank_M, rank_N, M)
##
###Checking for solution
##if rank_N==rank_M:
##	if rank_N == m:
##		print("Unique Solution Exists:",np.linalg.inv(N)@c)
##	else:
##		print("Infinite Number of Solutions")
##else:
##	print("No solution")
##
####Generating all lines
##k1 = 0
##k2 = 1
##
##x_A1B1 =  line_dir_pt(n1,A1,k1,k2)
###k1 = -4
###k2 = 4
##x_A2B2 =   line_dir_pt(n2,A2,k1,k2)
###
###Plotting all lines
##plt.plot(x_A1B1[0,:],x_A1B1[1,:],label='line 1')
##plt.plot(x_A2B2[0,:],x_A2B2[1,:],label='line 2')
##
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('./line/figs/lp_vertices.pdf')
plt.savefig('./line/figs/lp_vertices.eps')
subprocess.run(shlex.split("termux-open ./line/figs/lp_vertices.pdf"))
#else
#plt.show()

