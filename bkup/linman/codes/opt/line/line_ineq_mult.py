#Code by GVV Sharma
#December 25, 2019
#Released under GNU GPL
#Solving linear inequalities

import numpy as np
import matplotlib.pyplot as plt
from coeffs import *


#if using termux
import subprocess
import shlex
#end if


#Line Parameters
n1 = np.array([2,1])
n2 = np.array([1,1])
n3 = np.array([2,-3])

c1 = 4
c2 = 3
c3 = 6

#Intersection of the lines
A=line_intersect(n1,c1,n2,c2)
B=line_intersect(n2,c2,n3,c3)
C=line_intersect(n3,c3,n1,c1)

points = np.array((A,B,C))

#Filling up the desired region
plt.fill(points[:,0], points[:,1], 'k', alpha=0.3)

#Plotting points of Intersection
plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 ) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 ), B[1] * (1)+0.1 , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 - 0.05), C[1] * (1 - 0.1) , 'C')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor

#if using termux
plt.savefig('./line/figs/line_ineq_mult.pdf')
plt.savefig('./line/figs/line_ineq_mult.eps')
subprocess.run(shlex.split("termux-open ./line/figs/line_ineq_mult.pdf"))
#else
#plt.show()







