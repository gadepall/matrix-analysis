#Code by GVV Sharma
#November 12, 2019
#released under GNU GPL
import numpy as np
import matplotlib.pyplot as plt
from coeffs import *

#if using termux
import subprocess
import shlex
#end if



#Line points
n = np.array([1,-3]) 
c = -4
A = np.array([0,4/3]) 
P = np.array([1,2]) 
m = omat@n
k1 = -1.5
k2=0
R = ((np.outer(m,m)-np.outer(n,n))@P+2*c*n)/(np.linalg.norm(n)**2)
x_L = line_dir_pt(m,A,k1,k2)
k1 = -1
k2=1
x_M = line_dir_pt(n,P,k1,k2)
#Hero's formula
print(R)

plt.plot(x_L[0,:],x_L[1,:],label='$L$')
plt.plot(x_M[0,:],x_M[1,:],label='$M$')

plt.plot(P[0], P[1], 'o')
plt.text(P[0] * (1 + 0.1), P[1] * (1 - 0.1) , 'P')
plt.plot(R[0], R[1], 'o')
plt.text(R[0] * (1 + 0.1), R[1] * (1 - 0.1) , 'R')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
#if using termux
plt.savefig('./line/figs/line_reflect.pdf')
plt.savefig('./line/figs/line_reflect.eps')
subprocess.run(shlex.split("termux-open ./line/figs/line_reflect.pdf"))
#else
#plt.show()
