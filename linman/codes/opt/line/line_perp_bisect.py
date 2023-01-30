#Code by GVV Sharma
#Nov 16, 2019
#released under GNU GPL
import numpy as np
import matplotlib.pyplot as plt
from coeffs import *


#if using termux
import subprocess
import shlex
#end if



#Inputs
A = np.array([7,1]) 
B = np.array([3,5]) 
P = np.array([2,0]) 
n = np.array([1,-1]) 
m = omat@n
k1=0
k2=10
#Generating all lines
x_AB = line_gen(A,B)
x_perp_bisect = line_dir_pt(m,P,k1,k2)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_perp_bisect[0,:],x_perp_bisect[1,:],label='$Perpendicular Bisector$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(P[0], P[1], 'o')
plt.text(P[0] * (1 - 0.2), P[1] * (1) , 'P')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
#if using termux
plt.savefig('./line/figs/line_perp_bisect.pdf')
plt.savefig('./line/figs/line_perp_bisect.eps')
subprocess.run(shlex.split("termux-open ./line/figs/line_perp_bisect.pdf"))
#else
#plt.show()







