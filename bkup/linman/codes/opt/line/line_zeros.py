#Code by GVV Sharma
#Dec 16, 2019
#released under GNU GPL
#Plotting a line

import numpy as np
import matplotlib.pyplot as plt
from coeffs import *


#if using termux
import subprocess
import shlex
#end if



#Inputs
n = np.array([1,1]) 
c = 7
e1 = np.array([1,0]) 
e2 = np.array([0,1]) 
A = c*e1/(n@e1)
B = c*e2/(n@e2)
#Generating all lines
x_AB = line_gen(A,B)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
#if using termux
plt.savefig('./line/figs/line_icept.pdf')
plt.savefig('./line/figs/line_icept.eps')
subprocess.run(shlex.split("termux-open ./line/figs/line_icept.pdf"))
#else
#plt.show()







