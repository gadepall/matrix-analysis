import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys
lib_path = '/sdcard/IITH/matrix/CoordGeo'
sys.path.insert(0,lib_path)

#local import
from line.funcs import *
from triangle.funcs import *

#if using termux
import subprocess
import shlex
#end if

# Coordinates of A and B
#A = np.array([0, 0])
#B = np.array([36, 15])
# Calculate distance
#d = np.linalg.norm(A - B)
#print(d)

#1 Point A,B and C
I = np.array([7,-2]) 
J = np.array([5,1])   
K = np.array([3,4])
#2 Point A,B and C
L = np.array([8,1]) 
M = np.array([3,-4])   
N = np.array([2,-5])

#1 Generating all lines
x_IJ = line_gen(I,J)
x_JK = line_gen(J,K)
x_KI = line_gen(K,I)
#2 Generating all lines
x_LM = line_gen(L,M)
x_MN = line_gen(M,N)
x_NL = line_gen(N,L)


#1 Plotting all lines
plt.plot(x_IJ[0,:],x_IJ[1,:],color='blue')
plt.plot(x_JK[0,:],x_JK[1,:],color='blue')
#2 Plotting all lines
plt.plot(x_LM[0,:],x_LM[1,:],color='green')
plt.plot(x_MN[0,:],x_MN[1,:],color='green')

plt.plot(I[0], I[1], 'o',color='blue')
plt.text(I[0] * (1 + 0.1), I[1] * (1 - 0.1) , 'A(7,-2)')
plt.plot(J[0], J[1], 'o',color='blue')
plt.text(J[0] * (1 - 0.1), J[1] * (1) , 'B(5,1)')
plt.plot(K[0], K[1], 'o',color='blue')
plt.text(K[0] * (1 - 0.1), K[1] * (1) , 'C(3,4)')

plt.plot(L[0], L[1], 'o',color='g')
plt.text(L[0] * (1 + 0.1), L[1] * (1 - 0.1) , 'A(8,1)')
plt.plot(M[0], M[1], 'o',color='g')
plt.text(M[0] * (1 - 0.1), M[1] * (1) , 'B(3,-4)')
plt.plot(N[0], N[1], 'o',color='g')
plt.text(N[0] * (1 - 0.1), N[1] * (1) , 'C2,-5')


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux                        
plt.savefig('/sdcard/IITH/vector/vector-3/figs/vec.pdf')                                           
subprocess.run(shlex.split("termux-open /sdcard/IITH/vector/vector-3/figs/vec.pdf")) 
#plt.show()
