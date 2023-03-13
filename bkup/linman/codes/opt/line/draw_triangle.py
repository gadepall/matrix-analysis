#Code by GVV Sharma
#March 26, 2019
#released under GNU GPL
import numpy as np
import matplotlib.pyplot as plt
from coeffs import *


#if using termux
import subprocess
import shlex
#end if


n1 = np.array([2,1])
n2 = np.array([1,1])
n3 = np.array([2,-3])

c1 = 4
c2 = 3
c3 = 6


A=line_intersect(n1,c1,n2,c2)
B=line_intersect(n2,c2,n3,c3)
C=line_intersect(n3,c3,n1,c1)
#
#A = np.array([1,2])
#B = np.array([3,0])
#C = np.array([2.25,-0.5])

#Triangle sides
a = 4
b = 5
c = 6
p = (a**2 + c**2-b**2 )/(2*a)
q = np.sqrt(c**2-p**2)

#Triangle vertices
#A = np.array([1,2]) 
#B = np.array([3,0]) 
#C = np.array([2.25,-0.5]) 
#
#A1 = np.array([1,2])
#A2 = np.array([3,0])
#A3 = np.array([2.25,-0.5])
#A = np.array([p,q]) 
#B = np.array([0,0]) 
#C = np.array([a,0]) 
#

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor

#if using termux
plt.savefig('./line/figs/line_2d.pdf')
plt.savefig('./line/figs/line_2d.eps')
subprocess.run(shlex.split("termux-open ./line/figs/line_2d.pdf"))
#else
#plt.show()







