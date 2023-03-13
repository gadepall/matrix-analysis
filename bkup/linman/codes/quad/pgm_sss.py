#Code by GVV Sharma
#December 11, 2019
#released under GNU GPL
#Drawing a parallelogram  given 2 sides and a diagonal
import numpy as np
import matplotlib.pyplot as plt
from coeffs import *


#if using termux
import subprocess
import shlex
#end if


#Triangle sides
a = 6
b = 4.5
c = 7.5


#Coordinates of D
p = (a**2 + c**2-b**2 )/(2*a)
q = np.sqrt(c**2-p**2)
print(p,q, a**2+b**2-c**2)

#Parallelogram vertices
D = np.array([p,q]) 
B = np.array([0,0]) 
C = np.array([a,0]) 

#Mid point of BD
O =(B+D)/2

#Finding A
A = 2*O-C

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_DB = line_gen(D,B)
x_AD = line_gen(A,D)
x_CD = line_gen(C,D)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_DB[0,:],x_DB[1,:],label='$DB$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')
plt.plot(D[0], D[1], 'o')
plt.text(D[0] * (1 + 0.03), D[1] * (1 - 0.1) , 'D')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('./figs/quad/pgm_sss.pdf')
plt.savefig('./figs/quad/pgm_sss.eps')
subprocess.run(shlex.split("termux-open ./figs/quad/pgm_sss.pdf"))
#else
#plt.show()







