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


#Triangle sides
a = 10
c = 6
b = np.sqrt(a**2-c**2)

p = (a**2 + c**2-b**2 )/(2*a)
q = np.sqrt(c**2-p**2)

#Triangle vertices
A = np.array([p,q]) 
B = np.array([0,0]) 
C = np.array([a,0]) 
D = np.array([p,-q]) 


#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_CD = line_gen(C,D)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')
plt.plot(D[0], D[1], 'o')
plt.text(D[0] * (1 - 0.2), D[1] * (1) , 'D')

#Plotting the circle

theta = np.linspace(0,2*np.pi,50)
x = c*np.cos(theta)
y = c*np.sin(theta)

plt.plot(x,y)

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
#if using termux
plt.savefig('circle.pdf')
plt.savefig('circle.eps')
subprocess.run(shlex.split("termux-open circle.pdf"))
#else
#plt.show()







