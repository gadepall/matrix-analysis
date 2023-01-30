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

def tri_coord(a,b,c):
	p = (a**2 + c**2-b**2 )/(2*a)
	q = np.sqrt(c**2-p**2)
	A = np.array([p,q]) 
	return A


#Triangle sides
a = 4.5
b = 5.5
c = 4
d = 6
e = 7
ct = (a**2+e**2-b**2)/(2*a*e)
st = np.sqrt(1-ct**2)
P = np.array(([ct,-st],[st,ct]) )
#Quadrilateral vertices
C = tri_coord(a,b,e)
A = np.array([0,0]) 
B = np.array([a,0]) 
D = tri_coord(e,c,d)
D = P@D

print(np.linalg.norm(A-B))
print(np.linalg.norm(B-C))
print(np.linalg.norm(C-D))
print(np.linalg.norm(A-D))
print(np.linalg.norm(A-C))


#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(A,C)
x_DA = line_gen(D,A)
x_CD = line_gen(C,D)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.plot(x_DA[0,:],x_DA[1,:],label='$DA$')

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

#if using termux
plt.savefig('../figs/quad.pdf')
plt.savefig('../figs/quad.eps')
subprocess.run(shlex.split("termux-open ../figs/quad.pdf"))
#else
#plt.show()







