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

len = 100
degrad = 180/np.pi

#Quadrilateral sides
a = 4.5 #BC
b = 5.5 #CD
c = 4  #AD
d = 6 #AB
e = 7 #BD

#theta1 = DBC
t1 = np.arccos((a**2+e**2-b**2)/(2*a*e))
#theta2 = ABD
t2 = np.arccos((d**2+e**2-c**2)/(2*d*e))

print(t1*degrad,t2*degrad)

#Rotation matrix for t1
P = np.array(([np.cos(t1),-np.sin(t1)],[np.sin(t1),np.cos(t1)]) )



#Quadrilateral vertices
D,B,C =  tri_vert(a,b,e)
A1,B1,D1 =  tri_vert(e,c,d)
A = P@A1

#Printing coordinates
print(A1,A,B,C,D)

#Verifying coordinates
print(np.linalg.norm(A-B))
print(np.linalg.norm(B-C))
print(np.linalg.norm(C-D))
print(np.linalg.norm(A-D))
print(np.linalg.norm(B-D))


#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
#x_CA = line_gen(A,C)
x_BD = line_gen(B,D)
x_DA = line_gen(D,A)
x_CD = line_gen(C,D)

#Generating Circle
theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = e*np.cos(theta)
x_circ[1,:] = e*np.sin(theta)
x_circ = (x_circ.T + B).T


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
#plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_BD[0,:],x_BD[1,:],label='$BD$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.plot(x_DA[0,:],x_DA[1,:],label='$DA$')

#Plotting circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$circle$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(A1[0], A1[1], 'o')
plt.text(A1[0] * (1 + 0.1), A1[1] * (1 - 0.1) , '$A_1$')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) , 'C')
plt.plot(D[0], D[1], 'o')
plt.text(D[0] * (1 + 0.03), D[1] * (1 - 0.1) , 'D')
plt.plot(D1[0], D1[1], 'o')
plt.text(D1[0] * (1 + 0.03), D1[1] * (1 - 0.1) , '$D_1$')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('./figs/quad/quad_sss.pdf')
plt.savefig('./figs/quad/quad_sss.eps')
subprocess.run(shlex.split("termux-open ./figs/quad/quad_sss.pdf"))
#else
#plt.show()







