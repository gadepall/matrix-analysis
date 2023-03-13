#Program to plot  a hyperbola
#Code by GVV Sharma
#July 6, 2020
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#if using termux
import subprocess
import shlex
#end if

#setting up plot
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
len = 100
y = np.linspace(-5,5,len)

#Hyperbola parameters
V = 1/2*np.array(([0,1],[1,0]))
u = -3/2*np.array(([0,1]))
f = 2
#Reflection matrix
R = np.array(([0,1],[1,0]))
#Eigenvalues and eigenvectors
D_vec,P = LA.eig(V)
D = np.diag(D_vec)
#Hyperbola parameters at the origin
D1 = -R@D@R
D1_vec=np.diag(D1)
#Generating the positive hyperbola at the origin
x = np.sqrt((1-D1_vec[1]*(y**2))/D1_vec[0])

#Affine transformation parameters
c = -LA.inv(V)@u
k = np.sqrt(np.abs(f +u@c ))

#Affine transform 
z = np.hstack((np.vstack((x,y)),np.vstack((-x,y))))
x_vec = k*P.T@R@z+c[:,None]

#Tangent parameters
m = np.array(([1,2]))
Q = np.array(([0,-1],[1,0]))
m0 = LA.inv(P.T@R)@m
q_t =Q@D1@m0
print(m0*np.sqrt(2),Q@D1@m0*np.sqrt(2),q_t.T@D1@q_t)

#Plotting
plt.plot(z[0,0:len],z[1,0:len], color='b')
plt.plot(z[0,len:2*len],z[1,len:2*len], color='b', label = '$\mathbf{y}^T\mathbf{D}\mathbf{y}^T=1$')
plt.plot(x_vec[0,0:len],x_vec[1,0:len], color='r')
plt.plot(x_vec[0,len:2*len],x_vec[1,len:2*len], color='r', label = '$\mathbf{x}^T\mathbf{V}\mathbf{x}+2\mathbf{u}^T\mathbf{x}+f = 0$')
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best')
plt.title('Hyperbola')
plt.grid()

#if using termux
plt.savefig('./figs/hyperbola.pdf')
plt.savefig('./figs/hyperbola.eps')
subprocess.run(shlex.split("termux-open ./figs/hyperbola.pdf"))
#else
#plt.show()
#print(x[0:4], x[1],x[2],x[3])
