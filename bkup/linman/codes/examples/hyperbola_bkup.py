#Program to plot  a hyperbola
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
theta = np.linspace(-5,5,len)

#Given hyperbola parameters
#Eqn : x.T@V@x = F
V = np.array(([4,0],[0,-1]))
F = 36

#Standard Eqn : y.T@D@y=1
#comparing these equations, get :  
#y = P.T@x/sqrt(F)
#P.T@V@P = D
#P.T@P = I

eigval,eigvec = LA.eig(V)
print(eigval)
print(eigvec)

D = np.diag(eigval)
P = eigvec
print("D=\n",D)
print("P=\n",P)


#Generating points on the hyperbola at origin
#y = np.zeros((2,len))
#y[0,:] = 1/eigval[0]*np.cosh(theta)
#y[1,:] = 1/eigval[1]*np.sinh(theta)

#a=4
#b =-1 
#Standard hyperbola : y.T@D@y=1
y1 = np.linspace(-1,1,len)
#y2 = np.sqrt((1-D[0,0]*np.power(y1,2))/(D[1,1]))
#y3 = -1*np.sqrt((1-D[0,0]*np.power(y1,2))/(D[1,1]))
y2 = np.sqrt((1-D[0,0]*np.power(y1,2))/(D[1,1]))
y3 = -1*np.sqrt((1-D[0,0]*np.power(y1,2))/(D[1,1]))
y = np.hstack((np.vstack((y1,y2)),np.vstack((y1,y3))))

#Plotting standard hyperbola
plt.plot(y[0,:len],y[1,:len],color='b',label='Std hyperbola')
plt.plot(y[0,len+1:],y[1,len+1:],color='b')

#Plotting standard hyperbola
#plt.plot(y[0,:],y[1,:],label='Ellipse at origin')

#Affine Transformation
#Equation : y = P.T@(x-c)/(K**0.5)
x = (P @ (y)) * F**0.5

#Plotting required hyperbola
plt.plot(x[0,:len],x[1,:len],color='r',label='Hyperbola H')
plt.plot(x[0,len+1:],x[1,len+1:],color='r')

ax.plot()
plt.xlabel('$x$');plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()

# #if using termux
plt.savefig('./figs/hyperbola.pdf')
plt.savefig('./figs/hyperbola.eps')
subprocess.run(shlex.split("termux-open ./figs/hyperbola.pdf"))
# #else

#plt.show()
