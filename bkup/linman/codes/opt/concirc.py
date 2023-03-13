#Code by GVV Sharma
#Jan 12, 2019
#Released under GNU GPL
#Lagrange Multipliers
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shlex
from coeffs import *

#Line parameters
n =  np.array([3,-4]) 
c = 26
P = np.array([3,-5]) 
# Point of contact
Q = np.array([2.64,-4.52]) 

#Plotting the circles
numcirc = 5
x = P[0]*np.ones(numcirc)
y = P[1]*np.ones(numcirc)
rmin =0.6
r = np.arange(numcirc)*rmin/(numcirc-1)
phi = np.linspace(0.0,2*np.pi,100)
na=np.newaxis
# the first axis of these arrays varies the angle, 
# the second varies the circles
x_line = x[na,:]+r[na,:]*np.sin(phi[:,na])
y_line = y[na,:]+r[na,:]*np.cos(phi[:,na])
ax=plt.plot(x_line,y_line,'-')

#Plotting the line
k = np.array([7,8])
x_AB=line_norm_eq(n,c,k)
x_PQ=line_gen(P,Q)
bx=plt.plot(x_AB[0,:],x_AB[1,:])
cx=plt.plot(x_PQ[0,:],x_PQ[1,:])
plt.text(Q[0]*(1), Q[1]*(1-0.02),'Q')
plt.text(P[0]*(1), P[1]*(1-0.01),'P')
plt.axis('equal')
plt.grid()
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend([ax[numcirc-1], bx[0], cx[0]],['$||x-P||=r$','$n^Tx = c$', 'PQ'], loc='best')

#if using termux
plt.savefig('./figs/concirc.pdf')
plt.savefig('./figs/concirc.eps')
subprocess.run(shlex.split("termux-open ./figs/concirc.pdf"))
#else
#plt.show()









