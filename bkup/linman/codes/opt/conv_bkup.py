#Code by GVV Sharma
#December 25, 2019
#Released under GNU GPL
#Convex function
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import subprocess
import shlex
from coeffs import *


def f(x,a,b,d):
	return a*x**2+b*x+d

#Line parameters
n =  np.array([3,-4]) 
c = 26
P = np.array([3,-5]) 

A,B = line_icepts(n,c)
m = omat@n

#Parabola parameters
a = np.linalg.norm(A)**2
b = m@(A-P)
d = np.linalg.norm(A-P)**2

#Plotting the parabola
x = np.linspace(-5,5,50)#points on the x axis
vec_f = scipy.vectorize(f)
#Objective function
f=vec_f
plt.plot(x,f(x,a,b,d),color=(1,0,1))
plt.grid()
plt.xlabel('$x$')
plt.ylabel('$\ln x$')

#Convexity/Concavity
lam1 = -3
lam2 = 4
t = 0.3
lam = t *lam1 + (1-t)*lam2
f_a = f(lam1,a,b,d)
f_b = f(lam2,a,b,d)

f_c = f(lam,a,b,d)
f_c_hat = t *f_a + (1-t)*f_b

#Plot commands
plt.plot([a,a],[0,f_a],color=(1,0,0),marker='o',label="$f(a)$")
plt.plot([b,b],[0,f_b],color=(0,1,0),marker='o',label="$f(b)$")
plt.plot([c,c],[0,f_c],color=(0,0,1),marker='o',label="$f(\lambda a + (1-\lambda)b)$")
plt.plot([c,c],[0,f_c_hat],color=(1/2,2/3,3/4),marker='o',label="$\lambda f(a) + (1-\lambda)f(b)$")
plt.plot([a,b],[f_a,f_b],color=(0,1,1))
plt.legend(loc=2)
#plt.savefig('../figs/1.2.pdf')
#subprocess.run(shlex.split("termux-open ../figs/1.2.pdf"))
#plt.show()#Reveals the plot
#if using termux
plt.savefig('./figs/convex.pdf')
plt.savefig('./figs/convex.eps')
subprocess.run(shlex.split("termux-open ./figs/convex.pdf"))
#else
#plt.show()








