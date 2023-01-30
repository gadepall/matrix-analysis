#Code by Amey Waghmare, 
#Jan 16, 2020
#Revised by GVV Sharma
#Jan 17, 2020
#Released under GNU GPL
#Quadratic program example
#Minimum distance from a point to a parabola
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shlex
from coeffs import *

x = np.linspace(-2,2,100)
y = (x**2)+7
P = np.array([3,7])
Q = np.array([1.00002559,8.00005117])

ax=plt.plot(x,y)
plt.grid()
plt.axis('equal')


bx=plt.scatter(Q[0],Q[1])
plt.text(Q[0]+0.1,Q[1]+0.1,"Q")
plt.scatter(P[0],P[1])
plt.text(P[0]+0.1,P[1]+0.1,"P")


plt.legend(['$y = x^2+7$'])


#if using termux
plt.savefig('./figs/qp_parab.pdf')
plt.savefig('./figs/qp_parab.eps')
subprocess.run(shlex.split("termux-open ./figs/qp_parab.pdf"))
#else
#plt.show()






