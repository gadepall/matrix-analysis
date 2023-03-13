import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/home/pratik/AlphaBot2/Bluetooth-Cont')

from conics.funcs import *

Vert = np.array([[0],[3]])
Focus = np.array([[0],[5]])
O = np.array([[0],[0]])
e = np.linalg.norm(Focus-O)/np.linalg.norm(Vert-O)

n = (Vert -O)
n = n/np.linalg.norm(n)

print(f"n = {n}")

c =  n.T @ (O + (Vert -O)/e)

V = np.linalg.norm(n)**2*np.array([[1,0],[0,1]]) - e**2* n @ n.T

u = c*e**2*n - np.linalg.norm(n)**2*Focus

f = np.linalg.norm(n)**2*np.linalg.norm(Focus)**2 - c**2*e**2

print(f"Conic Equation: x^T {V} x + {2*u.T} x + {f} = 0")

len = 200
x_hyp = np.zeros((2,len))
x_hyp[1,:] = np.linspace(-4,4,len)
x_hyp[0] = hyper_gen(x_hyp[1])
hyper = np.zeros((len,2,1))



hyper = np.array(list(zip(x_hyp[1],x_hyp[0]))).T

hyper = np.array([[4,0],[0,3]]) @ hyper



plt.plot(Focus[0],Focus[1],'bo')

F_dash = 2*O-Focus
plt.plot(F_dash[0],F_dash[1],'bo')

plt.text(Focus[0],Focus[1]+0.8,'F1')
plt.text(F_dash[0],F_dash[1]-1,'F2')

plt.plot(Vert[0],Vert[1],'go')

V_dash = 2*O-Vert
plt.plot(V_dash[0],V_dash[1],'go')

plt.text(Vert[0],Vert[1]-1.2,'V1')
plt.text(V_dash[0],V_dash[1]+0.8,'V2')

plt.plot(hyper[0],hyper[1],'r')
plt.plot(hyper[0],-1*hyper[1],'r')



plt.grid()

plt.show()
