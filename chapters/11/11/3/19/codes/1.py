import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/home/jaswanth/6th-Sem/EE2802/CoordGeo')

from line.funcs import *
from conics.funcs import ellipse_gen
from params import *

A = np.array([[4,-1],[36,-1]])
B = np.array([[13],[37]])

C1 = np.linalg.solve(A,B)

e = np.sqrt(C1[0][0])
f = C1[1][0]
print(e,f)

# Given major axis on y-axis
n = np.array([[0],[1]])
V = (np.linalg.norm(n)**2 * np.eye(2)) - ((e**2)* (n@n.T))
print(V)

# Given center of the ellipse is origin
C = np.array([[0],[0]])
u = -1 * V@C
print(u)
print(f)

lamda, v = np.linalg.eig(V)
lamda1 = lamda[0]
lamda2 = lamda[1]

print(lamda1, lamda2)

f0 = ((u.T)@(np.linalg.inv(V))@u) - f
print(f0)
f0 = f0[0][0]

a = np.sqrt(abs(f0/lamda1))
b = np.sqrt(abs(f0/lamda2))
print(a,b)

P = np.array([[3],[2]])
Q = np.array([[1],[6]])

E = ellipse_gen(a,b)
plt.figure(figsize = (4.5,6))
plt.plot(E[0], E[1])
plt.scatter(C[0][0], C[1][0], color = 'orange')
plt.scatter(P[0][0], P[1][0], color = 'orange')
plt.text(P[0][0], P[1][0]+0.35, "P(3,2)")
plt.scatter(Q[0][0], Q[1][0], color = 'orange')
plt.text(Q[0][0]+0.15, Q[1][0]+0.15, "Q(1,6)")
plt.vlines(x = 0, ymin = -7.5, ymax = 7.5, colors= "red")
plt.grid()
plt.tight_layout()
plt.savefig("fig1.png")
