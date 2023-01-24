import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/home/pratik/CoordGeo')

from line.funcs import *

a = 7 
b = 5
theta = np.pi/6

P = np.array([[np.sqrt(a**2-b**2)],[0]])
Q =  -P 

n = np.array([[np.cos(theta)/a],[-1*np.sin(theta)/b]])

A0 = np.array([[a/np.cos(theta)],[0]])

c = n.T @ A0

d1 = abs(n.T@P - c)/ np.linalg.norm(n)
d2 = abs(n.T@Q - c)/ np.linalg.norm(n)

error = (d1*d2 - b**2)

print(error)

x = np.linspace(-7,7,200)
y = (-1*n[0][0]*x + c[0])/n[1][0]

plt.plot(x,y)

plt.plot(P[0],P[1],'bo')
plt.text(P[0]+0.2,P[1],'P (6,0)')

plt.plot(Q[0],Q[1],'bo')
plt.text(Q[0]+0.2,Q[1],'Q (-6,0)') 


plt.show()
