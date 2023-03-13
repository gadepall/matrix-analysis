import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/home/pratik/CoordGeo')

from conics.funcs import *
from line.funcs import *

length = 12
dist =3
theta = np.pi/3
theta2 = -1*np.pi/3

A = 12*np.array([[np.cos(theta)],[0]])
A2 =12*np.array([[np.cos(theta2)],[0]])
 

B = 12* np.array([[0],[np.sin(theta)]])
B2=12* np.array([[0],[np.sin(theta2)]])

m =A-B
m /= np.linalg.norm(m)

m2 =A2-B2
m2 /= np.linalg.norm(m2)

P = A- dist*m
P2 = A2-dist*m2

res = np.array([[np.cos(theta),np.cos(theta2)],[np.sin(theta),np.sin(theta2)]])
points = np.hstack((P,P2))

Q = res@ np.linalg.inv(points)

V = Q.T@Q
print(V)

#PLOTS
O = np.array([0,0])
c1 = circ_gen(O,1)

c1 = np.array([[9,0],[0,3]])@c1

plt.plot(c1[0],c1[1])

rod = line_gen(A,B)
plt.plot(rod[0],rod[1])

plt.scatter(P[0],P[1])
plt.text(P[0]+0.5,P[1]+0.5,'P')

plt.scatter(A[0],A[1])
plt.text(A[0],A[1]+0.5,'A')

plt.scatter(B[0],B[1])
plt.text(B[0]+0.5,B[1],'B')

plt.grid()
plt.savefig('../figs/Figure_1.png')
