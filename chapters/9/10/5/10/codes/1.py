import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,'/home/jaswanth/6th-Sem/EE2802/CoordGeo')
from line.funcs import *
from conics.funcs import *

def g(x, V, u, f):
    return x.T@V@x + 2*u.T@x + f

A = np.array([[0],[4]])
B = np.array([[0],[-4]])
C = np.array([[6],[6]])

C1 = (A+B)/2
r1 = np.linalg.norm(A-B)/2


C2 = (A+C)/2
r2 = np.linalg.norm(A-C)/2

c1 = circ_gen(C1.reshape(1,2), r1)
c2 = circ_gen(C2.reshape(1,2), r2)

# Equation of circle 1
V1 = np.eye(2)
u1 = -C1
f1 = np.linalg.norm(u1)**2 - r1**2

# Parameters of common chord
h = np.array([[0],[4]])
m = np.array([[-5],[3]])

# Condition for intersection
Del = (m.T@(V1@h + u1))**2 - g(h, V1, u1, f1) *(m.T@V1@m)
print(Del>0)

# Points of intersection of circle_1,line
a = m.T@V1@m
b = 2*m.T @(V1@h + u1)
c= g(h, V1, u1, f1)

alpha1= (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
beta1 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)

P = h + beta1*m
print(P)

# Verifying Point P lies on Equation of line BC
m1 = B -C
n1 = omat@m1
print(n1.T@(P-B))

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

plt.figure(figsize=(7,7.8))
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_BC[0,:],x_BC[1,:])
plt.plot(x_CA[0,:],x_CA[1,:])

A. resize(1,2)
B. resize(1,2)
C. resize(1,2)

tri_coords = np.vstack((A,B,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,0), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.plot(c1[0], c1[1])
plt.plot(c2[0], c2[1])

plt.plot(C1[0],C1[1], 'go')
# plt.text(C[0]+0.2,C[1],'C1)')
plt.plot(C2[0],C2[1],'go')
# plt.text(A[0]+0.2,A[1],'A (4,5)')

plt.grid()
plt.savefig("fig.png")
# plt.show()
