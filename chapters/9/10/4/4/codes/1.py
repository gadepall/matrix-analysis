import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,'/home/jaswanth/6th-Sem/EE2802/CoordGeo')
from line.funcs import *
from conics.funcs import *

def g(x, V, u, f):
    return x.T@V@x + 2*u.T@x + f

# Parameters of circle
O = np.array([[0],[0]])
r1 = 2
r2 = 3
# Parameters of line
h = np.array([[1],[0]])
m = np.array([[0],[1]])

# Equation of circle 1
V1 = np.eye(2)
u1 = np.zeros((2,1))
f1 = -4

# Equation of circle 2
V2 = np.eye(2)
u2 = np.zeros((2,1))
f2 = -9

# Condition for intersection
Del = (m.T@(V1@h + u1))**2 - g(h, V1, u1, f1) *(m.T@V1@m)
print(Del>0)

# Points of intersection of circle_1,line
a1 = m.T@V1@m
b1 = 2*m.T @(V1@h + u1)
c1= g(h, V1, u1, f1)

alpha1= (-b1 + np.sqrt(b1**2 - 4*a1*c1))/2*a1
beta1 = (-b1 - np.sqrt(b1**2 - 4*a1*c1))/2*a1

B = h + alpha1*m
C = h + beta1*m

# Points of intersection of circle_2, line
a2 = m.T@V2@m
b2 = 2*m.T @(V2@h + u2)
c2= g(h, V2, u2, f2)


alpha2= (-b2 + np.sqrt(b2**2 - 4*a2*c2))/2*a2
beta2 = (-b2 - np.sqrt(b2**2 - 4*a2*c2))/2*a2

A = h + alpha2*m
D = h + beta2*m

print(A,B,C,D, sep= "\n")
print(np.linalg.norm(A-B) == np.linalg.norm(C-D))
c1 = circ_gen(O.reshape((1,2)), r1)
c2 = circ_gen(O.reshape(1,2), r2)

plt.figure(figsize=(4,4))

tri_coords = np.vstack((A.reshape(1,2),B.reshape(1,2),C.reshape(1,2),D.reshape(1,2))).T

plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,0), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.plot(c1[0], c1[1])
plt.plot(c2[0], c2[1])

plt.vlines(x = 1, ymin = -4, ymax = 4, colors='green')
plt.plot(O[0],O[1], 'go')
plt.text(O[0]+0.2,O[1],'O(0,0)')
arr = np.arange(-4,5)
plt.xticks(arr)
plt.yticks(arr)
plt.grid()
plt.savefig("fig.png")
plt.show()