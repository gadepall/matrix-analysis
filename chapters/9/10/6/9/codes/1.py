import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,'/home/jaswanth/6th-Sem/EE2802/CoordGeo')
from line.funcs import *
from conics.funcs import *

def g(x, V, u, f):
    return x.T@V@x + 2*u.T@x + f

C1 = np.array([[2],[0]])
C2 = np.array([[-2],[0]])

# Equation of circle 1
V1 = np.eye(2)
u1 = -C1
f1 = -4

# Equation of circle 2
V2 = np.eye(2)
u2 = -C2
f2 = -4

r1 = np.sqrt(np.linalg.norm(u1)**2-f1)
r2 = np.sqrt(np.linalg.norm(u2)**2-f1)

c1 = circ_gen(C1.reshape(1,2), r1)
c2 = circ_gen(C2.reshape(1,2), r2)

plt.figure(figsize=(5,5))
plt.plot(c1[0], c1[1])
plt.plot(c2[0], c2[1])

plt.plot(C1[0],C1[1], 'go')
plt.text(C1[0]+0.2,C1[1]+0.1,'$C_1$(2,0)')
plt.plot(C2[0],C2[1], 'go')
plt.text(C2[0]-1,C2[1]+0.1,'$C_2$(-2,0)')

# Parameters of the common chord of circles 1,2
n_1 = (u1 - u2)/np.linalg.norm(u1 - u2)
c_1 = (f1-f2)/2

e1 = np.array([[1],[0]])
m1 = omat@n_1
print(c_1)
print((n_1.T@e1))
h1 = ((c_1/(n_1.T@e1)))*e1

a1 = m1.T@V1@m1
b1 = 2*m1.T @(V1@h1 + u1)
c1= g(h1, V1, u1, f1)

alpha1= (-b1 + np.sqrt(b1**2 - 4*a1*c1))/2*a1
beta1 = (-b1 - np.sqrt(b1**2 - 4*a1*c1))/2*a1

A = h1 + alpha1*m1
B = h1 + beta1*m1
A.reshape(1,2)
B.reshape(1,2)

plt.plot(A[0],A[1])
plt.text(A[0]+0.2,A[1],'$A$(0,2)')
plt.plot(B[0],B[1])
plt.text(B[0]+0.2,B[1],'$B$(0,-2)')

# Parameters of the Line passing through A
h2 = A.reshape(2,1)
m2 = np.array([[2],[1]])

# Point P
a2 = m2.T@V1@m2
b2 = 2*m2.T @(V1@h2 + u1)
c2= g(h2, V1, u1, f1)

alpha2= (-b2 + np.sqrt(b2**2 - 4*a2*c2))/(2*a2)
beta2 = (-b2 - np.sqrt(b2**2 - 4*a2*c2))/(2*a2)
P = h2 + alpha2*m2
print(P)

# Point Q
a3 = m2.T@V2@m2
b3 = 2*m2.T @(V2@h2 + u2)
c3= g(h2, V2, u2, f2)

alpha3= (-b3 + np.sqrt(b3**2 - 4*a3*c3))/(2*a3)
beta3 = (-b3 - np.sqrt(b3**2 - 4*a3*c3))/(2*a3)
Q = h2 + beta3*m2
print(Q)

print(np.linalg.norm(B-P) == np.linalg.norm(B-Q))

A = A.reshape(1,2)[0]
B = B.reshape(1,2)[0]
P = P.reshape(1,2)[0]
Q = Q.reshape(1,2)[0]
print(A,B,P,Q)
x_AP = line_gen(A,P)
x_PQ = line_gen(P,Q)
x_BP = line_gen(B,P)
x_BQ = line_gen(B,Q)

plt.plot(x_AP[0,:],x_AP[1,:], 'g')
plt.plot(x_PQ[0,:],x_PQ[1,:], 'g')
plt.plot(x_BP[0,:],x_BP[1,:], ls = '--')
plt.plot(x_BQ[0,:],x_BQ[1,:], ls = '--')

tri_coords = np.vstack((P,Q)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','Q']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                                  xytext=(-10,0), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
arr = np.arange(-5,6)
plt.xticks(arr)
plt.yticks(arr)
plt.xlabel("$x$")
plt.ylabel("$y$")    
plt.grid()
plt.savefig("fig.png")
plt.show()
