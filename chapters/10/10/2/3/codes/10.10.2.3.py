import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/home/pratik/CoordGeo')

from line.funcs import *
from triangle.funcs import *
from conics.funcs import *

ang_APB = np.radians(80)
theta = ang_APB /2
r = 5
O = np.array([[0],[0]])
A = np.array([[0],[r]])

#Line of angle

n1 = np.array([[-np.sin(theta)],[np.cos(theta)]])
c1 = 0

#Tangent Line

n2 = np.array([[-1,0],[0,1]]) @ (A -O)
c2 = n2.T @ A


##Solving for P
cond = np.vstack((n1.T,n2.T))
res = np.array([[c1],[c2[0][0]]])

P = np.linalg.solve(cond,res)

print(f"P={P}")

#Solving for angle

cos = (P-O).T @ (A-O)
cos /= np.linalg.norm(P-O)*np.linalg.norm(A-O)
phi = np.degrees(np.arccos(cos))

print(f"phi={phi}")


## Finding point B

OP_unit = (P-O)
#OP_unit = (OP_unit)/np.linalg.norm(OP_unit)

n = np.array([[0,1],[-1,0]]) @ OP_unit

plt.scatter(A[0],A[1])
plt.text(A[0],A[1],'A')
u = n.T@(O-A)/(n.T@n)
B = A + 2*u*n

##PLOTS

c1 = circ_gen(O.T,r)
plt.plot(c1[0],c1[1])

tan1 = line_gen(A,P)
plt.plot(tan1[0],tan1[1])

tan2 = line_gen(B,P)
plt.plot(tan2[0],tan2[1])


rad1 = line_gen(O,A)
plt.plot(rad1[0],rad1[1])

rad2 = line_gen(O,B)
plt.plot(rad2[0],rad2[1])

lin= line_gen(O,P)
plt.plot(lin[0],lin[1])

plt.scatter(A[0],A[1])
plt.text(A[0],A[1],'A')

plt.scatter(B[0],B[1])
plt.text(B[0],B[1],'B')

plt.scatter(O[0],O[1])
plt.text(O[0]-0.3,O[1],'O')

plt.scatter(P[0],P[1])
plt.text(P[0]+0.2,P[1],'P')

plt.xlim(-7,7)
plt.ylim(-7,7)
plt.grid()
plt.show()
