import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/home/pratik/CoordGeo/lines/codes/CoordGeo/')

from line.funcs import *
from conics.funcs import *

def mu_i(V, u, m, h, f):
    r1 = m.T@V@m
    r2 = m.T@(V@h+u)
    g = h.T@V@h + 2*u.T@h + f
    mu_1 = (-1*r2 + np.sqrt(r2**2 - r1*g))/(r1)
    mu_2 = (-1*r2 - np.sqrt(r2**2 - r1*g))/(r1)

    return mu_1, mu_2


a = 5
c = 5
theta = np.pi/3

B = np.array([[0],[0]])
C = np.array([[a],[0]])
A = c*np.array([[np.cos(theta)],[np.sin(theta)]])

#SIDE LENGTH b

b = np.linalg.norm(A-C)

#CIRCUMCENTER
con1 = (B-C).T
res1 = (B-C).T@(B+C)/2

con2 = (B-A).T
res2 =(B-A).T@(B+A)/2

con = np.vstack((con1,con2))
res = np.vstack((res1,res2))

O = np.linalg.solve(con,res)
print(f"CIRCUMCENTER:{O}")

#PARAMETERS FOR CIRCUMCIRCLE
R = np.linalg.norm(O-B)
V = np.array([[1,0],[0,1]])
u = -1*O
f = O.T@O - R**2


#ANGULAR BISECTORS
m1 = B+C-2*A
m2 = A+C-2*B
m3 = A+B-2*C

print(f"m1:{m1}, m2:{m2}, m3:{m3}")

#FINDING D
mu_D = mu_i(V,u,m1,A,f)[0]
D = A + mu_D*m1

print(f"D:{D}")

#FINDING E
mu_E = mu_i(V,u,m2,B,f)[0]
E = B + mu_E*m2
print(f"E: {E}")

#FINDING F
mu_F = mu_i(V,u,m3,C,f)[0]
F = C + mu_F*m3
print(f"F: {F}")

cos_ang = (D-E).T@(F-E)/((np.linalg.norm(D-E))*(np.linalg.norm(F-E)))

cos_2E = 2*cos_ang**2 -1
print(f"cos(2E) : {cos_2E}")
angE = np.arccos(cos_ang)

print(f"ANGLE E:{angE}")
print(f"Error = {np.pi/2 - theta/2 - angE}")

#PLOTTING

c1 = circ_gen(O.T,R)
plt.plot(c1[0],c1[1])
plt.plot(A[0],A[1],'ro')
plt.text(A[0]*(1+0.1),A[1]*(1-0.1),'A')

plt.plot(B[0],B[1],'bo')
plt.text(B[0]*(1+0.1),B[1]*(1-0.1),'B')

plt.plot(C[0],C[1],'go')
plt.text(C[0]*(1+0.1),C[1]*(1-0.1),'C')

plt.plot(D[0],D[1],'ro')
plt.text(D[0]*(1+0.1),D[1]*(1-0.1),'D')

plt.plot(E[0],E[1],'bo')
plt.text(E[0]*(1+0.1),E[1]*(1-0.1),'E')

plt.plot(F[0],F[1],'go')
plt.text(F[0]*(1+0.1),F[1]*(1-0.1),'F')

plt.plot(O[0],O[1],'yo')
plt.text(O[0]*(1+0.1),O[1]*(1-0.1),'O')

plt.xlim(-1,6)
plt.ylim(-2,5)

plt.savefig('../figs/Figure_1.png')
plt.show()



