import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/home/pratik/CoordGeo')

from line.funcs import *
from conics.funcs import *


O = np.array([[5],[0]])
B = np.array([[0],[0]])
r = 5
d = 6

#line passing through points of intersection

n = 2*(B-O)
c = np.linalg.norm(B)**2- np.linalg.norm(O)**2 +r**2 -d**2
m = np.array([[0,1],[-1,0]]) @ n
h = np.array([[c/n[0][0]],[0]])
print(f'line passing through A and C:x={h}+u{m}')

# Circle with center B and radius d

V = np.array([[1,0],[0,1]])
u = -2*B
f = np.linalg.norm(B)**2 - d**2


#Claculating mu's
g = h.T @ V @ h + 2*u.T@h +f

mVm = m.T @ V @ m

alph =  -m.T @( V@h +u)

mu1 = (alph + np.sqrt(alph**2 - g*mVm))/mVm
mu2 = (alph - np.sqrt(alph**2 - g*mVm))/mVm   

A = h + mu1[0][0] * m
C = h+ mu2[0][0]*m

print(f'A={A}')

dist = np.linalg.norm(A-C)

print(dist)

##PLOTS

c1 = circ_gen(O.T,r)
c2 = circ_gen(B.T,d)

lin1 = line_gen(A,C)
lin2 = line_gen(B,C)
lin3 = line_gen(A,B)

plt.plot(c1[0],c1[1])
plt.plot(c2[0],c2[1])
plt.plot(lin1[0],lin1[1])
plt.plot(lin2[0],lin2[1])
plt.plot(lin3[0],lin3[1])

plt.scatter(A[0][0],A[1][0])
plt.scatter(O[0][0],O[1][0])
plt.scatter(C[0][0],C[1][0])
plt.scatter(B[0][0],B[1][0])


plt.text(A[0][0],A[1][0]+0.3,'A')
plt.text(O[0][0]+0.2,O[1][0],'O')
plt.text(C[0][0],C[1][0]-0.5,'C')
plt.text(B[0][0]-0.5,B[1][0],'B')






plt.grid()
plt.show()
