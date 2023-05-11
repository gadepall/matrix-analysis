import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/home/pratik/CoordGeo/')

from line.funcs import *
from conics.funcs import *

def mu_i(V, u, m, h, f):
    r1 = m.T@V@m
    r2 = m.T@(V@h+u)
    g = h.T@V@h + 2*u.T@h + f
    mu_1 = (-1*r2 + np.sqrt(r2**2 - r1*g))/(r1)
    mu_2 = (-1*r2 - np.sqrt(r2**2 - r1*g))/(r1)

    return mu_1, mu_2


V = np.array([[4/9, 0], [0, 1]])
f = -4
u = np.array([[0], [0]])

# Centre of the ellipse
C = -1*np.linalg.inv(V)@u
print(f'Centre of the Ellipse : {C}')

# Major Axes of the ellipse
m = np.array([[1], [0]])

# Vertices of the ellipse
mu1, mu2 = mu_i(V, u, m, C, f)
v1 = C+ mu1*m
print(v1)
v2 = C+mu2*m
print(f'Vertices of the Ellipse : {v1} and {v2}' )

# Length of the major axes
print(f'Length of the major axes : {np.linalg.norm((mu1-mu2)*m)}')

# Length of the minor axes
n = m
mu1, mu2 = mu_i(V, u, n, C, f)
print(f'Length of the minor axes : {(np.linalg.norm((mu1-mu2)*n))}')

# Ecentricity of the ellipse
r3 = (np.linalg.norm(n)**2)*np.array([[1, 0], [0, 1]]) - V
if np.linalg.det(n@n.T) != 0:
    e = np.sqrt(r3@np.linalg.inv(n@n.T))
elif n[0] != 0:
    e = np.sqrt((r3[0][0])/n[0]**2)
else:
    e = np.sqrt((r3[1][1])/n[1]**2)

print(f'Ecentricity of the ellipse :{e}')

# Foci of the ellipse
s1 = e**4*(np.linalg.norm(n)**2) + np.linalg.norm(u)**2 - e**2
s2 = -2*(e**2)*(u.T@n)
s3 = -1*f
coeffs = [s1, s2, s3]
c = np.roots(coeffs)[1]
F = (c*e**2*n - u)/np.linalg.norm(n)**2

print(f'Focus of the ellipse : {F}')

#LATUS RECTA
l = np.array([[0,1],[-1,0]])@m

mu_lat1 , mu_lat2 = mu_i(V,u,l,F,f)

print(f'Length of latus recta : {np.linalg.norm((mu_lat1-mu_lat2)*l)}')

#PLOTS

c1 = circ_gen(u.T,1)

c1 = np.array([[3,0],[0,2]]) @ c1
plt.plot(c1[0],c1[1])

plt.scatter(F[0],F[1])
plt.text(F[0],F[1]+0.2,'F')

plt.scatter(C[0],C[1])
plt.text(C[0],C[1]+0.2,'C')

plt.scatter(v1[0],v1[1])
plt.text(v1[0],v1[1]+0.2,'V1')

plt.scatter(v2[0],v2[1])
plt.text(v2[0]+0.1,v2[1]+0.2,'V2')

lat= line_gen(F+mu_lat1*l, F+mu_lat2*l)
plt.plot(lat[0],lat[1])

maj =line_gen(v1,v2)
plt.plot(maj[0],maj[1])
plt.grid()
plt.savefig("../figs/Figure_1.png")
