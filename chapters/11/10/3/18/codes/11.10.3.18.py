import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/home/pratik/CoordGeo')

from line.funcs import *

A = np.array([[2],[0]])
B = np.array([[0],[3]])

m = A-B
Rot = np.array([[0,1],[-1,0]])
n = Rot @ m

O = np.array([[0],[0]])

dist = abs(n.T @ (O-A))/np.linalg.norm(n)

print(dist) 

x_AB =line_gen(A,B)

x = np.linspace(0,5,200)
y = 0*x

plt.plot(x_AB[0],x_AB[1])
plt.plot(x,y)
plt.plot(y,x)
plt.grid()

plt.text(A[0],A[1]+0.2,'x-intrecept (a,0)')
plt.text(A[0],A[1]-0.2,'(2,0)')

plt.text(B[0],B[1]+0.2,'y-intrecept (0,b)')
plt.text(B[0]+0.1,B[1]-0.2,'(0,3)')



plt.show()

