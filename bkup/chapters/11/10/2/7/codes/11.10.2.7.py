import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/pratik/CoordGeo')

#local imports
from line.funcs import *

A= np.array([[-1],[1]])
B= np.array([[2],[-4]])

m = A -B

R = np.array([[0,1],[-1,0]])
n = R @ m

print( n.T,'(x-',A,')')

x_AB = line_gen(A,B)

plt.plot(x_AB[0],x_AB[1])

plt.plot(A[0],A[1],'bo')
plt.text(A[0]+0.08,A[1],'A')

plt.plot(B[0],B[1],'bo')
plt.text(B[0]+0.08,B[1],'B')

plt.grid()
plt.show()
