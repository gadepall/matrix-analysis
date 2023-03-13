#Find the vector equation of the line that passes through the points (3,-2,-5) and (3, -2, 6)
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

import sys
sys.path.insert(0, '/home/lokesh/EE2802/EE2802-Machine_learning/CoordGeo')

from line.funcs import *

lambda_ = symbols('lambda')

A = np.array([3, -2, -5])
B = np.array([3, -2, 6])

print('Vector equation of the line that passes through the points (3,-2,-5) and (3, -2, 6) is:', A + lambda_*(B-A))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
x, y, z = [A[0], B[0]], [A[1], B[1]], [A[2], B[2]]
ax.scatter(x, y, z, c='red', s=100)
ax.plot(x, y, z, color='black')
plt.savefig('/home/lokesh/EE2802/EE2802-Machine_learning/12.11.2.9/figs/line.png')