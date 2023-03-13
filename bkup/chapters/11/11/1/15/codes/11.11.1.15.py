import scipy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/home/pratik/CoordGeo')

from line.funcs import *
from conics.funcs import *

O = np.array([[0],[0]])
r = 5
A = np.array([[-2.5],[3.5]])

OA = np.linalg.norm(O-A)

error = OA -r
print(error)

if error < 0.01 :
    print('point lies in the circle')
elif error > 00.01 :
    print('point lies outside the circle')
else:
    print('point lies on the circle')

c1 = circ_gen( O.T, r)

plt.figure(figsize=(4,4))
plt.plot(c1[0],c1[1])

plt.plot(O[0][0],O[1][0], 'go')
plt.text(O[0][0]+0.2,O[1][0],'O (0,0)')
plt.plot(A[0][0],A[1][0],'go')
plt.text(A[0][0]+0.2,A[1][0],'A (-2.5,3.5)')

plt.grid()
plt.show()
