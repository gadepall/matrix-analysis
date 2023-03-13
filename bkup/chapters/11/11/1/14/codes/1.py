import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0,'/home/jaswanth/6th-Sem/EE2802/CoordGeo')
from line.funcs import *
from conics.funcs import *

C = np.array([2,2])
A = np.array([4,5])

U = -C
f = -A@(A.T)- 2* A@(U.T)
print(f)

r = np.sqrt(np.linalg.norm(U)**2-f)

c1 = circ_gen(C, r)

# print(c1)
plt.figure(figsize=(4,4))
plt.plot(c1[0], c1[1])

plt.plot(C[0],C[1], 'go')
plt.text(C[0]+0.2,C[1],'C (2,2)')
plt.plot(A[0],A[1],'go')
plt.text(A[0]+0.2,A[1],'A (4,5)')

plt.grid()
plt.savefig("fig.png")
plt.show()