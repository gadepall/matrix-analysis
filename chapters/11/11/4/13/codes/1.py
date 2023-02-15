import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'/home/jaswanth/6th-Sem/EE2802/CoordGeo')

from line.funcs import *
from conics.funcs import hyper_gen
from params import *

V = np.array([[-1/3,0],[0,1]])
u = np.zeros((2,1))
f = 4

lamda, v = np.linalg.eig(V)
lamda1 = lamda[0]
lamda2 = lamda[1]

print(lamda1, lamda2)

a = np.sqrt(abs(f/lamda1))
b = np.sqrt(abs(f/lamda2))
print(a,b)

arr = np.arange (-3,3,0.001)
arr_1 = np.arange (-10,12,2)
M = np.array([[a,0],[0,b]])
# plt.figure(figsize = (5,5))

plt.plot(hyper_gen(arr)*M[0][0],arr*M[1][1], color = 'orange')
plt.plot(-hyper_gen(arr)*M[0][0],-arr*M[1][1], color = 'orange')
plt.vlines(x = 0, ymin = -6, ymax= 6,  linestyles='--')
plt.hlines(y = 0, xmin = -11, xmax = 11, linestyles='--')
plt.xticks(arr_1)
plt.grid()
plt.tight_layout()
plt.savefig("fig1.png")
# plt.show()
