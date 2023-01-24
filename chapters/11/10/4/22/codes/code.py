import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'/home/jaswanth/6th-Sem/EE2802/CoordGeo')
from line.funcs import *

P = np.array([1,2])
A = np.array([13/5, 0])
Q = np.array([5,3])
R = np.array([5,-3])

n = np.array([0,1])
print(n@(P-A)/np.linalg.norm(P-A))
print(n@(Q-A)/np.linalg.norm(Q-A))

x_PA = line_gen(P,A)
x_AQ = line_gen(A,Q)
x_QR = line_gen(Q,R)
x_RP = line_gen(R,P)

plt.plot(x_PA[0,:],x_PA[1,:])
plt.plot(x_AQ[0,:],x_AQ[1,:])
plt.plot(x_QR[0,:],x_QR[1,:])
plt.plot(x_RP[0,:],x_RP[1,:])
plt.axhline(y = 0, linestyle = '--')


tri_coords = np.vstack((P,A,Q,R)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','A','Q','R']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(2,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid()
plt.savefig("Fig.png")
plt.show()
