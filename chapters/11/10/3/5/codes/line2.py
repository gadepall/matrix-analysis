import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from numpy import linalg as LA
import math
import subprocess
import shlex


def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

A = np.array([-3,8])
B = np.array([9,-8])  
M = np.array([4,3])

n = np.array([4,3])
c = 12
d = 4
e1 = np.array([1,0])
n1 = n[0]*n[0] + n[1]*n[1]
norm_n = np.sqrt(n1)

x1 = (d*norm_n+c)/n.T@e1
x2 = (-d*norm_n+c)/n.T@e1

print(f"The two points are ({x1,0}) and ({x2,0})")

P = np.array([x2,0])
Q = np.array([x1,0])
C = np.array([6/5,12/5])
D = np.array([24/5,-12/5])

#generating lines
x_AB = line_gen(A,B)
x_PC = line_gen(P,C)
x_QD = line_gen(Q,D)

#plt.plot(x_AB[0,:],x_AB[1,:],label='Given line segment')
plt.plot(x_AB[0,:],x_AB[1,:],label='{}X={}'.format(M,c) )
plt.plot(x_PC[0,:],x_PC[1,:],label='PC')
plt.plot(x_QD[0,:],x_QD[1,:],label='QD')

sqr_vert = np.vstack((P,Q,C,D)).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
vert_labels = ['P(-2,0)','Q(8,0)','C','D']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (sqr_vert[0,i], sqr_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')

plt.xlabel('$x$')                    
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.savefig('/sdcard/download/latexfiles/line/figs/line2.png')
plt.show()            
