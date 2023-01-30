import numpy as np
import cvxpy as cp

#if using termux
import subprocess
import shlex
#end if

Cost = np.array([[10],[-70]])
costcon = 1900

A = np.array([[1,1],[-1,-1]])
b = np.array([[8],[-4]])

C = np.array([[1,0],[0,1]])
d = np.array([[5],[5]])

x = cp.Variable((2,1),nonneg=True)

J = cp.matmul(Cost.T,x) + costcon
obj = cp.Minimize(J)

constraints = [ A@x <= b , C@x <= d]

cp.Problem(obj,constraints).solve()

print("Cost is ",J.value[0][0],"\n",x.value[0][0],"units must be transported from factory P to A\n",x.value[1][0],"units must be transported from factory P to B")
print("",8 - x.value[0][0]-x.value[1][0],"units must be transported from factory P to C")


import matplotlib.pyplot as plt
from coeffs import *

orth = np.array([[0,1],[-1,0]])

n1 = np.array([[1],[1]])
n2 = np.array([[-1],[-1]])

c1 = b[0]
c2 = b[1]

m1 = orth@n1
m2 = orth@n2


pa = np.array([[4],[4]])
pa1 = pa + (2*m1)
pa2 = pa - (2*m1)

pb = np.array([[2],[2]])
pb1 = pb + (2*m2)
pb2 = pb - (2*m2)

x_AA = line_gen(pa1,pa2)
x_BB = line_gen(pb1,pb2)


plt.plot(x_AA[0,:],x_AA[1,:],label="[1 1]x<=8")
plt.plot(x_BB[0,:],x_BB[1,:],label="[-1 -1]x<=4")



x_XX = line_gen(np.array([[0],[0]]),np.array([[5],[0]]))
x_YY = line_gen(np.array([[0],[0]]),np.array([[0],[5]]))

plt.plot(x_XX[0,:],x_XX[1,:],label="[1 0]x>=0")
plt.plot(x_YY[0,:],x_YY[1,:],label="[0 1]x>=0")

nx = np.array([[1],[0]])
mx = orth@nx
px = np.array([[5],[0]])
px1 = px + (0.5*mx)
px2 = px - (4*mx)

ny = np.array([[0],[1]])
my = orth@ny
py = np.array([[0],[5]])
py1 = py + (4*my)
py2 = py - (0.5*my)

mxx = line_gen(px1,px2)
myy = line_gen(py1,py2)

plt.plot(mxx[0,:],mxx[1,:],label="[1 0]x<=5")
plt.plot(myy[0,:],myy[1,:],label="[0 1]x<=5")


plt.scatter(x.value[0],x.value[1],label="OPT PT")
plt.text(x.value[0],x.value[1],"OPT PT")


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('./figs/lp_transport.pdf')
plt.savefig('./figs/lp_transport.eps')
subprocess.run(shlex.split("termux-open ./figs/lp_transport.pdf"))
#plt.legend()
#plt.grid()
#plt.axis("equal")
#plt.show()
