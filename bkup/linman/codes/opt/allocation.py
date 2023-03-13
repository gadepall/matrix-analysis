# solution for problem 7.8 from https://github.com/gadepall/school/blob/master/ncert/optimization/gvv_ncert_opt.pdf
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt
#if using termux
import subprocess
import shlex
#end if

A = np.array(( [1, 1], [20, 10 ]))
b = np.array([ 50, 800 ]).reshape((2,-1))
c = np.array([ 10500, 9000 ])

X = Variable((2,1))

f = c @ X
obj = Maximize(f)

constraints = [A @ X <= b]

Problem(obj, constraints).solve()
print("Number of hectares of land with X crop",X.value[0])
print("Number of hectares of land with Y crop",X.value[1])
print("Maximum profit =",f.value)

#for plotting
x=np.linspace(0,50,100)

try:
    Y1= (b[0] - A[0][0]*x)/A[0][1]
except ZeroDivisionError:
    Y1  =b[0]/A[0][0]

try:
    Y2= (b[1] - A[1][0]*x)/A[1][1]
except ZeroDivisionError:
    Y2  =b[1]/A[1][0]

plt.plot(x,Y1,label="X+Y=50")
plt.plot(x,Y2,label="20*X+10*Y=800")
Y3=np.zeros(x.shape)
plt.plot(x,Y3,label="Y=0")
plt.plot(Y3,x,label="X=0")
plt.plot([30],[20],"o",label="Optimzed point",color="k")
plt.fill_between(x, Y1, Y2,alpha=0.5,color="r")
plt.fill_between(x,Y1,Y3,alpha=0.5)
plt.fill_between(x,Y2,Y3,alpha=0.5)
plt.xlabel("X, Number of hectares cultivating x")
plt.ylabel("Y, Number of hectares cultivating y")
plt.legend()
plt.grid()
#plt.show()

#if using termux
plt.savefig('./figs/lp_allocation.pdf')
plt.savefig('./figs/lp_allocation.eps')
subprocess.run(shlex.split("termux-open ./figs/lp_allocation.pdf"))
