#Solution to the allocation problem
import pulp as p
from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt

#if using termux
import subprocess
import shlex
#end if

#Line parameters
from cvxpy import *
import numpy as np

A = np.array(( [1, 1], [20, 10 ])).T
b = np.array([ 50, 800 ]).reshape((2,-1))
c = np.array([ 10500, 9000 ])

x = Variable((2,1))

f = c @ x
obj = Maximize(f)

constraints = [A.T @ x <= b]


Problem(obj, constraints).solve()

print("Number of hectares of land with X crop",x.value[0])
print("Number of hectares of land with Y crop",x.value[1])
print("Maximum profit =",f.value)
#
