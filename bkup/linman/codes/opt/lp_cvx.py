#Code by GVV Sharma, 
#December 29, 2018
#Released under GNU GPL
#Solving a linear program using cvx
from cvxpy import *
import numpy as np
#from numpy import matrix

A = np.array(( [1.0, 1.0], [3.0, 1.0 ])).T
b = np.array([ 50.0, 90.0 ]).reshape((2,-1))
c = np.array([ 4.0, 1.0 ])

x = Variable((2,1),nonneg=True)
#Cost function
f = c@x
obj = Maximize(f)
#Constraints
constraints = [A.T@x <= b]

#solution
Problem(obj, constraints).solve()

print(f.value,x.value)

