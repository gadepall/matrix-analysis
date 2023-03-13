#Code by Amey Waghmare, 
#Jan 16, 2020
#Revised by GVV Sharma
#Jan 17, 2020
#Released under GNU GPL
#Quadratic program example
#using cvx
import numpy as np
from cvxpy import *


#Parabola parameters
P = np.array([3,7]).reshape(2,-1)
V = np.array([[1,0],[0,0]])
u = np.array([0,-1]).reshape(2,-1)

x = Variable((2,1))

#Cost function
f =  quad_form(x-P, np.eye(2))
obj = Minimize(f)

#Constraints
constraints = [quad_form(x,V) + u.T@x +7 <= 0]

#solution
Problem(obj, constraints).solve()

print(np.sqrt(f.value),x.value)



