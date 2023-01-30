#Code by GVV Sharma
#Jan 19, 2020
#Released under GNU GPL
#Semidefinite program example
#using cvx
import numpy as np
from cvxpy import *


#Parabola parameters
P = np.array([0,5]).reshape(2,-1)
q_0 = -2*P
Q_0 = np.array([[1,0],[0,0]])
Q_1 = np.eye(2)
q_1 = -2*np.array([0,1]).reshape(2,-1)
c_0 = np.linalg.norm(P)^2
x = Variable((2,1))
theta = Variable
A = np.block([[np.eye(2),-n],[n.T, 0]])
#Cost function
f = theta
#f =  quad_form(x-P, np.eye(2))
obj = Minimize(f)

#Constraints
#constraints = [quad_form(x,V) + u.T@x +7 <= 0]
constraints = [quad_form(x,V) + u.T@x +7 <= 0]

#solution
Problem(obj, constraints).solve()

print(np.sqrt(f.value),x.value)



