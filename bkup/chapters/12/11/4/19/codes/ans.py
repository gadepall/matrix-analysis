import numpy as np
import sympy as sp

#Normal of planes
n1 = np.array([1, -1, 2])
n2 = np.array([3,1,1])

#Point on line
A = np.array([1,2,3])

#Direction of line
m = np.cross(n1,n2)

lambda1 = sp.Symbol('lambda')
print("Equation of the line ", A + lambda1*m)
