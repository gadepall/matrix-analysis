#Find the equation of the plane through the intersection of the planes 3x–y + 2z–4 = 0 and x + y + z–2 = 0 and the point (2,2,1)

import numpy as np
import sympy as sp

# Define the planes
#p = n'x - c
n1 = np.array([3, -1, 2])
n2 = np.array([1,1,1])
c1 = 4
c2 = 2

lambda_ = sp.symbols('lambda')

#given point is x
x = np.array([2,2,1])

equation = (n1 + lambda_*(n2))@x - (c1 + lambda_*(c2))

lambda_ = sp.solve(equation, lambda_)[0]

normal = (n1 + lambda_*(n2))
print('The equation of plane is {}x + {}y + {}z + {} = 0'.format(normal[0], normal[1], normal[2], -(c1 + lambda_*(c2))))