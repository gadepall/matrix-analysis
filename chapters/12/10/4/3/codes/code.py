import numpy as np
from fractions import Fraction

# Define the vector
theta1=np.pi/3
theta2=np.pi/4
a1=np.cos(theta1)
a2=np.cos(theta2)
a3=np.sqrt(1-(a1**2)-(a2**2))
a = np.array([a1,a2,a3])

frac = [Fraction(x).limit_denominator() for x in a]
print(frac)
print(a)
