import numpy as np
from fractions import Fraction

# Define the vector
a = np.array([1,3,7])
b  = np.array([7,-1,8])
# Find the magnitude of the vector
mag = (np.linalg.norm(b))**2

# Divide the vector by its magnitude to find the direction cosines
c = (((a.T)@b) / mag)*b

frac = [Fraction(x).limit_denominator() for x in c]
print(frac)
print(c)

