import numpy as np
from fractions import Fraction

# Define the vector
vec = np.array([-18, 12, -4])
# Find the magnitude of the vector
mag = np.linalg.norm(vec)
B=vec/mag
print(B)
