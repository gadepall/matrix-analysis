import numpy as np

# Define the vector
vec = np.array([1, 2, 3])
e1  = np.array([1,0,0])
e2  = np.array([0,1,0])
e3  = np.array([0,0,1])
# Find the magnitude of the vector
mag = np.linalg.norm(vec)
a = np.linalg.norm(e1)
b = np.linalg.norm(e2)
c = np.linalg.norm(e3)

# Divide the vector by its magnitude to find the direction cosines
dir_cos1 = ((vec.T)@e1) / (mag*a)
dir_cos2 = ((vec.T)@e2) / (mag*b)
dir_cos3 = ((vec.T)@e3) / (mag*c)

print(dir_cos1,dir_cos2,dir_cos3)
