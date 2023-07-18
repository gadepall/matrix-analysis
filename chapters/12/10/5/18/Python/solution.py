import numpy as np

# Define the vector
e1  = np.array([1,0,0])
e2  = np.array([0,1,0])
e3  = np.array([0,0,1])

# calculate cross product of the vectors
a = np.cross(e2,e3)
b = np.cross(e1,e3)
c = np.cross(e1,e2)

# calculate dot product of the vectors
d = e1@a
e = e2@b
f = e3@c

# print result
print(d+e+f)
