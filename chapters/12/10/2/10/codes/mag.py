import numpy as np

#Given vector
A = np.array([[5.0],[-1.0],[2.0]])

#Given magnitude
m = 8.0

#Calculate the constant
c = m/np.linalg.norm(A)

#Compute the required vector
v = c*A
print(v)
