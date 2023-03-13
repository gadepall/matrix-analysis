import numpy as np

a = np.array([1,1])
b = np.array([2,1])

X = np.cross((a-b),(a+b))
Y = 2*(np.cross(a,b))
print("LHS :",X)
print("RHS :",Y)

