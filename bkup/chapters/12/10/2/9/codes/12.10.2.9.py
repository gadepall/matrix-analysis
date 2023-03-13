import numpy as np
a = np.array([[2],[-1],[2]])
b = np.array([[-1],[1],[-1]])
u = a+ b
u_hat = u/np.linalg.norm(u)
print(u_hat)
