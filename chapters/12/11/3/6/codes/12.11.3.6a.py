import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
#Generate line points
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

M = np.array([[1, 6, -4, 1], [1, 4, -2, 1], [-1, -5, 3, 1]])

if np.linalg.matrix_rank(M[:, :-1]) < np.linalg.matrix_rank(M):
    print("The system has no solution.")
else:
   solution = np.linalg.solve(M[:, :-1], M[:, -1])
   print('{}x={}'.format(solution, 1)) 

A= np.array([1,1,-1])
B= np.array([6,4,-5])
C= np.array([-4,-2,3])

m=B-C
print('direction vector(m) =', m)
# Calculate the normal vector of the plane
v1 = C - A
v2 = B - A
normal = np.cross(v1, v2)
print("x = {} + \u03BB{}".format(A, m))
# Define the plane function
def plane(x, y):
    return (-normal[0] * x - normal[1] * y - np.dot(normal, A)) * 1. / normal[2]

# Generate the meshgrid
xx, yy = np.meshgrid(range(-10, 11), range(-10, 11))

# Evaluate the plane function
z = plane(xx, yy)
# Plot the plane
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, z)
ax.scatter(A[0], A[1], A[2], color='red')
ax.text(A[0],A[1],A[2], 'A')
ax.scatter(B[0], B[1], B[2], color='red')
ax.text(B[0],B[1],B[2], 'B')
ax.scatter(C[0], C[1], C[2], color='red')
ax.text(C[0],C[1],C[2], 'C')
x_CB=line_gen(C,B)
plt.plot(x_CB[0],x_CB[1],x_CB[2])
plt.show()
