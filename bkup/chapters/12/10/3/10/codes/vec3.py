import numpy as np

A = []
for i in range(3):
    A.append(float(input("Enter the x, y and z coordinate for A : ")))
A = np.array(A)

B = []
for i in range(3):
    B.append(float(input("Enter the x, y and z coordinate for B : ")))
B = np.array(B)

# Orthogonality calculation

orth = A.T@B

if orth == 0:
    print("A and B are orthogonal")
else:
    print("A and B are not orthogonal")
