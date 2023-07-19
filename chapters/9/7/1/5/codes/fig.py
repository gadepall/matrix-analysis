import numpy as np
import matplotlib.pyplot as plt

def line_gen(A, B):
    len = 10
    dim = A.shape[0]
    x_AB = np.zeros((dim, len))
    lam_1 = np.linspace(0, 1, len)
    for i in range(len):
        temp1 = A + lam_1[i] * (B - A)
        x_AB[:, i] = temp1.T
    return x_AB

# Given information
a = 9
theta = np.radians(30)
A = np.array([0, 0])
e1 = np.array([1, 0])
B = a * e1
c = 8

# Calculate Q and P
Q = np.array([c * np.cos(theta), c * np.sin(theta)])
P = np.array([c * np.cos(theta), -c * np.sin(theta)])

# Generate lines AQ and AP
x_AQ = line_gen(A, Q)
x_AP = line_gen(A, P)

# Generate lines BP and BQ
x_BP = line_gen(B, P)
x_BQ = line_gen(B, Q)

# Plotting lines AB, AQ, AP, BP, and BQ
x_AB = line_gen(A, B)

plt.plot(x_AB[0, :], x_AB[1, :], label='$AB$')
plt.plot(x_AQ[0, :], x_AQ[1, :], label='$AQ$')
plt.plot(x_AP[0, :], x_AP[1, :], label='$AP$')
plt.plot(x_BP[0, :], x_BP[1, :], label='$BP$')
plt.plot(x_BQ[0, :], x_BQ[1, :], label='$BQ$')

# Labeling the coordinates
tri_coords = np.vstack((A, B, Q, P)).T
plt.scatter(tri_coords[0, :], tri_coords[1, :])
vert_labels = ['A', 'B', 'Q', 'P']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, (tri_coords[0, i], tri_coords[1, i]), textcoords="offset points",
                 xytext=(0, 10), ha='center')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.title('Triangles APB and AQB')
plt.show()
