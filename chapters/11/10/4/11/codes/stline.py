import numpy as np
import matplotlib.pyplot as plt
import math as ma
import matplotlib.pyplot as plt
from numpy import linalg as LA 

def slope(m):
    m1 = np.array([2, 1])  # direction vector of 1st line
    m2 = np.array([1, m])  # direction 2nd line
    cos_theta = np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2))
    return cos_theta

m1 = 1/2  # slope of the 1st line
theta = np.deg2rad(45)  # Angle

# Calculate the slopes using numpy roots
coeff = [3, -8, -3]  # Coefficients of the quadratic equation
roots = np.roots(coeff)  # Calculate the roots

# Extract the real roots
m2 = roots[np.isreal(roots)].real
m2_1 = m2[0]
m2_2 = m2[1]

print("Slope of the first line: ", m1)  # results
print("Slope of the second line (Case 1): ", m2_1)
print("Slope of the second line (Case 2): ", m2_2)

def line_dir_pt(m, P, k1, k2):
    length = 10
    dimensions = P.shape[0]
    x_AB = np.zeros((dimensions, length))
    lam_1 = np.linspace(k1, k2, length)
    for i in range(length):
        temp1 = P + lam_1[i] * m
        x_AB[:, i] = temp1.T
    return x_AB

# Input parameters
P = np.array([3, 2])
P1 = np.array([0, -3/2])

# Generating the lines
k1 = -10
k2 = 10  # Adjusted range for Line 1
x_m1P = line_dir_pt(np.array([1, m1]),P1 , k1, k2)
x_m2_1P = line_dir_pt(np.array([1, m2_1]), P, k1, k2)
x_m2_2P = line_dir_pt(np.array([1, m2_2]), P, k1, k2)

# Plotting the lines
plt.plot(x_m1P[0, :], x_m1P[1, :], label='Line 1: x-2y=3')
plt.plot(x_m2_1P[0, :], x_m2_1P[1, :], label='Line 2 (Case 1): 3x-y=7')
plt.plot(x_m2_2P[0, :], x_m2_2P[1, :], label='Line 2 (Case 2): x+3y = 9')

# Labeling the coordinates
tri_coords = np.vstack((P,)).T
plt.scatter(tri_coords[0, :], tri_coords[1, :])
vert_labels = ['P']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, (tri_coords[0, i], tri_coords[1, i]), textcoords="offset points", xytext=(0, 10), ha='center')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.grid(True)
plt.title('Line Equations')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
