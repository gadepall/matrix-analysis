import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_lines(p1, d1, p2, d2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    t = np.linspace(0, 10, 100)
    x1 = p1[0] + t * d1[0]
    y1 = p1[1] + t * d1[1]
    z1 = p1[2] + t * d1[2]
    x2 = p2[0] + t * d2[0]
    y2 = p2[1] + t * d2[1]
    z2 = p2[2] + t * d2[2]
    ax.plot(x1, y1, z1, label='Line 1')
    ax.plot(x2, y2, z2, label='Line 2')
    ax.legend()
    plt.show()
p1 = [1, 2, 3]
d1 = [-3, -20/7, 2]
p2 = [1, 2, 3]
d2 = [-30/7, 1, -5]
draw_lines(p1, d1, p2, d2)

#verifying the inner product
#u = np.array([-3, -20/7, 2])
#v = np.array([-30/7, 1, -5])

result = np.dot(d1,d2)
print(result)
