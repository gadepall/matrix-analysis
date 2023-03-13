import numpy as np
import matplotlib.pyplot as plt
#Input parameters
A = np.array([2, -1, 4])
m = np.array([1, 2, -1])
lambda1=2
r=A+lambda1*m
print('vector line equation for given points is A+\u03BBm:',r)
#plotting line
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
x, y, z = [A[0], m[0]], [A[1], m[1]], [A[2], m[2]]
ax.scatter(x, y, z, c='red', s=50)
ax.plot(x, y, z, color='blue')
plt.savefig('/sdcard/Download/codes/lines/12.11.2.5/figs/fig.png')
plt.show()
