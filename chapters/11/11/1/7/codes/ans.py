#Find the centre and radius of the circle x^2 +y^2 − 4x − 8y − 45 = 0.

import numpy as np
import matplotlib.pyplot as plt

u = np.array([-2, -4])
f = -45

center = -u
radius = np.sqrt(np.linalg.norm(u)**2 - f)

#Generating all the points on the circle
len = 100
theta = np.linspace(0,2*np.pi,len)
x_circ = np.zeros((2,len))
x_circ[0,:] = radius*np.cos(theta)
x_circ[1,:] = radius*np.sin(theta)
x_circ = (x_circ.T + center).T

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$circle$')
plt.plot(center[0],center[1],'o')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/lokesh/EE2802/EE2802-Machine_learning/11.11.1.7/figs/circle.png')
