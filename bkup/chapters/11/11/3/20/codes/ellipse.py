import numpy as np
import matplotlib.pyplot as plt
import os

#Generating points on an ellipse
def ellipse_gen(a,b):
	len = 10000
	theta = np.linspace(0,2*np.pi,len)
	x_ellipse = np.zeros((2,len))
	x_ellipse[0,:] = a*np.cos(theta)
	x_ellipse[1,:] = b*np.sin(theta)
	return x_ellipse

a = np.sqrt(52)
b = np.sqrt(13)
E = ellipse_gen(a,b)
plt.plot(E[0], E[1])
plt.grid()
plt.tight_layout()
plt.savefig('../figs/ellipse.png')
os.system('termux-open ../figs/ellipse.png')
