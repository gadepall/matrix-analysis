import numpy as np
import matplotlib.pyplot as plt
import os

#Generating points on a hyperbola
def hyper_gen(a,b):
	len = 10000
	theta = np.linspace(0,np.pi/3,len)
	x_hyper = np.zeros((2,len))
	x_hyper[0,:] = a/np.cos(theta)
	x_hyper[1,:] = b*np.tan(theta)
	return x_hyper

a = np.sqrt(49)
b = np.sqrt(343/9)
H = hyper_gen(a,b)
P1 = np.array([[-7.0],[0.0]])
P2 = np.array([[7.0],[0.0]])
plt.plot(H[0], H[1],'b')
plt.plot(H[0], -H[1],'b')
plt.plot(-H[0], H[1],'b')
plt.plot(-H[0], -H[1],'b')
plt.plot(P1[0],P1[1],'k.')
plt.plot(P2[0],P2[1],'k.')
plt.text(P1[0]+1e-1,P1[1],'P$_1$')
plt.text(P2[0]+1e-1,P2[1],'P$_2$')
plt.grid()
plt.tight_layout()
plt.savefig('../figs/hyperbola.png')
os.system('termux-open ../figs/hyperbola.png')
