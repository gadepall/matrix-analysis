import numpy as np
import matplotlib.pyplot as plt
import os

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

#Point
A = np.array([[2.0],[2.0*np.sqrt(3)]])

#Slope
theta = 75.0
m = np.array([[np.tan(np.radians(theta))],[1.0]])

#Generate another point
B = A+m

#Generate the line
L = line_gen(A,B)

#Plot the line
plt.plot(L[0],L[1])
plt.grid()
plt.savefig('../figs/line.png')
os.system('termux-open ../figs/line.png')
