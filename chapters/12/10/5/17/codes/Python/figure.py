import numpy as np 
import matplotlib.pyplot as plt 
from numpy import linalg as LA 
import math

a = (2*np.pi)/3
b = (np.pi)/3
O = np.array(([0, 0])) 
A = np.array(([1, 0])) 
B = np.array(([np.cos(a), np.sin(a)])) 
C = np.array(([np.cos(b), np.sin(b)]))

def line_gen(A,B):
   len =2
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB

#Generating all lines
x_OA = line_gen(O,A)
x_OB = line_gen(O,B)
x_OC = line_gen(O,C)

#Plotting all lines
plt.plot(x_OA[0,:],x_OA[1,:],label='a')
plt.plot(x_OB[0,:],x_OB[1,:],label='b')
plt.plot(x_OC[0,:],x_OC[1,:],label='c')
plt.text(0.5, -0.05, 'a')
plt.text(-0.35, 0.45, 'b')
plt.text(0.25, 0.37, 'c')
#Labeling the coordinates
tri_coords = np.vstack((O,A,B,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])

plt.xlabel('$X-Axis$')
plt.ylabel('$Y-axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.title('Sum of two unit vectors',size=15)
#if using termux
plt.savefig('/sdcard/Download/Assignment/Vector-algebra/12.10.5.17/codes/Python/figs/fig')
#plt.show()
