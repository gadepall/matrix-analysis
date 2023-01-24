#find the equation of the line which satisfy the given conditions: Intersecting the y-axis at a distance of 2 units above the origin and making an angle of pi/6 with positive direction of the x-axis.

import numpy as np
import matplotlib.pyplot as plt

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

omat = np.array([[0, -1], [1, 0]])

m = np.array([1, np.tan(np.pi/6)])
n = omat@m

A = np.array([0, 2])
c = n@A
print(n,c)
print(n[0],'*','x','+',n[1],'*','y','=',(n[0]*A[0]+n[1]*A[1]),sep="")

x_AB = line_gen(A,A+m)


plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(A[0],A[1],'k.')
plt.text(A[0],A[1], "A")
plt.grid()
plt.tight_layout()
plt.axis('equal')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('/home/lokesh/EE2802/EE2802-Machine_learning/11.10.2.6/figs/line.png')
