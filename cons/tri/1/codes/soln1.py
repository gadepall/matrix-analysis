import numpy as np
from numpy import linalg as la
from numpy.linalg import norm
import math
def my_func():
  global a,b,c,k,theta,A,B,C,theta1,AB,AC,BC,m,n
a=7
k=13
theta=(75*np.pi/180)
c=(a**2-k**2)/(2*(k+(2*a*np.cos(theta))))
print(c)
A=c*np.block([np.cos(theta),np.sin(theta)])
print(A)
B=np.block([0,0])
C=np.block([a,0])
AB=c*np.block([np.cos(theta),np.sin(theta)])
BC=np.block([a,0])
AC=np.block([(a-(c*np.cos(theta))),(0-(c*np.sin(theta)))])
print((math.dist(A,B)))
print((math.dist(A,C)))
print((math.dist(B,C)))
m=np.inner(AB,BC)
n=norm(AB)*norm(BC)
theta1=np.arccos(m/n)
print(theta1)
if(theta==theta1):
  print("The values of angles are equal")
else:
  print("The values of angles are not equal")




