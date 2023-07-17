import numpy as np
from numpy import linalg as la
from numpy.linalg import norm

def my_func():
  global A,B,C,D,m1,m2
A=np.array([4,7,8])
B=np.array([2,3,4])
C=np.array([-1,-2,1])
D=np.array([1,2,5])

m1=B-A
m2=D-C
print(m1)
print(m2) 
R=(m1@m2)/(norm(m1)*norm(m2))
theta=np.pi - np.arccos(R)
print(theta)
if (theta==0):
  print("The two lines having direction vector m1 and m2 are parallel")
else:
  print("The two lines having direction vector m1 and m2 are not parallel")

