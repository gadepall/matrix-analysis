import numpy as np
from numpy import linalg as la
from numpy.linalg import norm
def my_func():
  global a,b,c
a=np.array([2,3,-1])
b=np.array([1,-2,1])
c=a+b
unit_vec=5*(c/norm(c))
print(unit_vec)

