import numpy as np
from numpy import linalg as la
from numpy.linalg import norm

def my_func():
  global a,b,c,m1,m2,theta
a=int(input("value of a :"))
b=int(input("value of b :"))
c=int(input("value of c :"))


m1=np.array([a,b,c])
m2=np.array([b-c,c-a,a-b])
R=(m1@m2)/(norm(m1)*norm(m2))
theta = (np.arccos(R))*(180/np.pi)
print("The value of theta is ",theta," degree")

