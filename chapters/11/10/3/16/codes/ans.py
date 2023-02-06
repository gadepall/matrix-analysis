#If p and q are the lengths of perpendiculars from the origin to the lines x cos θ−y sin θ = k cos 2θ and x sec θ + y cosec θ = k, respectively, prove that p**2 + 4q**2 = k**2

#take theta = pi/6, k = 2
from sympy import *
import numpy as np
from numpy import linalg as LA

omat = np.array([[0,1],[-1,0]])

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

def norm(A):
  return sqrt(A[0]**2+A[1]**2)

theta = np.pi/6
k = 2

x = Symbol('x')
y = Symbol('y')
print('The equation of lines are:', x*cos(theta)-y*sin(theta), '=', k*cos(2*theta), 'and', x/cos(theta)+y/sin(theta), '=', k)

n1 = np.array([cos(theta), -sin(theta)])
n2 = np.array([1/cos(theta), 1/sin(theta)])

A1 = k*cos(2*theta)
A2 = k

P = np.array([0, 0])

print(norm(n1)**2)
print(norm(n2)**2)

print(A2- n2@P)

p = abs(A1-n1@P)/(norm(n1))
q = abs(A2-n2@P)/(norm(n2))

print('p**2 + 4q**2 = ', p**2 + 4*q**2, '= k**2 =', k**2)