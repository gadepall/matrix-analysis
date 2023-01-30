import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
l1=sym.Symbol('l1')
m1=sym.Symbol('m1')
n1=sym.Symbol('n1')
l2=sym.Symbol('l2')
m2=sym.Symbol('m2')
n2=sym.Symbol('n2')
A=np.array([l1,m1,n1])
B=np.array([l2,m2,n2])
res=np.cross(A,B)

print("The vector perpendicular to both the vectors is : {}".format(res))
