import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
ma=1
mb=1
mc=1
a=sym.Symbol('a')
b=sym.Symbol('b')
c=sym.Symbol('c')
print("According to question : ")
D=(a+b+c)**2
print("{} =0".format(D))
e=ma**2+mb**2+mc**2
print("{}+2(ab+bc+ca) =0".format(e))
#consider ab+bc+ca=x so it becomes a algebraic expression
coeff=[2,3]
r=np.roots(coeff)
print(r)
