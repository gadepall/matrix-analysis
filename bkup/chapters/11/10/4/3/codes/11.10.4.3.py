import cmath
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
import math as ma
from numpy import linalg as LA
import math
import numpy as np
from sympy import *  
a = float(input('Enter a: '))  
b = float(input('Enter b: '))  
c = float(input('Enter c: ')) 
  
# calculate the discriminant  
d = (b**2) - (4*a*c)  
  
# find two solutions  
sol1 = (-b-cmath.sqrt(d))/(2*a)  
sol2 = (-b+cmath.sqrt(d))/(2*a)  
print('The solution are {0} and {1}'.format(sol1,sol2)) 
#finding the  two b values
print(sol1.real)#printing only real  value of b
print(sol2.real)
b = Symbol('b')
#here we are finding a values for  two  sets of  b 
def res(b1):
  return a+b1-1;
b1=-2
r1=res(b1)
print(r1)#printing first 'a' value
p1=solve(r1) 
def res(b2):
 return a+b2-1;
b2=3
r2=res(b2)
print(r2)#printing second 'a' value
p2=solve(r2)
e1=np.array(([1,0]))
e2=np.array(([0,1]))
A=e1*sol2.real
B=e2*sol1.real
C=e1*r1
D=e2*r2
#generating a line
def line_gen(A,B):
    len=10
    dim = A.shape[0]
    x_AB = np.zeros((dim,len))
    lam_1 = np.linspace(0,1,len)
    for i in range(len):
        temp1 = A + lam_1[i]*(B-A)
        x_AB[:,i] = temp1.T
    return x_AB
x = np.linspace(-5,5,100)
x_AB=line_gen(A,B)
x_CD=line_gen(C,D)
#Directional vector and normal vector
m1=A-B
print(m1)
m2=C-D
print(m2)
p=np.array(([0,1],[-1,0]))
n1=p@m1
n2=p@m2
print(n1)
print(n2)
#plotting the lines
plt.plot(x_AB[0,:],x_AB[1,:],label='(2 -3)x=-6')
plt.plot(x_CD[0,:],x_CD[1,:],label='(-3 2)x=-6')
sqr_vert = np.vstack((A,B,C,D)).T
plt.scatter(sqr_vert[0,:],sqr_vert[1,:])
vert_labels = ['(3,0)','(0,-2)','(-2,0)','(0,3)']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt,
            (sqr_vert[0,i], sqr_vert[1,i]),
            textcoords = 'offset points',
            xytext = (0,10),
            ha='center')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()                                      
plt.axis('equal')
plt.savefig('/sdcard/Download/codes/lines/11.10.4.3/figs/fig.pdf')
plt.show()
