#Code by Anusha Jella(works on termux)
#Sep 26, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To find the point on the x axis which is equidistant from 
#two given points

#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0, '/sdcard/Download/anusha1/CoordGeo')        #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from sympy import *

#if using termux
import subprocess
import shlex
#end if 
  
x, y = symbols('x y')
def f(x):
    return 1/(x-1)
#expr = 1/(x-1)
print("Expression : {} ".format(f(x)))
   

def f1(x,y,q,A,u,f):
    s=np.array([[x],[y]])
    return (A@q+u).T@s + u.T@q +f
#Use sympy.Derivative() method 
expr_diff = Derivative(f(x), x)  
eq1=expr_diff.doit()      
print("Value of the derivative : {} ".format(eq1))

#x_val=(solve(eq1,-1))
x_val=np.array(solve(x**2-2*x))
print(x_val)
y_val=np.array(f(x_val))
print(y_val)
A=np.vstack([x_val,y_val]).T
print("eq:",f(2))


e1 = np.array((1,0)).reshape(2,1)
e2 = np.array((0,1)).reshape(2,1)

#Input parameters
V = np.array([[0,0.5],[0.5,0]])
u = np.array((0,-0.5)).reshape(2,1)
f =-1

#Intermediate parameters
f0 = np.abs(f+u.T@LA.inv(V)@u)

#Eigenvalues and eigenvectors
lam1 = -1

#Normal vectors to the conic
n1 = np.array((-lam1,1))

#kappa
den1 = n1.T@LA.inv(V)@n1

k1 = np.sqrt(f0/(den1))

q11 = LA.inv(V)@((k1*n1-u.T).T)
q12 = LA.inv(V)@((-k1*n1-u.T).T)

y11=f1(x,y,q11,V,u,f)
y12=f1(x,y,q12,V,u,f)
print(y11,"=0",y12,"=0")

x1 = np.linspace(-5,5,100)
y1=1/(x1-1) 
y21=3-x1
y22=-1-x1

# Create the plot
plt.plot(x1,y1,label='y =(x+1)^-1 ')
plt.plot(x1,y21,label='y=3-x')
plt.plot(x1,y22,label='y=-1-x')
plt.scatter(q11[0],q11[1])
plt.annotate('q11',(q11[0],q11[1]))
plt.scatter(q12[0],q12[1])
plt.annotate('q12',(q12[0],q12[1]))
# Add a title
plt.title('')

# Add X and y Label
plt.xlabel('x axis')
plt.ylabel('y axis')

# Add a grid
plt.grid(alpha=1,linestyle='--')

# Add a Legend
plt.legend()
plt.savefig('/sdcard/Download/anusha1/python1/conic1.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Download/anusha1/python1/conic1.pdf"))
# Show the plot
plt.show()
