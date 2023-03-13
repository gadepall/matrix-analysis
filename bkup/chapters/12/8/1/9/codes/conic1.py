import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from numpy import linalg as LA
from pylab import *

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/ganga/matrix/CoordGeo') 


#local imports
#local imports
from line.funcs import *
from triangle.funcs import *
#from conics.funcs import circ_gen
from conics.funcs import *
from sympy import *

#if using termux
import subprocess
import shlex
#end if

#x,y=symbols('x y')
#def affine_transform(P,c,x):
 #   return P@x + c
def fun1(x):
    y=x-x**2
    return y
def fun2(x):
    y=-x-x**2
    return y
#Input parameters

O=np.array(([0,0]))

#for parabola
V = np.array([[1,0],[0,0]])
#for line
V_1=np.array([[0,0],[0,0]]) 
V_2=np.array([[0,0],[0,0]]) 
 
u = np.array(([0,-1/2]))
u_1 = np.array(([-1/2,1/2]))
u_2 = np.array(([1/2,1/2]))
f = 0

#for computing stright line
lamda,P = LA.eigh(V)
print(P)
print("lamda is",lamda)
if(lamda[1] == 0):  # If eigen value negative, present at start of lamda 
    lamda = np.flip(lamda) # e value 
    P = np.flip(P,axis=1)   #e vectors in col


eta = u@P[:,0]
a = np.vstack((u.T + eta*P[:,0].T, V))
b = np.hstack((-f, eta*P[:,0]-u)) 
#center = LA.lstsq(a,b,rcond=None)[0]
 
n = np.sqrt(lamda[1])*P[:,0]
c = 0.5*(LA.norm(u)**2 - lamda[1]*f)/(u.T@n)
F = np.array(([0,1]))
fl = LA.norm(F)


#Finding k values for Points A and B
#m = np.array([1,0]) #direction vector
m=omat@n
d =((m.T@(V@F + u))*2 - (F.T@V@F + 2*u.T@F + f)*(m.T@V@m))
k1 = (d - m.T@(V@F + u))/(m.T@V@m)
k2 = (-d - m.T@(V@F + u))/(m.T@V@m)
print("k1",k1)
print("k2",k2)
A = F + k1*m
B = F + k2*m
print("m",m)
print("intersection of Point A " ,A)
print("intersection of Point B" ,B)


#Generating all shapes
x=np.linspace(-1,1,50)
y=x**2
y1=abs(x)
plt.plot(x,y,label='parabola')
plt.plot(x,y1)
plt.xlim([-2, 2])
#plt.plot(x[:,0],x[:,1],label='parabola')


# finding area
I1 ,er1= quad(fun1,0,1)
I2,er2=quad(fun2,-1,0)
area= I1+I2
print("area=",area)




#Shading the region between parabola and line
fill_between(x, y1,y,color='blue', alpha=.2)
#fill_between(-x1,0,y1,color ='blue',alpha=.2)
#plt.fill_between(x,0,y,color='red',alpha=.2)
       
#Labeling the coordinates
plot_coords = np.vstack((A,B,O)).T
plt.scatter(plot_coords[0,:], plot_coords[1,:])
vert_labels = ['A','B','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (plot_coords[0,i], plot_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/home/ganga/matrix/figs/con3.pdf')  
#subprocess.run(shlex.split("termux-open ")))
#else
plt.show()
