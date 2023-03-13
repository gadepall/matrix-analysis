import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sympy as sym
def circ_gen(O,r):
 len = 50
 theta = np.linspace(0,2*np.pi,len)
 x_circ = np.zeros((2,len))
 x_circ[0,:] = r*np.cos(theta)
 x_circ[1,:] = r*np.sin(theta)
 x_circ = (x_circ.T + O).T
 return x_circ 
c=np.array([3,2])
r=np.sqrt(3**2-2**2)
#equation of circle
# circle parameters
u=-c
f=u@(u.T)-r**2
print(f)
x=sym.Symbol('x')
y=sym.Symbol('y')
X=np.array([x,y])
print("The equation of circle is : ")
print(" ")
E=X@(X.T)+2*(u@(X.T))+f
print("{} = 0".format(E))
 


##Generating the circle
x_circ= circ_gen(c,r)

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
#Labeling the coordinates
tri_coords = np.vstack((c))
plt.scatter(tri_coords[0],tri_coords[1])

vert_labels = ['c']
for i, txt in enumerate(vert_labels):
    label = "{}".format(txt) #Form label as A(x,y)
    plt.annotate(label, # this is the text
            (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
#plt.savefig('/sdcard/circle/11.11.1.4/figs/ass.pdf')
plt.show()
