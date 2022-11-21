#Code by GVV Sharma (works on termux)
#February 12, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To find the centre of a circle given the end points of a diameter


#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/user/Documents/CoordGeo')         #path to my scripts
from sympy import*
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if
h,k,f = symbols('h,k,f')
I = np.array([[1,0],[0,1]])
V = I
x1 = np.array(([2,3]))
x2 = np.array(([-1,1]))
U = np.array(([h,k]))
f1 = x1.T@V@x1+2*U.T@(x1)+f
f2 = x2.T@V@x2+2*U.T@(x2)+f
f3 = [1,-3]@U-11
A = np.array([[2,3,1],[-2,2,1],[1,-3,0]])
B = np.array([[-13,-2,-11]]).reshape(3,1)
[h,k,f]= LA.inv(A)@B
u1=np.array(([h,k]))
print(u1)
#Centre and radius
c = -LA.inv(V)@u1
print(c)
r = np.sqrt(65/4)

##Generating all lines
#x_AB = line_gen(A,B)
#x_Xy = line_gen(X,y)
y = np.linspace(-6,3,50)
x = (3*y+11)
plt.plot(x,y)
##Generating the circle
x_circ= circ_gen(c.T,r)

#Plotting all lines
#plt.plot(x_AB[0,:],x_AB[1,:],label='$chord$')
#plt.plot(x_Xy[0,:],x_Xy[1,:],label='$line joining$')
#plt.plot(x_cB[0,:],x_cB[1,:],label='$line joining$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')


#Labeling the coordinates
#tri_coords = np.vstack((p,c,A,B,X,y)).T
tri_coords= c
#print(tri_coords,tri_coords1)
#print(tri_coords[0,0])
#plt.scatter(tri_coords[0,:],tri_coords[1,:])
plt.scatter(tri_coords[0],tri_coords[1])

#vert_labels = ['p','c','B','A','X','y']
#for i, txt in enumerate(vert_labels):
#    plt.annotate(txt, # this is the text
#            (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
#                 textcoords="offset points", # how to position the text
#                xytext=(0,10), # distance from text to points (x,y)
#                ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/home/user/Documents/Assignments/assg_6/fig.pdf')
subprocess.run(shlex.split("termux-open /home/user/Documents/assg_6/fig.pdf"))
#else
plt.show()




