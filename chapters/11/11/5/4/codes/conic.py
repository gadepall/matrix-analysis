import numpy as np
import matplotlib.pyplot as plt


import sys
sys.path.insert(0,'/sdcard/FWCmodule1/conic/code/CoordGeo') #path to my scripts
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if

#Standard ellipse
a = 4
b = 2
x = ellipse_gen(a,b)
#Plotting the ellipse
plt.plot(x[0,:],x[1,:],label='Standard semi-ellipse')


lambda_1=1/16
lambda_2=1/4

#basis vectors
e_1=np.array([1,0]) #along x-axis
e_2=np.array([0,1]) #along y-axis

#eccentricity
e=np.sqrt(1-(lambda_1/lambda_2))
f_0=1

#foci
F_1=(e*np.sqrt(f_0/((lambda_2)*(1-e**2))))*e_1
F_2=(-e*np.sqrt(f_0/((lambda_2)*(1-e**2))))*e_1

#center
c=(F_1+F_2)/2

#vertices
A1=a*e_1
A2 = -A1
B1 =b*e_2



#Points of intersection of a conic section with a line
m=np.array([0,1])
v=np.array([[1/16,0],[0,1/4]])
u=np.array([0,0])
f=-1
#calculating q
d=A1-c
p=np.array([1.5,0])
q=d-p

A,H=inter_pt(m,q,v,u,f)
print("Height =" ,round(H[1],2))

#Generating all lines
x_ma = line_gen(A1,A2)
x_H = line_gen(H,q)
#Plotting all lines
plt.plot(x_ma[0,:],x_ma[1,:],label='$Major Axis$')
plt.plot(x_H[0,:],x_H[1,:],label='$Height$')






#Labeling the coordinates
ellipse_coords = np.vstack((A1,A2,B1,c,H,q)).T
plt.scatter(ellipse_coords[0,:], ellipse_coords[1,:])
vert_labels = ['$A_1$','$A_2$','$B_1$','c','H','q' ]
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (ellipse_coords[0,i], ellipse_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.savefig('/sdcard/FWCmodule1/conic/output.pdf')
subprocess.run(shlex.split("termux-open /sdcard/FWCmodule1/conic/output.pdf"))
#plt.show()
