import numpy as np
import matplotlib.pyplot as plt
import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/assignment4/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
a=4.5
b=4.5
c=10
d=2.5
theta1 = 25* np.pi/180
theta2 = 120* np.pi/180
theta3 =  np.pi/3
theta4 = 35* np.pi/180
A =a*np.array(([np.cos(theta3),np.sin(theta3)]))
B =c*np.array(([np.cos(theta1),np.sin(theta1)]))
D =d*np.array(([np.cos(theta2),np.sin(theta2)]))
E=np.array(([0,0]))
e1=np.array(([2,0]))
F=a*e1
B=(a*e1)+A
C=b*np.array(([np.cos(theta4),np.sin(theta4)]))
C=(b*e1)+D
V1=A-D
V2=D-E
V3=B-C
V4=C-F
#ar(ADE)=ar(BCF)
V5=(0.5)*np.linalg.norm(np.cross(V1,V2))
print(V5)
V6=(0.5)*np.linalg.norm(np.cross(V3,V4))
print(V6)
##Generating all lines
x_DA = line_gen(D,A)
x_DE = line_gen(D,E)
x_DC = line_gen(D,C)
x_AB = line_gen(A,B)
x_CB = line_gen(C,B)
x_CF = line_gen(C,F)
x_BF = line_gen(B,F)
x_EF = line_gen(E,F)
x_AE = line_gen(A,E)

#Plotting all lines
plt.plot(x_DA[0,:],x_DA[1,:])
plt.plot(x_DE[0,:],x_DE[1,:])
plt.plot(x_DC[0,:],x_DC[1,:])
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_CB[0,:],x_CB[1,:])
plt.plot(x_CF[0,:],x_CF[1,:])
plt.plot(x_BF[0,:],x_BF[1,:])
plt.plot(x_EF[0,:],x_EF[1,:])
plt.plot(x_AE[0,:],x_AE[1,:])

#Labeling the coordinates
tri_coords = np.vstack((D,A,E,C,B,F)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['D','A','E','C','B','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/Download/FWC/trunk/matrix/ABC.pdf')
subprocess.run(shlex.split("termux-open  /sdcard/Download/FWC/trunk/matrix/ABC.pdf"))
#else
plt.show()
