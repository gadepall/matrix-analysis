#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/FWC/Matrices/Circle/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
r = 3       #radius
d = 7       #distance of points from center
O = np.array([0,0])   #center


e1 = np.array(([1,0]))
P = -d*e1
Q = d*e1
theta = np.arccos(r/d)
A = -r*e1
B = r*e1
alpha = 3.1415-(theta)
P1 = r*np.array([np.cos(alpha),np.sin(alpha)])
P2 = r*np.array([np.cos(alpha),-np.sin(alpha)])

Q1 = r*np.array([np.cos(theta),np.sin(theta)])
Q2 = r*np.array([np.cos(theta),-np.sin(theta)])

print('point of contact p1 = ',P1)
print('point of contact p2 = ',P2)
print('point of contact Q1 = ',Q1)
print('point of contact Q2 = ',Q2)

##Generating all lines
xAB = line_gen(A,B)
xPA = line_gen(P,A)
xBQ = line_gen(B,Q)

xPP1 = line_gen(P,P1)
xPP2 = line_gen(P,P2)

xQQ1 = line_gen(Q,Q1)
xQQ2 = line_gen(Q,Q2)

##Generating the circle
x_circ= circ_gen(O,r)

#Plotting all lines
plt.plot(xAB[0,:],xAB[1,:],label='$Diameter$')
plt.plot(xPA[0,:],xPA[1,:])
plt.plot(xBQ[0,:],xBQ[1,:])

plt.plot(xPP1[0,:],xPP1[1,:],label='$Tangent1$')
plt.plot(xPP2[0,:],xPP2[1,:],label='$Tangent2$')

plt.plot(xQQ1[0,:],xQQ1[1,:],label='$Tangent3$')
plt.plot(xQQ2[0,:],xQQ2[1,:],label='$Tangent4$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((P,P1,A,P2,O,Q1,B,Q2,Q)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['P','P1','A','P2','O','Q1','B','Q2','Q']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-5,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/FWC/matrices/Circle/circlep.pdf')
#subprocess.run(shlex.split("termux-open '/sdcard/FWC/Matrices/Circle/circlep.pdf'"))
#else
#plt.show()
