#Code by GVV Sharma (works on termux)
#March 7, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To draw a circle with a tangent and chord subtending an angle at the centre


#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/dinesh/matrices2/CoordGeo')         #path to my scripts
#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from conics.funcs import *
#if using termux
import subprocess
import shlex
#end if
I =  np.eye(2)
e1 =  I[:,0]
#Input parameters
r = 1
O = np.zeros(2)
thetadeg2 = 100
theta2 = thetadeg2*np.pi/180
aq = 2*r*np.sin(theta2/2)
Q =  np.array(([-np.cos(theta2/2),np.sin(theta2/2)]))
thetadeg1= 10
theta1 = thetadeg1*np.pi/180
ap = 2*r*np.sin(theta1/2)
P =  np.array(([-np.cos(theta1/2),np.sin(theta1/2)]))

thetadeg3 = 30
theta3 = thetadeg3*np.pi/180
ar = 2*r*np.sin(theta3/2)
R =  np.array(([np.cos(theta3/2),np.sin(theta3/2)]))
m1=P-Q
m2=R-Q
x=(m1.transpose()@m2)/(LA.norm(m1) * LA.norm(m2))
angle1=mp.acos(x)*(180/np.pi)
print(angle1)
m3=P-R
m4=P-O
y=(m3.transpose()@m4)/(LA.norm(m3) * LA.norm(m4))
angle2=mp.acos(y)*(180/np.pi)
print(angle2)
##Generating the line 
xPQ = line_gen(P,Q)
xPR = line_gen(P,R)
xPO = line_gen(P,O)
xQR = line_gen(Q,R)
xOR = line_gen(O,R)
##Generating the circle
x_circ= circ_gen(O,r)
#Plotting all lines
plt.plot(xPQ[0,:],xPQ[1,:],label='Chord')
plt.plot(xPR[0,:],xPR[1,:],label='Chord')
plt.plot(xPO[0,:],xPO[1,:],label='Radius')
plt.plot(xQR[0,:],xQR[1,:],label='Chord')
plt.plot(xOR[0,:],xOR[1,:],label='Radius')
#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='Circle')
#Labeling the coordinates
tri_coords = np.vstack((O,P,Q,R)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','P','Q','R']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-5,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/dinesh/matrices2/fig-1.pdf')
subprocess.run(shlex.split("termux-open /sdcard/dinesh/matrices2/fig-1.pdf"))
#else
plt.show()

