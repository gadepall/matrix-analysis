#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/surabhi/Downloads/surabhi22/test/codes/CoordGeo')         #path to my scripts


#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
r1 = 4
r2 = 3       #radius
d = 8      #distance of points from center
O = np.array([0,0])   #center

e1 = np.array(([1,0]))

C = d*e1

theta = np.arccos(r1/d)

alpha = np.arcsin(r2/d)

t = d*np.cos(alpha)

P1 = r1*np.array([np.cos(theta),np.sin(theta)])
P2 = r1*np.array([np.cos(theta),-np.sin(theta)])

Q1 = t*np.array([np.cos(alpha),np.sin(alpha)])
Q2 = t*np.array([np.cos(alpha),-np.sin(alpha)])

print('point of contact p1 = ',P1)
print('point of contact p2 = ',P2)
print('point of contact Q1 = ',Q1)
print('point of contact Q2 = ',Q2)

##Generating all lines
xAB = line_gen(O,C)

xPP1 = line_gen(C,P1)
xPP2 = line_gen(C,P2)

xQQ1 = line_gen(O,Q1)
xQQ2 = line_gen(O,Q2)

##Generating the circle
x1_circ= circ_gen(O,r1)
x2_circ= circ_gen(C,r2)

#Plotting all lines
plt.plot(xAB[0,:],xAB[1,:])

plt.plot(xPP1[0,:],xPP1[1,:],label='$Tangent1$')
plt.plot(xPP2[0,:],xPP2[1,:],label='$Tangent2$')

plt.plot(xQQ1[0,:],xQQ1[1,:],label='$Tangent3$')
plt.plot(xQQ2[0,:],xQQ2[1,:],label='$Tangent4$')

#Plotting the circle
plt.plot(x1_circ[0,:],x1_circ[1,:],label='$Circle$')
plt.plot(x2_circ[0,:],x2_circ[1,:],label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((O,P1,P2,Q1,Q2,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','P1','P2','Q1','Q2','C']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-5,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x-axis$')
plt.ylabel('$y-axis$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')


#plt.savefig('/sdcard/Download/circlefig.pdf')
#subprocess.run(shlex.split("termux-open '/sdcard/Download/circlefig.pdf'"))

plt.show()
