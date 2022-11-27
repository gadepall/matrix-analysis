#To find the area of the region

#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
from pylab import *

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/CoordGeo/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if


I =  np.eye(2)
e1 =  I[:,0]
#Input parameters

#line parameters
n =  np.array(([0.577350269,-1])) #normal vector
m =  omat@n #direction vector
c = 0
q = c/(n@e1)*e1 #x-intercept
print(q)

#circle parameters
V = I
u = np.zeros(2)
r = np.sqrt(4)
f = -r**2
O = -u #Centre

#Points of intersection of line with circle
x,x1 = inter_pt(m,q,V,u,f)
print(x,x1)
#p=np.array(([2,0]))

#Generating the line 
k1 = -1.732
k2 =1.732
xline = line_dir_pt(m,q,k1,k2)

##Generating the circle
x_circ= circ_gen(O,r)

#Plotting all lines
plt.plot(xline[0,:],xline[1,:],label='$Line$')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')


#Labeling the coordinates
tri_coords = np.vstack((O,x,x1)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','$x$','$x_1$']
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

 #use set_position
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')

#if using termux
plt.savefig('/sdcard/Download/fwc/conic-assignment/conics-fig.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Download/fwc/conic-assignment/conics-fig.pdf"))
#else
#plt.show()







