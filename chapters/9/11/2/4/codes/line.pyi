#Code by GVV Sharma (works on termux)
#March 1, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To draw a quadrilateral circumscribing a circle


#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/ramesh/fwc2022/python/10/codes/CoordGeo')         #path to my scripts

#local imports
from line.funcs import* 
#from triangle.funcs import 
#from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

theta = np.pi/6
r=11/(1+np.cos(theta)+np.sin(theta))
X = r*np.array(([np.cos(theta),np.sin(theta)]))
Z = r*np.array(([np.cos(theta),0]))
Y = np.array(([0,0])) 
print("X=",X)
print("Y",Y)
print("Z",Z)




#Z = np.array(([rcos(theta),0]))

#r=11/(1+np.cos(theta)+np.sin(theta))


##Generating all lines
x_XY = line_gen(X,Y)
x_XZ = line_gen(X,Z)
x_YZ = line_gen(Y,Z)



#Plotting all lines
plt.plot(x_XY[0,:],x_XY[1,:])#,label='$Diameter$')
plt.plot(x_XZ[0,:],x_XZ[1,:])#,label='$Diameter$')
plt.plot(x_YZ[0,:],x_YZ[1,:])#,label='$Diameter$')






#Labeling the coordinates
tri_coords = np.vstack((X,Y,Z)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['X','Y','Z']
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
#plt.savefig('/home/ramesh/fwc2022/python/10/codes/CoordGeo/figs11.pdf')
#subprocess.run(shlex.split("termux-open  /home/ramesh/fwc2022/python/10/codes/CoordGeo/figs11.pdf"))
#else
plt.show()
