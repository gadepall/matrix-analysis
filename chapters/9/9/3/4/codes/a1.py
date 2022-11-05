#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To verify areas of two triangles are equal


#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/mat/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
k1 = 4
k2=6
e1 = np.array([0,1])
e2 = np.array([0,-1])
e3 = np.array([-1,0])
e4 = np.array([1,0])
C = k1*e1
D = k1*e2
O = np.array([0,0])
A =k2*e3
B =k2*e4 
a =(A-B)
b =(C-B)
c =(B-D)

##Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_OD = line_gen(O,D)
x_CD = line_gen(C,D)
x_BD = line_gen(B,D)
x_AD = line_gen(A,D)
areaABC= 0.5*np.linalg.norm(np.cross(b,a))
areaABD= 0.5*np.linalg.norm(np.cross(c,a))
print( areaABC,areaABD)
if(areaABC == areaABD):
  print("areas are equal")
else:
    print("areas are not equal")
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Diameter$')
plt.plot(x_BC[0,:],x_BC[1,:])#,label='$Diameter$')
plt.plot(x_CA[0,:],x_CA[1,:])#,label='$Diameter$')
plt.plot(x_OD[0,:],x_OD[1,:])#,label='$Diameter$')
plt.plot(x_CD[0,:],x_CD[1,:])#,label='$Diameter$')
plt.plot(x_BD[0,:],x_BD[1,:])#,label='$Diameter$')
plt.plot(x_AD[0,:],x_AD[1,:])#,label='$Diameter$')


#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/Download/mat/10/figs/a1.pdf')
subprocess.run(shlex.split("termux-open /sdcard/Download/mat/10/figs/a1.pdf"))
#else
#plt.show()
