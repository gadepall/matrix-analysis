#Code by GVV Sharma (works on termux)
#March 1, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To verify the given points are collinar


#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/ramesh/maths/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
A =np.array(([3,0]))
B =np.array(([-2,-2]))
C =np.array(([8,2]))

#parameters  of transpose vectors
D =A-B
E =A-C
print("B transpose:")
print(B.T)
print("C transpose:")
print(C.transpose())
print("matrix of transpose")
F  =np.array(([D,E]))
print(F)
print("rank of matrix")
print(np.linalg.matrix_rank(F))

if(np.linalg.matrix_rank(F) == 1):
    print("points are collinear")
else:
    print("points are non collinear")

##Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)



#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Diameter$')
plt.plot(x_BC[0,:],x_BC[1,:])#,label='$Diameter$')



#Labeling the coordinates
tri_coords = np.vstack((A,B,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C']
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
plt.savefig('/sdcard/ramesh/maths/figs6.pdf')
subprocess.run(shlex.split("termux-open /sdcard/ramesh/maths/figs6.pdf"))
#else
#plt.show()






