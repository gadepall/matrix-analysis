
#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import math as ma
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/matrix/code/CoordGeo')         #path to my scripts

#local imports

from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if

       


#Input parameters
A=  np.array(([0,2]))
B=(2*(ma.pi))/3      
D=np.array(([0,-2]))




#Direction vector
m=np.array(([ma.cos(B),ma.sin(B)]))                                                              
z=np.array(([0,1],[-1,0]))                           
n=z@m                                     








##Generating the line 
k1=-8
k2=3
x_AB = line_dir_pt(m,A,k1,k2)
x_CD = line_dir_pt(m,D,k1,k2)



#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='Line equation')
plt.plot(x_CD[0,:],x_CD[1,:],label='Line equation')



#Labeling the coordinates
tri_coords = np.vstack((A,D)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','D']
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
plt.savefig('/sdcard/matrix/code/fig.pdf')
subprocess.run(shlex.split("termux-open /sdcard/matrix/code/fig.pdf"))
#else
#plt.show()               
