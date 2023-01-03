import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sys
lib_path = '/sdcard/IITH/matrix/CoordGeo'
sys.path.insert(0,lib_path)

#local import
from line.funcs import *
from triangle.funcs import *

#if using termux
import subprocess
import shlex
#end if

# Coordinates of A and B
A = np.array([0, 0])
B = np.array([36, 15])
# Calculate distance
d = np.linalg.norm(A - B)
print(d)

#Point A and B
I = np.array([0,0]) 
J = np.array([36,15])   
K = np.array([36,0])

#Generating all lines
x_IJ = line_gen(I,J)
x_JK = line_gen(J,K)
x_KI = line_gen(K,I)

#Plotting all lines
plt.plot(x_IJ[0,:],x_IJ[1,:],color='orange',label='$AB=39$')
plt.plot(x_JK[0,:],x_JK[1,:],color='gray',linestyle="--")
plt.plot(x_KI[0,:],x_KI[1,:],color='gray',linestyle="--")
plt.plot(I[0], I[1], 'o')
plt.text(I[0] * (1 + 0.1), I[1] * (1 - 0.1) , 'A')
plt.plot(J[0], J[1], 'o')
plt.text(J[0] * (1 - 0.1), J[1] * (1) , 'B')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux                        
plt.savefig('/sdcard/IITH/vector/vec.pdf')                                           
subprocess.run(shlex.split("termux-open /sdcard/IITH/vector/vec.pdf")) 
#plt.show()
