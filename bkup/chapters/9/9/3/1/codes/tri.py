#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/Download/CoordGeo') 
 

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
r=math.sqrt(20)
theta=math.radians(63.45)
BC=6

A = np.array([r*math.cos(theta),r*math.sin(theta)])
B = np.array([0,0])
C = np.array([BC,0])
D = (B+C)/2
k=2.65
E = np.array([k, 12-4*k])
M = np.array([k,0])
N = np.array([r*math.cos(theta),0])


##Generating all lines
xAB = line_gen(A,B)
xBC = line_gen(B,C)
xCA = line_gen(C,A)
xDA = line_gen(D,A)

xBE = line_gen(B,E)
xCE = line_gen(C,E)



#Plotting all lines
plt.plot(xAB[0,:],xAB[1,:])
plt.plot(xBC[0,:],xBC[1,:])
plt.plot(xCA[0,:],xCA[1,:])
plt.plot(xDA[0,:],xDA[1,:])

plt.plot(xBE[0,:],xBE[1,:])
plt.plot(xCE[0,:],xCE[1,:])



#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,E)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E']
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
plt.show()


plt.savefig('/sdcard/Download/codes/Tri fig 1.png')
plt.savefig('/sdcard/Download/codes/trifig1.pdf')
subprocess.run(shlex.split("termux-open '/sdcard/Download/codes/trifig1.pdf'"))



print("Part - 1")
BD = BC/2
AN = r*math.sin(theta)
Ar_ABD = (1/2)*BD*AN
print("Area of ABD is",Ar_ABD)

CD = BC/2
Ar_ACD = (1/2)*CD*AN
print("Area of ACD is",Ar_ACD)


print("")
print("Part - 2")

EM = 12-4*k
Ar_EBD = (1/2)*BD*EM
print("Area of EBD is",Ar_EBD)

Ar_ECD = (1/2)*CD*EM
print("Area of ECD is",Ar_ECD)

print("")

Ar_ABE = Ar_ABD - Ar_EBD
print("Area of ABE is",Ar_ABE)

Ar_ACE = Ar_ACD - Ar_ECD
print("Area of ACE is",Ar_ACE)

if Ar_ABE == Ar_ACE:
    print("Hence Proved")
else:
    print("Not proved")
