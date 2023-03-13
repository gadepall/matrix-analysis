#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/github/cbse-papers/CoordGeo')         #path to my scripts
#sys.path.insert(0,'/sdcard/Download/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
AB=16
AE=8
CF=10
CD=AB

m = math.sqrt((CF*CF)/(AB*AB-CF*CF))


D = np.array([0,0])
C = np.array([CD,0])
A = np.array([AE/m,AE])
B = np.array([(AE/m) + AB ,AE])
E = np.array([AE/m,0])
F = np.array([CF/(2*m),CF/2])



##Generating all lines
xAB = line_gen(A,B)
xBC = line_gen(B,C)
xCD = line_gen(C,D)
xDA = line_gen(D,A)

xCF = line_gen(C,F)
xAE = line_gen(A,E)



#Plotting all lines
plt.plot(xAB[0,:],xAB[1,:])
plt.plot(xBC[0,:],xBC[1,:])
plt.plot(xCD[0,:],xCD[1,:])
plt.plot(xDA[0,:],xDA[1,:])

plt.plot(xCF[0,:],xCF[1,:])
plt.plot(xAE[0,:],xAE[1,:])



#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,E,F)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F']
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
plt.text(15, 8.25, '16 cm', fontsize = 11, color='Red')
plt.text(11, 1.5, '10 cm', fontsize = 11, color='Red', rotation=-30)
plt.text(10.25, 4.5, '8 cm', fontsize = 11, color='Red',rotation=90)

#plt.savefig('fig 1.jpeg')
plt.savefig('/sdcard/github/matrix-analysis/chapters/9/8/2/6/codes/fig1.pdf')
subprocess.run(shlex.split("termux-open '/sdcard/github/matrix-analysis/chapters/9/8/2/6/codes/fig1.pdf'"))

print("Proof: \n")


Area = CD*AE
AD = Area/CF

print("CD =",CD)
print("AE =",AE)
print("Area = CD X AE")
print("Area =",CD,"X",AE)
print("Area =",Area)

print("")

print("Area = AD X CF")
print(Area,"= AD X",CF)
print("AD =",Area,"/",CF)
print("AD =",AD,"cm")
