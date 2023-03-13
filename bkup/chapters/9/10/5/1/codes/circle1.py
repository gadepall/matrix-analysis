#Python libraries for math and graphics
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys #for path to external scripts
sys.path.insert(0,'/sdcard/matrices/circle/CoordGeo')  #path to my scripts

#local imports
#from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen
from conics.funcs import *

#if using termux
import subprocess
import shlex
#end if




#input parameters
r = 1
P= np.zeros(2)
O=  np.array(([r,0])) #normal vector




alphadeg = 180
alpha= alphadeg*np.pi/180
ac = 2*r*np.sin(alpha/2)
C =  np.array(([ac*np.sin(alpha/2),ac*np.cos(alpha/2)]))

betadeg = 90
beta = betadeg*np.pi/180
ad = 2*r*np.sin(beta/2)
A =  np.array(([ad*np.sin(beta/2),ad*np.cos(beta/2)]))



gammadeg = 300
gamma = gammadeg*np.pi/180
ab = 2*r*np.sin(gamma/2)
D =  np.array(([ab*np.sin(gamma/2),ab*np.cos(gamma/2)]))

phydeg = 150
phy = phydeg*np.pi/180
ad = 2*r*np.sin(phy/2)
B =  np.array(([ad*np.sin(phy/2),ad*np.cos(phy/2)]))


#Proof of the problem
#angle ADC

m1=A-D
m2=C-D

x=(m1.transpose()@m2)/(LA.norm(m1) * LA.norm(m2))

angle=mp.acos(x)*(180/np.pi)

print(angle)




##Generating the line 
I = np.eye(2)
e1 = I[:,0]

m = e1
k1 = -2
k2 = 2
xline = line_dir_pt(m,O,k1,k2)
xCP = line_gen(C,P)
xOB = line_gen(O,B)
xOA = line_gen(O,A)
xCD = line_gen(C,D)
xAD = line_gen(A,D)

##Generating the circle
x_circ= circ_gen(O,r)

#Plotting all lines
plt.plot(xline[0,:],xline[1,:],label='Axis')
plt.plot(xCP[0,:],xCP[1,:],label='Diameter')
plt.plot(xOA[0,:],xOA[1,:],label='Radius')
plt.plot(xOB[0,:],xOB[1,:],label='Radius')
plt.plot(xCD[0,:],xCD[1,:],label='Chord')
plt.plot(xAD[0,:],xAD[1,:],label='Chord')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='Circle')


#Labeling the coordinates
tri_coords = np.vstack((O,A,B,C,D,P)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','A','B','C','D','P']
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
plt.savefig('/sdcard/matrices/circle/CoordGeo/circle1.pdf')
subprocess.run(shlex.split("termux-open /sdcard/matrices/circle/CoordGeo/circle1.pdf"))
#else
#plt.show()
