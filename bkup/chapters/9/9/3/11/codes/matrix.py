import numpy as np
import matplotlib.pyplot as plt
import sys                                          #for path to external scripts
sys.path.insert(0,'/sdcard/gokul/matrices/lines/codes/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
r1=4
r2=8
r3=6.5
r4=4
theta1 =53* np.pi/180
theta2 =85* np.pi/180
theta3 =120* np.pi/180
B =r2*np.array(([np.cos(theta1),np.sin(theta1)]))
A =r3*np.array(([np.cos(theta2),np.sin(theta2)]))
E =r4*np.array(([np.cos(theta3),np.sin(theta3)]))
D=np.array(([0,0]))
e1=np.array(([1,0]))
C=r1*e1
F=C-A+B
v1=C-A
v2=C-F
ar_t1=0.5*np.linalg.norm((np.cross(v1,v2)))
v3=A-C
v4=A-B
ar_t2=0.5*np.linalg.norm((np.cross(v3,v4)))
#E=np.array(([-5,3]))
#v1=E-A
#v2=E-D
#v3=D-A
#v4=D-C
ar3=0.5*np.linalg.norm((np.cross(A,E)))
ar4=0.5*np.linalg.norm((np.cross(A,C)))
area=ar3+ar4

print("Ar(ACF)=",ar_t1)
print("Ar(ABC)=",ar_t2)
print("Ar(AEDF)=",area+ar_t1)
print("Ar(ABCDE)=",area+ar_t2)


##Generating all lines
x_DC = line_gen(D,C)
x_CB = line_gen(C,B)
x_BA = line_gen(B,A)
x_AE = line_gen(A,E)
x_ED = line_gen(E,D)
x_AC = line_gen(A,C)
x_BF = line_gen(B,F)
x_CF = line_gen(C,F)
x_AF = line_gen(A,F)

#Plotting all lines
plt.plot(x_DC[0,:],x_DC[1,:])
plt.plot(x_CB[0,:],x_CB[1,:])
plt.plot(x_BA[0,:],x_BA[1,:])
plt.plot(x_AE[0,:],x_AE[1,:])
plt.plot(x_ED[0,:],x_ED[1,:])
plt.plot(x_AC[0,:],x_AC[1,:])
plt.plot(x_CF[0,:],x_CF[1,:])
#plt.plot(x_FB[0,:],x_FB[1,:])
plt.plot(x_AF[0,:],x_AF[1,:])
plt.plot(x_BF[0,:],x_BF[1,:])

#Labeling the coordinates
tri_coords = np.vstack((D,C,A,B,E,F)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['D','C','A','B','E','F']
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

plt.savefig('/sdcard/gokul/matrices/lines/images/matrix.pdf')
subprocess.run(shlex.split("termux-open  /sdcard/gokul/matrices/lines/images/matrix.pdf"))

