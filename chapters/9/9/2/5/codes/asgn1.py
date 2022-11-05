#Code by ANUSHA JELLA(works on termux)
#September 14, 2022
#License
#https://www.gnu.org/licenses/gpl-3.0.en.html
#To find the fourth vertex and the diagonal given three vertices of a rectangle


#Python libraries for math and graphics
import numpy as np
#import random
import math
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0,'/home/anu/anusha1/CoordGeo')         #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

#Input parameters
a=5
b=4
theta1=np.pi*4/9
e1=np.array([1,0])
S=np.array(([0,0]))
#print(Rand(0,3,1))
P=a*np.array([np.cos(theta1),np.sin(theta1)])
R =b*e1+S
#print(P,R)
#P = np.array(([0.5,0]))
#R=np.array(([4,-3]))
Q = P+R-S
#print(C)
k1=1.5
print(k1)
A=(k1*P+Q)/(k1+1)
#B=np.array(([5.5,0]))
B=A+R-S
e_1 = np.array(([1,1]))
#Length of the sides
d_sp = LA.norm(S-P)
d_rq=LA.norm(R-Q)
#print(d_sp,d_rq)
d_pq=LA.norm(P-Q)
d_sr=LA.norm(S-R)
#print(d_pq,d_sr)

d_sa = LA.norm(S-A)
d_rb=LA.norm(R-B)
#print(d_sa,d_rb)
d_ab=LA.norm(A-B)
d_sr=LA.norm(S-R)
#print(d_ab,d_sr)

S1=S-R
P1=S-P
#v1 = S1 / np. linalg. norm(S1)
#v2 = P1 / np. linalg. norm(P1)
#dot_product = np. dot(v1, v2)
v1=S1@P1
v2=np.linalg.norm(S1)*np.linalg.norm(P1)
angle = np.arccos((v1/v2))    # angle between SR,SP
R1=R-S
Q1=R-Q
v11=R1@Q1
v12=np.linalg.norm(Q1)*np.linalg.norm(R1)
angle1=np.arccos((v11/v12))   # angle between SR andSA

#print("theta1",math.degrees(angle),math.degrees(angle1))
ar1=np.linalg.norm(np.cross(S1,R1))  #Area of PQRS
S2=S-A
R2=R-B
v22=S2@S1
ar2=np.linalg.norm(np.cross(S1,S2)) # area of ABRS
v23=np.linalg.norm(S1)*np.linalg.norm(S2)
v32=R2@R1
v33=np.linalg.norm(R2)*np.linalg.norm(R1)
angle2=np.arccos((v22/v23))
angle3=np.arccos((v32/v33))


## x point on BR line
k=1.5
x =(k*B+R)/(k+1) 
#x=(B+R)/2
Y=S+x-A
# angle between AX nad SX
Ax1=x-A
Sx1=x-S
ar3=0.5*np.linalg.norm(np.cross(Ax1,Sx1)) #area of AXS

print(ar1,ar2,ar3)

##Generating all lines
x_SP = line_gen(S,P)
x_PQ = line_gen(P,Q)
x_QR = line_gen(Q,R)
x_RS = line_gen(R,S)
x_SA = line_gen(S,A)
x_RB=line_gen(R,B)
x_AB=line_gen(A,B)
x_Ax=line_gen(A,x)
x_Sx=line_gen(S,x)
x_Ry=line_gen(R,Y)
x_sy=line_gen(S,Y)

#Plotting all lines
plt.plot(x_SP[0,:],x_SP[1,:])#,label='$Diameter$')
plt.plot(x_PQ[0,:],x_PQ[1,:])#,label='$Diameter$')
plt.plot(x_QR[0,:],x_QR[1,:])#,label='$Diameter$')
plt.plot(x_RS[0,:],x_RS[1,:])#,label='$Diameter$')
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Diameter$')
plt.plot(x_SA[0,:],x_SA[1,:])
plt.plot(x_RB[0,:],x_RB[1,:])
plt.plot(x_Ax[0,:],x_Ax[1,:])
plt.plot(x_Sx[0,:],x_Sx[1,:])
#plt.plot(x_Ry[0,:],x_Ry[1,:],'.')
#plt.plot(x_sy[0,:],x_sy[1,:],'.')
#Labeling the coordinates
tri_coords = np.vstack((S,Q,P,R,A,B,x)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['S','Q','P','R','A','B','X']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

#plt.xlabel('$x$')
#plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/home/anu/anusha1/python1/rect2.pdf')
plt.show()
#subprocess.run(shlex.split("termux-open  /storage/emulated/0/github/cbse-papers/2020/math/10/solutions/figs/matrix-10-6.pdf"))
#else

