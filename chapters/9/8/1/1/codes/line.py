#Code by Suresh Srinivas (works on termux)
#To find the angles and sides of the trapezium
#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import sys     #for path to external scripts

def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB


sys.path.insert(0,'/home/CoordGeo')
#local imports

#if using termux
import subprocess
import shlex
#end if

#Input parameters
a=3
b=5
c=9
d=13
x=360/(a+b+c+d)
print("Angle proportional constant =",x)

x=12
AngleA=x*a
AngleB=x*b
AngleC=x*c
AngleD=x*d
e_1 = np.array(([1,0])) #standard basis vector

print("Angle A =",AngleA, "Angle B =", AngleB,"Angle C =", AngleC,"Angle D =", AngleD)
  
#Given points
A = np.array(([-1.55,4.96]))
B = np.array(([ 0.06,1.43]))
C = np.array(([ 0.969,2.70]))
O = np.array(([ 0, 0]))
D = np.array(([-0.14,4.18]))



x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_AC = line_gen(A,C)
x_AD = line_gen(A,D)
x_BD = line_gen(B,D)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$')
plt.plot(x_AC[0,:],x_AC[1,:],label='$AC$')
plt.plot(x_AD[0,:],x_AD[1,:],label='$AD$')
plt.plot(x_BD[0,:],x_BD[1,:],label='$BD$')


#Labeling the coordinates

plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O','D']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x_axis$')
plt.ylabel('$y_axis$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.title('trapezium')
#if using termux
plt.savefig('/home/beeresuresh/Desktop/FWC/Assignments/line-assignment/linefig.pdf')  
plt.show()
