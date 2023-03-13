#Code by Meer Tabres Ali (works on termux)
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
A = np.array(([-2, 2]))
B = np.array(([ 2, 2]))
C = np.array(([ 4, 0]))
D = np.array(([-4, 0]))
O = np.array(([ 0, 0]))
e_1 = np.array(([1,0])) #standard basis vector

print("---------------------")
print("1. Calculation of Angles A, B")
angleBAD =np.arccos(((A-B)@(A-D))/((np.linalg.norm(A-B)) * np.linalg.norm(A-D)))*57.296
print("Angle BAD=", angleBAD)

angleABC =np.arccos(((B-A)@(B-C))/((np.linalg.norm(B-A)) * np.linalg.norm(B-C)))*57.296
print("Angle ABC=",angleABC)
print("Therefore Angle BAD = Angle ABC = 135")

print("---------------------")
print("2. Calculation of Angles C, D")
angleBCD =np.arccos(((C-B)@(C-D))/((np.linalg.norm(C-B)) * np.linalg.norm(C-D)))*57.296
print("Angle BCD=",angleBCD)

angleADC =np.arccos((D-A)@(D-C)/((np.linalg.norm(D-A)) * np.linalg.norm(D-C)))*57.296
print("Angle ADC=",angleADC)
print("Angle ADC = Angle BCD = 45")
print("---------------------")
print("3. Calculation of Diagonals AC, BD")
p = np.linalg.norm(A-C)
P=((A-C)/p)
print("Diogonal AC=",p,"Unit Vector (A-C)=",P)

q = np.linalg.norm(B-D)
Q=((B-D)/q)
print("Diagonal BD",q, "Unit Vector (B-D)=",Q)
print("Therefore Digonal AC is equal to Diagonal BD")  

print("---------------------")
print("4. Comparing Triangles ABC and BAD")
print("Base AC=", p,"Base BD=", q)
print("AC=BD, Bases of Triangles ABC, BAD are equal")
print("Angle BAD=", angleBAD, "AngleABC=", angleABC)
print("angleBAD, angleABC are equal")
print("Bases and Opposite angles of Triangle ABC , Triangle BAD are similar")
print("Therefore Triangle ABC = Triangle BAD")
print("---------------------")
  
  
  
  
  
#Given points
A = np.array(([-2,2]))
B = np.array(([ 2,2]))
C = np.array(([ 4,0]))
O = np.array(([ 0, 0]))
D = np.array(([-4,0]))



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
tri_coords = np.vstack((A,B,C,O,D)).T
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
plt.savefig('/home/administrator/Assignment4/linefig.pdf')  
plt.show()
