#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB

#if using termux
import subprocess
import shlex
#end if

#Input parameters
A = np.array(([4,2]))
B = np.array(([6,5]))
C = np.array(([1,4]))
   
#Formula of an 
D = (B+C)/2

print("D=", D)

E = (A+C)/2

print("E=",E)

F = (A+B)/2

print("F=",F)

#Array vectors are given
A = np.array(([4, 2]))
B = np.array(([ 6, 5]))
C = np.array(([1, 4]))
D = np.array(([3.5, 4.5]))
E = np.array(([2.5, 3]))
F = np.array(([5, 3.5]))
n = 2/1
#Formula of an 
P = (A+n*D)/(1+n)

print("P=", P)

Q = (B+n*E)/(1+n)

print("Q=",Q)

R = (C+n*F)/(1+n)

print("R=",R)

G = (D+E+F)/3
print("G=",G)

#
#Generating all lines
x_AB = line_gen(A,B)
x_BC= line_gen(B,C)
x_CA = line_gen(C,A)
x_AD = line_gen(A,D)
x_BE = line_gen(B,E)
x_CF =line_gen(C,F)
#
#
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB=BC$')
plt.plot(x_BC[0,:],x_BC[1,:])#,label='$Diameter$')
plt.plot(x_CA[0,:],x_CA[1,:])#,label='$Diameter$')
plt.plot(x_AD[0,:],x_AD[1,:])#,label='$Diameter$')
plt.plot(x_BE[0,:],x_BE[1,:])#,label='$Diameter$')
plt.plot(x_CF[0,:],x_CF[1,:])#,label='$Diameter$')

#
#
#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,E,F)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A''(4,2)','B''(6,5)','C''(1,4)','D''($\dfrac{7}{2}$,$\dfrac{9}{2}$)','E''($\dfrac{5}{2}$,3)','F''(5,$\dfrac{7}{2}$)']
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

plt.show()
