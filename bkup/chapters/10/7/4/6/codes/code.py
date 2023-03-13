import numpy as np 
import matplotlib.pyplot as plt 
from numpy import linalg as LA 
import math
from fractions import Fraction 

  
#Two aray vectors are given  
n=Fraction(1,3)
A = np.array(([4, 6])) 
B = np.array(([1, 5])) 
C = np.array(([7, 2])) 
 
D = (A+B*n)/(n+1)
E=  (A+C*n)/(n+1)


print ('Intersection point of side AB',D,'Intersection point of side AC',E)

def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB
#
#Generating all lines
x_AB = line_gen(A,B)
x_BC= line_gen(B,C)
x_AC = line_gen(A,C)
x_DE= line_gen(D,E)

#
#
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_AC[0,:],x_AC[1,:],label='$AC$')
plt.plot(x_DE[0,:],x_DE[1,:],label='$DE$')
#
#
#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,E)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A''(4,6)','B''(1,5)','C''(7,2)','D''  ($\dfrac{13}{4}$,$\dfrac{23}{4}$)','E''  ($\dfrac{19}{4}$,$\dfrac{20}{5}$)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,-19), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.title('Area of Triangle ABC & ADE',size=12)
#if using termux
#plt.savefig('../figs/fig.pdf')
#else
plt.show()
