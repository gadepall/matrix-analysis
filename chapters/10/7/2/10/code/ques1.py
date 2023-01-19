import numpy as np 
import matplotlib.pyplot as plt 
from numpy import linalg as LA 
import math 
import sys     #for path to external scripts 
  
def bd(point1,point2,point3,point4):
    A=np.array(point1)
    B=np.array(point2)
    C=np.array(point3)
    D=np.array(point4)
    AD = np.array((A-D))
    BA = np.array((B-A))
    area = np.cross(AD,BA)
    print(area)
    return area


point1= np.array(([3, 0]))
point2= np.array(([4, 5]))
point3= np.array(([-1, 4]))
point4= np.array(([-2, -1]))
length = bd(point1,point2,point3,point4)
A = np.array(([3, 0]))
B = np.array(([4, 5]))
C= np.array(([-1,4]))
D = np.array(([-2, -1])) 
 
def line_gen(A,B): 
   len =10 
   dim = A.shape[0] 
   x_AB = np.zeros((dim,len)) 
   lam_1 = np.linspace(0,1,len) 
   for i in range(len): 
     temp1 = A + lam_1[i]*(B-A) 
     x_AB[:,i]= temp1.T 
   return x_AB 

 
 
x_AB = line_gen(A,B) 
x_BC = line_gen(B,C) 
x_CD = line_gen(C,D) 
x_DA = line_gen(D,A) 
x_BD = line_gen(B,D)
x_AC = line_gen(A,C)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$') 
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$') 
plt.plot(x_CD[0,:],x_CD[1,:],label='$CD$') 
plt.plot(x_DA[0,:],x_DA[1,:],label='$DA$') 
plt.plot(x_BD[0,:],x_BD[1,:],label='$BD$') 
plt.plot(x_AC[0,:],x_AC[1,:],label='$AC$') 

plt.xlabel('$x-axis$') 
plt.ylabel('$y-axis$') 
plt.legend(loc='best') 
plt.grid() # minor 
plt.axis('equal') 
plt.title('Rhombus',size=12) 
plt.text(3,0,'   A(3,0)') 
plt.text(4,5,'   B(4,5)') 
plt.text(-1,4,'   C(-1,4)') 
plt.text(-2,-1,'   D(-2,-1)') 
#plt.text(6,6,'   B-D(6,6)') 
#plt.text(4,-4,'   A-C(4,-4)') 

#if using termux
plt.savefig('../fig.pdf')
#else
plt.show()