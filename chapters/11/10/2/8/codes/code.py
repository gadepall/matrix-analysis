import numpy as np  
import mpmath as mp  
import math as ma  
import matplotlib.pyplot as plt  
from numpy import linalg as LA  
from fractions import Fraction   
  
def line_dir_pt(m,A,k1,k2):  
  len = 10  
  dim = A.shape[0]  
  x_AB = np.zeros((dim,len))  
  lam_1 = np.linspace(k1,k2,len)  
  for i in range(len):  
    temp1 = A + lam_1[i]*m  
    x_AB[:,i]= temp1.T  
  return x_AB  
 
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
 
#Normal vector is 
theta=np.radians 
m=np.array(([np.cos(theta(30)),np.sin(theta(30))]))   
d=5 
c=d*np.linalg.norm(m) 
print('normal vector is',m) 
print('the value of c is',c) 
print('The equation of line is',np.array2string(m),'x=',int(c)) 
 
#Input parameters  
A=  np.array(([(8/3)*np.sqrt(3),2])) 
B=  np.array(([-(8/3)*np.sqrt(3),-2])) 
O=  np.array(([0,0])) 
#Direction vector  
k1=1 
k2=6 
#m=np.array(([np.sqrt(3)/2,1/2]))                                                                
z=np.array(([0,1],[-1,0]))                             
n=m@z 
 
##Generating the line   
k1=5  
k2=-2  
 
x_AB = line_dir_pt(n,A,k1,k2)  
x_AC = line_dir_pt(n,B,k1,k2)  
x_OA = line_gen(O,A) 
x_OB = line_gen(O,B) 
  
#Plotting all lines  
plt.plot(x_AB[0,:],x_AB[1,:],label='$\\left(\dfrac{\sqrt{3}}{2} \\dfrac{1}{2}\\right)$ X=5') 
plt.plot(x_AC[0,:],x_AC[1,:],label='$\\left(\dfrac{\sqrt{3}}{2} \\dfrac{1}{2}\\right)$ X=-5')  
plt.plot(x_OA[0,:],x_OA[1,:],label='d=5')    
plt.plot(x_OB[0,:],x_OB[1,:],label='d=5')     
 
#Labeling the coordinates  
tri_coords = O.T  
plt.scatter(tri_coords[0], tri_coords[1])  
vert_labels = ['O(0,0)']  
for i, txt in enumerate(vert_labels):  
      plt.annotate(txt, # this is the text  
                 (tri_coords[0], tri_coords[1]), # this is the point to label  
                 textcoords="offset points", # how to position the text  
                 xytext=(0,10), # distance from text to points (x,y)  
                 ha='center') # horizontal alignment can be left, right or center  
  
    # Add labels and show the plot  
plt.xlabel('x')  
plt.ylabel('y')  
plt.title('Equation of line')  
plt.legend(loc='upper left')  
plt.grid()  
plt.axis('equal')  
#plt.show() 
plt.savefig('../figs/fig.pdf')
