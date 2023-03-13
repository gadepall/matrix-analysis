#Python libraries for math and graphics   
import numpy as np   
import mpmath as mp   
import math as ma   
import matplotlib.pyplot as plt   
from numpy import linalg as LA   
   
def line_dir_pt(m,A,k1,k2):   
  len = 10   
  dim = A.shape[0]   
  x_AB = np.zeros((dim,len))   
  lam_1 = np.linspace(k1,k2,len)   
  for i in range(len):   
    temp1 = A + lam_1[i]*m   
    x_AB[:,i]= temp1.T   
  return x_AB    
   
#Input parameters    
  
A = np.array(([2.4,3]))  
B = np.array(([4,1]))  
P=  np.array(([0,2.1111111111]))  
  
#Direction vector   
s=np.array(([-7/9,1]))   
m=np.array(([7,-9]))                                                                 
z=np.array(([0,1],[-1,0]))                              
n=m@z  
print(n)  
t=s@A 
print(t)  
##Generating the line    
k1=-0.8 
k2=0.8 
k3=-7 
k4=9  
x_AB = line_dir_pt(n,P,k1,k2)   
x_CD = line_dir_pt(s,A,k3,k4)   
  
   
#Plotting all lines   
plt.plot(x_AB[0,:],x_AB[1,:],label='(7 9)x=19')  
plt.plot(x_CD[0,:],x_CD[1,:],label='(-0.7 1)x=-2.1')  
    
#Labeling the coordinates  
tri_coords = np.vstack((A,B)).T  
plt.scatter(tri_coords[0,:], tri_coords[1,:])  
vert_labels = ['A''(2.4,3) ','B''(4,1)']  
for i, txt in enumerate(vert_labels):  
    plt.annotate(txt, # this is the text  
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label  
                 textcoords="offset points", # how to position the text  
                 xytext=(15,10), # distance from text to points (x,y)  
                 ha='center') # horizontal alignment can be left, right or center  
   
    # Add labels and show the plot   
plt.xlabel('x')   
plt.ylabel('y')   
plt.title('points (2.4,3) and (4,1) intersects the line 7x-9y19= 0 at right angle')   
plt.legend(loc='upper right')   
plt.grid()   
plt.axis('equal')   
plt.show()  

