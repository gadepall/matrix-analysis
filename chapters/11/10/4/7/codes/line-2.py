
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
A=  np.array(([3,2])) 
D=np.array(([0,6])) 
 
#Direction vector 
m=np.array(([3,2]))                                                               
z=np.array(([0,1],[-1,0]))                            
n=m@z
i=n*D
j=i[0]
print(n) 
##Generating the line  
k1=-4
k2=4 
x_AB = line_dir_pt(n,A,k1,k2) 
x_CD = line_dir_pt(m,D,k1,k2) 
 
 
 
#Plotting all lines 
plt.plot(x_AB[0,:],x_AB[1,:],label='2x-3y+18=0') 
plt.plot(x_CD[0,:],x_CD[1,:],label='(-2 3)X=-18')
 
 
 
#Labeling the coordinates 
tri_coords = D.T 
plt.scatter(tri_coords[0], tri_coords[1]) 
vert_labels = ['D(0,6)'] 
for i, txt in enumerate(vert_labels): 
      plt.annotate(txt, # this is the text 
                 (tri_coords[0], tri_coords[1]), # this is the point to label 
                 textcoords="offset points", # how to position the text 
                 xytext=(0,10), # distance from text to points (x,y) 
                 ha='center') # horizontal alignment can be left, right or center 
 
    # Add labels and show the plot 
plt.xlabel('x') 
plt.ylabel('y') 
plt.title('Equation of line drawn perpendicular which meets y-axis 2x-3y+18=0') 
plt.legend(loc='best') 
plt.grid() 
plt.axis('equal') 
plt.show()
#plt.show()
plt.savefig('../figs/fig.pdf')

