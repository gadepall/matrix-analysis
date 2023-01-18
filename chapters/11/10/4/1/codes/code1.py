#p-1
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
A=  np.array(([0,(6/5)])) 
k=3
m=np.array(([k-3,(-4+k**2)]))                                                               
z=np.array(([0,1],[-1,0]))                            
n=z@m
f=m@A

print('equation of line parallel to x-axis with k=',k,'is',np.array2string(m),'x=',int(f))

##Generating the line  
k1=-1 
k2=1 
x_AB = line_dir_pt(n,A,k1,k2) 
 
#Plotting all lines 
plt.plot(x_AB[0,:],x_AB[1,:],label='({})X={}'.format(" ".join([str(i) for i in m]),int(f))) 
 
    # Add labels and show the plot 
plt.xlabel('x') 
plt.ylabel('y') 
plt.title('Equation of line parallel to x-axis for k=3') 
plt.legend(loc='best') 
plt.grid() 
plt.axis('equal') 
#plt.show()
plt.savefig('../figs/fig1.pdf')
