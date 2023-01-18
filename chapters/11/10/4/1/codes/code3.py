#p-3
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
A=  np.array(([0,0]))
#Direction vector 
k1=1
k2=6
m=np.array(([k1-3,-4+k1**2]))
o=np.array(([k2-3,-4+k2**2]))                                                               
z=np.array(([0,1],[-1,0]))                            
n=m@z
p=o@z
print('equation of line passing through origin (0,0) with k=',k1,'is',np.array2string(m),'x=0')
print('equation of line passing through origin (0,0) with k=',k2,'is',np.array2string(o),'x=0') 
##Generating the line  
s=1 
t=-2 
u=-0.2
v=0.2
x_AB = line_dir_pt(n,A,s,t) 
x_CD = line_dir_pt(p,A,u,v) 
 
#Plotting all lines 
plt.plot(x_AB[0,:],x_AB[1,:],label='({})x=0'.format(" ".join([str(i) for i in m])))   
plt.plot(x_CD[0,:],x_CD[1,:],label='({})x=0'.format(" ".join([str(i) for i in o])))  
 
    # Add labels and show the plot 
plt.xlabel('x') 
plt.ylabel('y') 
plt.title('Equation of line passing through origin') 
plt.legend(loc='best') 
plt.grid() 
plt.axis('equal') 
#plt.show()
plt.savefig('../figs/fig3.pdf')

