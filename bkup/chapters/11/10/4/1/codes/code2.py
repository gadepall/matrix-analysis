#p-2
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
A=  np.array(([-4,0]))
D= np.array(([(24/5),0])) 
#Direction vector 
k1=2
k2=-2
m=np.array(([k1-3,4-k1**2]))
o=np.array(([k2-3,4-k2**2]))                                                               
z=np.array(([0,1],[-1,0]))                            
n=m@z
p=o@z
f=m@A
g=o@D
print('equation of line parallel to y-axis with k=',k1,'is',np.array2string(m),'x=',int(f))
print('equation of line parallel to y-axis with k=',k2,'is',np.array2string(o),'x=',int(g))
##Generating the line  
s=-1 
t=1 
u=-5
v=5
x_AB = line_dir_pt(n,A,u,v) 
x_CD = line_dir_pt(p,D,s,t) 
 
#Plotting all lines 
plt.plot(x_AB[0,:],x_AB[1,:],label='({})x={}'.format(" ".join([str(i) for i in m]),int(f))) 
plt.plot(x_CD[0,:],x_CD[1,:],label='({})x={}'.format(" ".join([str(i) for i in o]),int(g)))  
 
    # Add labels and show the plot 
plt.xlabel('x') 
plt.ylabel('y') 
plt.title('Equation of line parallel to y-axis for k=$\pm$2') 
plt.legend(loc='upper right') 
plt.grid() 
plt.axis('equal') 
#plt.show()
plt.savefig('../figs/fig2.pdf')

