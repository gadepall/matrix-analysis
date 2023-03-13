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
A=  np.array(([0,1]))     
D=np.array(([0,-1]))

#Direction vector
m=np.array(([1,1]))   #use normal vector here                                                           
z=np.array(([0,1],[-1,0]))                           
n=z@m                                     
print(n)
print(m@A)
print(m@D)

##Generating the line 
k1=-4
k2=2
x_AB = line_dir_pt(m,A,k1,k2)
x_CD = line_dir_pt(m,D,k1,k2)

 

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='Line equation')
plt.plot(x_CD[0,:],x_CD[1,:],label='Line equation')


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('/sdcard/matrix/code/fig.pdf')
#subprocess.run(shlex.split("termux-open /sdcard/matrix/code/fig.pdf"))
#else
plt.show()
