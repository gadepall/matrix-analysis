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
 
 
def line_gen(A,B): 
   len =10 
   dim = A.shape[0] 
   x_AB = np.zeros((dim,len)) 
   lam_1 = np.linspace(0,1,len) 
   for i in range(len): 
     temp1 = A + lam_1[i]*(B-A) 
     x_AB[:,i]= temp1.T 
   return x_AB 
    
#Input parameters 
P=  np.array(([-1,2]))      
D=np.array(([0,0])) 
 
#Direction vector 
m=np.array(([1,-2]))   #use normal vector here                                                            
z=np.array(([0,1],[-1,0]))                            
n=z@m                                      
print(n) 
print(m@P) 
print(m@D) 
 
 
##Generating the line  
k1=-2 
k2=2 
x_AB = line_dir_pt(n,P,k1,k2) 
x_CD = line_gen(D,P) 
 
  
#Plotting all lines 
plt.plot(x_AB[0,:],x_AB[1,:],label='x-2y+5') 
plt.plot(x_CD[0,:],x_CD[1,:],label='OP') 
 
tri_coords = P.T   
plt.scatter(tri_coords[0], tri_coords[1])   
vert_labels = ['P(-1,2)']   
for i, txt in enumerate(vert_labels):   
      plt.annotate(txt, # this is the text   
                 (tri_coords[0], tri_coords[1]), # this is the point to label   
                 textcoords="offset points", # how to position the text   
                 xytext=(0,10), # distance from text to points (x,y)   
                 ha='center') # horizontal alignment can be left, right or center   
   
    # Add labels and show the plot  
 
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
