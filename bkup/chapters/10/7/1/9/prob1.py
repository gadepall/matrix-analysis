import numpy as np 
import matplotlib.pyplot as plt 
from numpy import linalg as LA 
import math 
import sys     #for path to external scripts 
 
 
P = np.array(([5, -3])) 
R = np.array(([4, 6])) 
Q = np.array(([0 ,1])) 
R_1= np.array(([-4, 6]))

x = (np.linalg.norm(P-Q))
y = (np.linalg.norm(R-Q))
print('distance AO = ',x)
print('distance BO = ',y)
if(x==y):
  print("Therfore Point O is equidistant from point A and Point B")
 
def line_gen(A,B): 
   len =10  
   dim = A.shape[0]
   x_AB = np.zeros((dim,len)) 
   lam_1 = np.linspace(0,1,len) 
   for i in range(len): 
     temp1 = A + lam_1[i]*(B-A) 
     x_AB[:,i]= temp1.T 
   return x_AB 
 
   
x_QP = line_gen(Q,P) 
x_QR = line_gen(Q,R) 
x_QR_1= line_gen(Q,R_1)

 
 
#Plotting all lines 
plt.plot(x_QP[0,:],x_QP[1,:],label='$QP$') 
plt.plot(x_QR[0,:],x_QR[1,:],label='$QR$') 
plt.plot(x_QR_1[0,:],x_QR_1[1,:],label='$QR\u2081$') 
 
 
#Labeling the coordinates 
tri_coords = np.vstack((P,R,R_1,Q)).T 
plt.scatter(tri_coords[0,:], tri_coords[1,:]) 
vert_labels = ['P','R','R\u2081','Q'] 
for i, txt in enumerate(vert_labels): 
    plt.annotate(txt, # this is the text 
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label 
                 textcoords="offset points", # how to position the text 
                 xytext=(0,10), # distance from text to points (x,y) 
                 ha='center') # horizontal alignment can be left, right or center 
 
plt.xlabel('$x-axis$') 
plt.ylabel('$y-axis$') 
plt.legend(loc='best') 
plt.grid() # minor 
plt.axis('equal') 
plt.title('equidistant point',size=12) 
plt.text(5,-3,'   (5,-3)') 
plt.text(4,6,'   (4,6)') 
plt.text(-4,6,' (-4,6)')
plt.text(0,1,'   (0,1)') 
#if using termux
plt.savefig('/sdcard/FWC/vectormaths/figs/fig.pdf')
#else
#plt.show()
