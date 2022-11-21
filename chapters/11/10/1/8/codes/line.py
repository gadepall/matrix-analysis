
#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#import sys                                          #for path to external scripts
#sys.path.insert(0, '/sdcard/Download/c2_fwc/trunk/CoordGeo')        #path to my scripts
#Generate line points
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB
  
#local imports
#from line.funcs import*
#from triangle.funcs import*
#from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if
A =np.array(([1,-1]))                          
B =np.array(([2,1]))                       
C =np.array(([4,5]))
 
#parameters  of transpose vectors
D =A-B
E =A-C
print("A transpose")
print(A.T)
print("B transpose:")
print(B.T)
print("C transpose:")
print(C.transpose())                        
print("matrix of transpose")
F  =np.array(([D,E]))                       
print(F)
print("rank of matrix")                      
print(np.linalg.matrix_rank(F))
 
if(np.linalg.matrix_rank(F) == 1):
      print("points are collinear")
else:
      print("points are non collinear")
D=2/4+2/4
print("a/h+b/k = ",D)      
 
 ##Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Diameter$')
plt.plot(x_BC[0,:],x_BC[1,:])#,label='$Diameter$')

#Labeling the coordinates
tri_coords = np.vstack((A,B,C)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','c']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/Download/c2_fwc/trunk/line_assignment/docs/sline.png')
subprocess.run(shlex.split("termux-open '/sdcard/Download/c2_fwc/trunk/line_assignment/docs/line.pdf'")) 
#else
plt.show()





