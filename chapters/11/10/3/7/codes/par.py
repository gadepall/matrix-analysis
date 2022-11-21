import numpy as np
import matplotlib.pyplot as plt

#import sys
#sys.path.insert(0,'/sdcard/Download/matrices/CoordGeo')

#from line.funcs import *
#from triangle.funcs import *
#from conics.funcs import circ_gen *
import subprocess
import shlex
m=np.array(([4,3]))
A=np.array(([0,-1/2]))
B=np.array(([-2,3]))
n=np.array(([3,-4]))
c=n@B
print('the line equation is 3x-4y=',c)
def line_dir_pt(m,A,k1,k2):
  len = 10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(k1,k2,len)
  for i in range(len):
    temp1 = A + lam_1[i]*m
    x_AB[:,i]= temp1.T
  return x_AB

line1=line_dir_pt(m,A,-5,5)
line2=line_dir_pt(m,B,-5,5)

plt.plot(line1[0,:],line1[1,:],'r',label='$3x-4y+2=0$')
plt.plot(line2[0,:],line2[1,:],'r',label='$3x-4y+18=0$')

#Labeling the coordinates
tri_coords = np.vstack((A,B)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.show()

plt.savefig('/sdcard/Download/matrices/fig/par.pdf')
subprocess.run(shlex.split("termux-open '/sdcard/Download/matrices/fig/par.pdf'"))
