import sys                                          #for path to external scripts
sys.path.insert(0,'/home/student/Downloads/iith-fwc-2022-23-main/CoordGeo')         #path to my scripts
#Circle Assignment
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

#local imports
from line.funcs import *


#input parameters
P=np.array([0,-4]);
B=np.array([8,0]);
M=(P+B)/2
O=np.array([0,0])
plt.plot(M[0],M[1],'ro');

#generating line
x_R1 = line_gen(P,B);
x_R2 = line_gen(O,M);

#plotting line
plt.plot(x_R1[0,:],x_R1[1,:],label='$line1$');
plt.plot(x_R2[0,:],x_R2[1,:],label='$line2$');

#direction vector of line joining two points
m=O-M
#print(m)
#finding slope of OM line
e1=np.array([[1],[0]])
e2=np.array([[0],[1]])
m1=(m)@(e1)
m2=(m)@(e2)
slope=m2/m1
print("slope of a line:",slope);

#Labeling the coordinates
tri_coords = np.vstack((O,M,P,B)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O(0,0)','M(4,-2)','P','B']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


# use set_position
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')

plt.legend()
plt.grid(True) # minor
#plt.savefig('/sdcard/download/codes/line_assignment/document/line.png')
#plt.subprocess('/sdcard/download/codes/line_assignment/document/line.pdf')

plt.axis('equal')
plt.show()
