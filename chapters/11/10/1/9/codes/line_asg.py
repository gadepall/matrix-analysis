import sys                                                    #for path to external scripts
sys.path.insert(0,'/home/student/CoordGeo')
#sys.path.insert(0,'/sdcard/Downloads/codes/CoordGeo')         #path to my scripts


#import module
import random as rd
import subprocess
import shlex
import numpy as np
import matplotlib.pyplot as plt

#local imports
from line.funcs import  *

# initialize given points
A=np.array([-2,-1])
B=np.array([4,0])
C=np.array([3,3])
D=np.array([-3,2])

#generating parallelogram
p=line_gen(A,B)
q=line_gen(C,D)
r=line_gen(B,C)
s=line_gen(A,D)

#plotting parallelogram
plt.plot(p[0,:],p[1,:],label='$P$')
plt.plot(q[0,:],q[1,:],label='$Q$')
plt.plot(r[0,:],r[1,:],label='$R$')
plt.plot(s[0,:],s[1,:],label='$S$')

#direction vectors
P=B-A 
Q=C-D 
R=B-C
S=A-D

#verifying the vertices
if np.array_equal(P, Q) and np.array_equal(R,S):  
	print('these vertices form parallelogram');
else:
	print('these vertices do not  form parallelogram');

#labelling
tri_coords = np.vstack((A,B,C,D)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A(-2,-1)','B(4,0)','C(3,3)','D(-3,2)']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(2,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() # minor
plt.axis([-8,8,-8,10])
plt.show()

#plt.savefig('/sdcard/Download/codes/Line_assignment.pdf)
#subprocess.run(shlex.split("termux-open '/sdcard/Download/codes/Line_assignment.pdf' "))
