import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          
sys.path.insert(0,'/home/admin999/navya/matrix/CoordGeo')
from line.funcs import *

r1=5
r2=8
D=np.array(([0,0]))
e1=np.array(([1,0]))
C=r2*e1
theta1 = np.pi*2/5
A = r1*np.array(([np.cos(theta1),np.sin(theta1)]))
B=A+C-D
Q=D+C-A
P = (D+C)/2

v1=B-C
v2=P-C
ar_t1=0.5*np.linalg.norm((np.cross(v1,v2)))

v3=D-P
v4=Q-D
ar_t2=0.5*np.linalg.norm((np.cross(v3,v4)))
#BPC='{0:.2g}'.format(ar_t1)
#DPQ='{0:.2g}'.format(ar_t2)

print("Ar(BPC)=",ar_t1)
print("Ar(DPQ)=",ar_t2)
if(ar_t1==ar_t2):
    print("area of triangle BPC is equal to area of triangle DPQ")
else : print("area of triangle BPC is not equal to area of triangle DPQ")
#print(A)
#print(B)
#print(C)
#print(Q)
x_DC = line_gen(D,C)
x_DA = line_gen(D,A)
x_AC = line_gen(A,C)
x_QC = line_gen(Q,C)
x_DQ = line_gen(D,Q)
x_CB = line_gen(C,B)
x_AB = line_gen(A,B)
x_AQ = line_gen(A,Q)
x_BP = line_gen(B,P)
#x_BH1= line_gen(B,H1)

plt.plot(x_DC[0,:],x_DC[1,:])#,label='$Line')
plt.plot(x_DA[0,:],x_DA[1,:])#,label='$Line')
plt.plot(x_AC[0,:],x_AC[1,:])#,label='$Line')
plt.plot(x_QC[0,:],x_QC[1,:])#,label='$Line')
plt.plot(x_DQ[0,:],x_DQ[1,:])#,label='$Line')
plt.plot(x_CB[0,:],x_CB[1,:])#,label='$Line')
plt.plot(x_AB[0,:],x_AB[1,:])#,label='$Line')
plt.plot(x_AQ[0,:],x_AQ[1,:])#,label='$Line')
plt.plot(x_BP[0,:],x_BP[1,:])#,label='$Line')
#plt.plot(x_BH1[0,:],x_BH1[1,:])#,label='$Line')

tri_coords = np.vstack((A,C,D,Q,B,P)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','C','D','Q','B','P']
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
plt.axis([-2,12,-6,6])
plt.savefig('/home/admin999/navya/matrix/line.pdf')
plt.show()
