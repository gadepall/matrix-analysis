
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

import sys                                          #for path to external scripts
sys.path.insert(0, '/sdcard/Download/10/codes/CoordGeo')        #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if


def midy(a,b):
    e_2 = np.array(([0,1])) #standard basis vector
    #Diection vector
    n = a-b
    print(n)

    #Computations
    c = (np.linalg.norm(a)**2- np.linalg.norm(b)**2)/2

    x = c/(n@e_2)

    #Output
    Q = x*e_2
    return Q
    #Output
def mid(a,b):
    e_1 = np.array(([1,0])) #standard basis vector
    #Diection vector
    n = a-b
    print(n)

    #Computations
    c = (np.linalg.norm(a)**2- np.linalg.norm(b)**2)/2

    x = c/(n@e_1)

    #Output
    P = x*e_1

    #Output
    return P
A = np.array(([0,0]))
B = np.array(([5,0]))
C = np.array(([5,7]))
D = np.array(([0,7]))
    #Given points
P = mid(A,B)
Q = midy(B,C)
Q = Q + np.array(([B[0],0]))
R = mid(C,D)
R = R + np.array(([0,C[1]]))
S = midy(D,A)
S = S+np.array(([A[0],0]))
#print(P)
    #Output
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)
x_PQ = line_gen(P,Q)
x_QR = line_gen(Q,R)
x_RS = line_gen(R,S)
x_SP = line_gen(S,P)
x_PR = line_gen(P,R)
x_QS = line_gen(Q,S)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_BC[0,:],x_BC[1,:])
plt.plot(x_CD[0,:],x_CD[1,:])
plt.plot(x_DA[0,:],x_DA[1,:])
plt.plot(x_PQ[0,:],x_PQ[1,:])
plt.plot(x_QR[0,:],x_QR[1,:])
plt.plot(x_RS[0,:],x_RS[1,:])
plt.plot(x_SP[0,:],x_SP[1,:])
plt.plot(x_PR[0,:],x_PR[1,:])
plt.plot(x_QS[0,:],x_QS[1,:])
#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,P,Q,R,S)).T
#tri_coords = np.vstack((B,C,Q)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','P','Q','R','S']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
'''

tri_coords = np.vstack((A,C,D,B,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','C','D','B','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

'''
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/Download/FWC/trunk/matrix_Assignments/fig.pdf')
subprocess.run(shlex.split("termux-open '/sdcard/Download/FWC/trunk/matrix_Assignments/fig.pdf'")) 
#else
#plt.show()
#test comment
