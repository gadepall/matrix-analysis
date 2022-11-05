import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import sys                                          #for path to external scripts
sys.path.insert(0, '/home/bhavani/Documentsmatrix/matrix_conic/CoordGeo')        #path to my scripts

#local imports
#from line.funcs import *
#from triangle.funcs import *
#from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex
#end if

def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

d = 7
theta = np.pi/2.5
r = 8
A = np.array([0,0])
B = np.array([d,0])
D = np.array([r*np.cos(theta) , r*np.sin(theta)])
C = np.array([(D[0]/1.5)+B[0] , (D[1]/1.5)+B[1]])   #D/2.5+B

    #Given points
P = np.array((A+B)/2)		#(A+B)/2
Q = np.array((B+C)/2)		#(B+C)/2
R = np.array((C+D)/2)		#(C+D)/2
S = np.array((A+D)/2)		#(A+D)/2

#print(int(P[0]) , int(P[1]))
#print(int(Q[0]) , int(Q[1]))
#print(int(R[0]) , int(R[1]))
#print(int(S[0]) , int(S[1]))

#Finding Direction Vector of SR
dsr = np.array(R-S)

#Direction vector of line AC
dac = np.array(C-A)
#print(dac)
dac2 = np.array((C-A)/2)

dpq = np.array(Q-P)
if((dsr==dac2).all() ):
	print("1. SR || AC")
	print("   SR = AC/2")
if((dsr==dac2).all() and (dpq == dsr).all()):
	print("2. PQ = SR")
	print("3. PQRS is a parallelogram")

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CD = line_gen(C,D)
x_DA = line_gen(D,A)
x_PQ = line_gen(P,Q)
x_QR = line_gen(Q,R)
x_RS = line_gen(R,S)
x_SP = line_gen(S,P)
x_AC = line_gen(A,C)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_BC[0,:],x_BC[1,:])
plt.plot(x_CD[0,:],x_CD[1,:])
plt.plot(x_DA[0,:],x_DA[1,:])
plt.plot(x_PQ[0,:],x_PQ[1,:])
plt.plot(x_QR[0,:],x_QR[1,:])
plt.plot(x_RS[0,:],x_RS[1,:])
plt.plot(x_SP[0,:],x_SP[1,:])
plt.plot(x_AC[0,:],x_AC[1,:])
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
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/Download/matrix_line/line1.pdf')
subprocess.run(shlex.split("termux-open '/sdcard/Download/codes/matrix_line/line1.pdf'")) 
#else
#plt.show()
#test comment
