#Python libraries for math and graphics
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import sys      #for path to external scripts
sys.path.insert(0,'/sdcard/Download/parv/CoordGeo')

#local imports
from conics.funcs import circ_gen


#if using termux
import subprocess
import shlex

def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB

#input parameters
r = 1
O = np.array([0,0])
theta1 = (11*math.pi)/18

#Calculations to find the angle
P = O+np.array([math.cos(theta1),math.sin(theta1)])
Q = O+np.array([1,0]) 
n1 = P-O
n2 = Q-O
m1 = np.array([math.sin(theta1),-math.cos(theta1)])
m2 = np.array([0,1])
theta = math.acos((m1.T@m2)/(np.linalg.norm(m1)*np.linalg.norm(m2)))*(180/math.pi)
print(f"The angle between the two tangents is {theta} degrees")

#Calculation of tangent point
A = np.array([n1,n2])
b = np.array([1,1])
T = np.linalg.solve(A,b)

#Generating the circle
x_circ= circ_gen(O,r)

#generating the lines
x_TP = line_gen(T,P)
x_TQ = line_gen(T,Q)
x_QC = line_gen(Q,O)
x_PC = line_gen(P,O)

#plotting
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')
plt.plot(x_TP[0,:],x_TP[1,:],label='$TP$')
plt.plot(x_TQ[0,:],x_TQ[1,:],label='$TQ$')
plt.plot(x_QC[0,:],x_QC[1,:],label='$QO$')
plt.plot(x_PC[0,:],x_PC[1,:],label='$PO$')

#Labeling the coordinates
tri_coords = np.vstack((T,P,Q,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['T','P','Q','O']
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
#if using termux
#plt.savefig('../figs/problem1.pdf')
#subprocess.run(shlex.split("termux-open '../figs/problem1.pdf'")) 
plt.savefig('/sdcard/Download/latexfiles/tangent/figs/tangent2.png')
#subprocess.run(shlex.split("termux-open "+os.path.join(script_dir, fig_relative)))
#else
plt.show()
