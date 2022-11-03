import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as  LA

import sys  #for path to external scripts
sys.path.insert(0,'/sdcard/FWCmodule1/line/code/CoordGeo') #path to my scripts

#local imports
from line.funcs import *
from triangle.funcs import *
from conics.funcs import circ_gen

#if using termux
import subprocess
import shlex


r=5
b=4
K=3 
theta= np.pi/3

A=r*np.array(([np.cos(theta),np.sin(theta)]))
B=np.array(([0,0]))
e1=np.array(([1,0]))
C=b*e1 
D=A+C-B


#points P and Q

P=(B+K*D)/(K+1)
Q=(D+B*K)/(1+K)



#generating all lines

X_AB=line_gen(A,B)
X_BC=line_gen(C,B)
X_CD=line_gen(C,D)
X_DA=line_gen(D,A)
X_BD=line_gen(B,D) 
X_AQ=line_gen(A,Q)
X_AP=line_gen(A,P)
X_QC=line_gen(Q,C)
X_PC=line_gen(P,C)
#plotting all lines

plt.plot(X_AB[0,:],X_AB[1,:])
plt.plot(X_BC[0,:],X_BC[1,:])
plt.plot(X_CD[0,:],X_CD[1,:])
plt.plot(X_DA[0,:],X_DA[1,:])
plt.plot(X_BD[0,:],X_BD[1,:])
plt.plot(X_AQ[0,:],X_AQ[1,:])
plt.plot(X_AP[0,:],X_AP[1,:])
plt.plot(X_QC[0,:],X_QC[1,:])
plt.plot(X_PC[0,:],X_PC[1,:])

#Direction vectors
m_1 = A-B
m_2 = D-C
m_3 = B-C
m_4 = A-D
m_5 = B-Q
m_6 = P-D

n_1 = A-P
n_2 = Q-C
n_3 = A-Q
n_4 = P-C


if (m_3.all()==m_4.all()) and (m_5.all()==m_6.all()) and (n_1.all()==n_2.all()):
	print("(i)∆ APD ≅ ∆ CQB")
   
if n_1.all()==n_2.all():
   print("(ii) AP=CQ")
   
if (m_1.all()==m_2.all()) and (m_5.all()==m_6.all()) and (n_3.all()==n_4.all()):
   print("(iii)∆ AQB ≅ ∆ CPD")
   
if n_3.all()==n_4.all():
	print("(iV) AQ=PC")
	print(" (v) Quadrilateral APCQ is a parallelogram")

#Labeling the coordinates
tri_coords =np.vstack((B,C,D,A,Q,P)).T

plt.scatter(tri_coords[0,:],tri_coords[1,:])
vert_labels = ['B','C','D','A','Q','P']
for i,txt in enumerate(vert_labels):
	plt.annotate(txt,
			(tri_coords[0,i],tri_coords[1,i]),
			textcoords="offset points",
			xytext=(0,10),
			ha='center')

plt.xlabel('$X$')
plt.ylabel('$Y$')
#plt.legend(loc='best')
plt.grid()
plt.axis('equal')

#plt.show()
plt.savefig('/sdcard/FWCmodule1/line/output.pdf')
subprocess.run(shlex.split("termux-open /sdcard/FWCmodule1/line/output.pdf"))







