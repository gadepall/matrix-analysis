import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as  LA
import subprocess
import shlex

#construction of triangle
#input parameters
a=4
b=3


#points
A=np.array([0,b])
B=np.array([a,0])
C=np.array([0,0])
M=(1/2)*(A+B)
D=(2*M)-C

#generating a line
def line_gen(A,B):
    len=10
    dim = A.shape[0]
    x_AB = np.zeros((dim,len))
    lam_1 = np.linspace(0,1,len)
    for i in range(len):
        temp1 = A + lam_1[i]*(B-A)
        x_AB[:,i] = temp1.T
    return x_AB


#generating all lines
X_AB=line_gen(A,B)
X_CB=line_gen(C,B)
X_CA=line_gen(C,A)
X_DB=line_gen(D,B)
X_DC=line_gen(D,C)

#plotting all lines
plt.plot(X_AB[0,:],X_AB[1,:],label='AB')
plt.plot(X_CB[0,:],X_CB[1,:],label='BC')
plt.plot(X_CA[0,:],X_CA[1,:],label='AC')
plt.plot(X_DB[0,:],X_DB[1,:],label='DB')
plt.plot(X_DC[0,:],X_DC[1,:],label='DC')

#Length of sides
l_1 = np.linalg.norm(A-B)
l_2 = np.linalg.norm(C-B)
l_3 = np.linalg.norm(C-A)
l_4 = np.linalg.norm(D-B)
l_5 = np.linalg.norm(D-C)
l_6 = np.linalg.norm(A-M)
l_7 = np.linalg.norm(B-M)
l_8 = np.linalg.norm(C-M)
l_9 = np.linalg.norm(D-M)

#Directional vectors

m1 = A-B
m2 = C-B
m3 = C-A
m4 = D-B
m5 = D-C
m6 = A-M
m7 = B-M
m8 = C-M
m9 = D-M

angleAMC=np.degrees(np.arccos(np.dot(m6,m8)/(l_6*l_8)))
angleDMB=np.degrees(np.arccos(np.dot(m9,m7)/(l_9*l_7)))
angleACB=np.degrees(np.arccos(np.dot(m3,m2)/(l_3*l_2)))
angleDBC=np.degrees(np.arccos(np.dot(m4,m2)/(l_4*l_2)))

#(i) ∆ AMC ≅ ∆ ADE
if (l_6.all()==l_7.all()) and (angleAMC == angleDMB) and (l_9.all()==l_8.all()):
  print("(i) ∆ AMC ≅ ∆ ADE")

#(ii) ∠ DBC is a right angle.
if np.dot(m4,m2) == 0:
        print("(ii) ∠ DBC=",angleDBC)

#(iii)∆ DBC ≅ ∆ ACB
if (l_2.all()==l_1.all()) and (angleACB == angleDBC) and (l_4.all()==l_3.all()):
  print("(iii) ∆ DBC ≅ ∆ ACB")


#(iv)CM=(1/2)AB
if l_8 == (1/2)*(l_1):
  print("(iv) CM=(1/2)AB")

#Labeling the coordinates
tri_coords =np.vstack((B,C,D,A,M)).T
plt.scatter(tri_coords[0,:],tri_coords[1,:])
vert_labels = ['B','C','D','A','M']
for i,txt in enumerate(vert_labels):
  plt.annotate(txt,(tri_coords[0,i],tri_coords[1,i]),
      textcoords="offset points",
      xytext=(0,10),
      ha='center')

plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.legend(loc='best')
plt.axis('equal')
plt.savefig('/sdcard/Download/codes/lines/9.7.1.8/figs/fig.pdf')
plt.show()
