import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
#local imports
#from line.funcs import *
#from triangle.funcs import *
#from conics.funcs import circ_gen
#from conics.funcs import *
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB
  
theta = np.pi/3
a =8
c=5
b=round(np.sqrt((a**2)+(c**2)-(2*a*c*np.cos(theta))))
print("b=",b)
B = np.array([0,0])
C = np.array([a,0])
e1 =np.array(([1,0]))
#theta_1=np.pi/3
A = c*np.array([np.cos(theta),np.sin(theta)])
print("A=",A)
print("B=",B)
print("C=",C)
D=(2*c*np.sin(theta/2))*e1
print("D=",D)

angBCA=np.arccos(((a**2)+(b**2)-(c**2))/(2*a*b))
print("angle BCA",np.degrees(angBCA))
angACE=(np.pi/2)-(theta/2)
print('angACE=',np.degrees(angACE))
angC=np.pi-(angBCA+angACE)
print('angC=',np.degrees(angC))
E=C+(2*b*np.sin(theta/2))*np.array([np.cos(angC),np.sin(angC)])
print(E)
#Directional vectors
m_1=B-A
m_2=C-B
m_3=C-A
m_4=D-A
m_5=D-E
m_6=E-A

f1 = np.linalg.norm(m_1)
f2 = np.linalg.norm(m_2)
f3 = np.linalg.norm(m_3)
f4 = np.linalg.norm(m_4)
f5 = np.linalg.norm(m_5)
f6 = np.linalg.norm(m_6)
angleBAC=np.degrees(np.arccos(np.dot(m_1,m_3)/(f1*f3)))
angleDAE=np.degrees(np.arccos(np.dot(m_4,m_6)/(f4*f6)))
angleDAC=np.degrees(np.arccos(np.dot(m_4,m_3)/(f4*f3)))
angleCAE=np.degrees(np.arccos(np.dot(m_3,m_6)/(f3*f6)))
print("BAC=",round(angleBAC))
print("DAE=",round(angleDAE))
print("DAC=",angleDAC)
print("CAE=",angleCAE)


print("length of EA",round(f6))
print("length of AC",round(f3))
print("length of AB",round(f1))
print("length of AD",round(f4))
print("length of BC",round(f2))
print("length of DE",round(f5))
#proving the triangles are congruent


if (round(f1)==round(f4)) and round(angleBAC) == round(angleDAE) and round(f3)==round(f6):
  print("(i)∆ ABC ≅ ∆ ADE")
  print("(ii) BC=DE")
  
#generating lines
x_AB = line_gen(A,B)
x_CB = line_gen(C,B)
x_CA = line_gen(C,A)
x_AD = line_gen(A,D)
x_DE = line_gen(D,E)
x_CE = line_gen(C,E)
x_AE = line_gen(A,E)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:])
plt.plot(x_CB[0,:],x_CB[1,:])
plt.plot(x_CA[0,:],x_CA[1,:])
plt.plot(x_AD[0,:],x_AD[1,:])
plt.plot(x_DE[0,:],x_DE[1,:])
plt.plot(x_CE[0,:],x_CE[1,:])
plt.plot(x_AE[0,:],x_AE[1,:])

#Labeling the coordinates
tri_coords = np.vstack((A,B,C,D,E)).T
#tri_coords = np.vstack((B,C,Q)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C', 'D','E']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid() 
plt.axis('equal')
plt.savefig('/sdcard/Download/codes/lines/9.7.1.6/fig/fig.pdf') 
plt.show()
