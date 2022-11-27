import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math

import subprocess
import shlex
def ellipse_gen(a,b):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_ellipse = np.zeros((2,len))
	x_ellipse[0,:] = a*np.cos(theta)
	x_ellipse[1,:] = b*np.sin(theta)
	return x_ellipse
def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB


#Standard ellipse
a = 3
b = 4
#x = ellipse_gen(a,b)
v=np.array(([16,0],[0,9]))
v1=np.linalg.inv(v)
print(v1)
u=np.array([0,0])
f=-(a*b)
n1=np.array([0,1])
n2=np.array([1,0])
print(f)

#axes
k1 = +(np.sqrt((u@v1@u)-f)/(n1@v1@n1))
k2 = -(np.sqrt((u@v1@u)-f)/(n1@v1@n1))
print(k1)
print(k2)
q1=v1@(k1*n1-u)+np.array([0,0.54])
print(q1)
q2=v1@(k2*n1-u)-np.array([0,0.54])
print(q2)
k3=+(np.sqrt((u@v1@u)-f)/(n2@v1@n2))
k4 = -k3
print(k3)
print(k4)
q3=v1@(k3*n2-u)-np.array([0.46,0])
print(q3)
q4=v1@(k4*n2-u)+np.array([0.46,0])
print(q4)
P=np.array([-3,4]) #x-intercept
R= -P #y-intercept
Q= np.array([3,4]) #x-intercept
S= -Q #y-intercept


#y=np.linspace(-1,1,100)
#y=np.linspace(-1,1,100)
#t1=0+4*y
#plt.plot(y,t1)
#generating the lines
xq1q2= line_gen(q1,q2)
xq3q4 = line_gen(q3,q4)
xPQ=line_gen(P,Q)
xQR=line_gen(Q,R)
xRS=line_gen(R,S)
xSP=line_gen(S,P)

#generating the ellipse
x_ellipse=ellipse_gen(a,b)

#Plotting the ellipse
plt.plot(x_ellipse[0,:],x_ellipse[1,:],label='Ellipse')
#plotting the lines
plt.plot(xq1q2[0,:],xq1q2[1,:],label='$Tangent$')
plt.plot(xq3q4[0,:],xq3q4[1,:],label='$Tangent$')
plt.plot(xPQ[0,:],xPQ[1,:],label='$Tangent$')
plt.plot(xQR[0,:],xQR[1,:],label='$Tangent$') 
plt.plot(xRS[0,:],xRS[1,:],label='$Tangent$')
plt.plot(xSP[0,:],xSP[1,:],label='$Tangent$')
#Labeling the coordinates
ellipse_coords = np.vstack((q1,q2,q3,q4,P,Q,R,S)).T
plt.scatter(ellipse_coords[0,:], ellipse_coords[1,:])
vert_labels = ['q1','q2','q3','q4','P','Q','R','S']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (ellipse_coords[0,i], ellipse_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('./home/apiiit-rkv/conic.pdf')
#plt.savefig('/home/user/txhome/storage/shared/github/training/math/figs/ellipse.png')
#plt.show()

plt.savefig('/home/apiiit-rkv/Desktop/fwc_matrix/matrix_conics/conic_1.pdf')
#subprocess.run(shlex.split("termux-open ./figs/ellipse.pdf"))
#else
plt.show()
