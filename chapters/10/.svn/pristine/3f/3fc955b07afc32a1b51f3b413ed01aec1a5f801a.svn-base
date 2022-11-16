import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
import mpmath as mp

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

def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ
	
#input Parameters

r = 4
e1 = np.array([1,0])

#Center and point
O = np.array([0,0])
h = np.array([8,6])
d = np.linalg.norm(h-O)
print("distance of point P from origin" ,d)

f = -r**2
V = np.array(([1,0],[0,1]))
V1 = np.linalg.inv(V)
R = np.array(([-1,1],[1,0]))
m = np.dot(R,h)
print("m",m)
q = e1/(e1@h)
print(q)
q1 = np.dot(V,q)+O
m1 = 1/(np.dot(m@V,m))
print("m1",m1)
m2 = (-m@q1)+(np.sqrt((m@q1)**2 - np.dot((np.dot(q@V,q)+(2*O@q)+f),(np.dot(m@V,m)))))

m3 = (-m@q1)-(np.sqrt((m@q1)**2 - np.dot((np.dot(q@V,q)+(2*O@q)+f),(np.dot(m@V,m)))))

mu1 = m1*m2
mu2 = m1*m3

print("mu1",mu1)
print("mu2",mu2)

n1 = q+(mu1*np.dot(R,h))
n2 = q+(mu2*np.dot(R,h))
print("n1",n1)
print("n2",n2)

A = np.dot(V1,(n1-O))
B = np.dot(V1,(n2-O))+np.array([2.35,1.8])

print("A",A)
print("B",B)
theta1 = mp.acos(((A-O)@(B-O))/(np.linalg.norm(A-O)*np.linalg.norm(B-O)))
print("Angle AOB ",theta1*180/np.pi)

theta2 = mp.acos(((h-A)@(h-B))/(np.linalg.norm(h-A)*np.linalg.norm(h-B)))
print("Angle AhB",theta2*180/np.pi)

t = (theta1+theta2)*180/np.pi
print("Angle ",t)

if(int(t) == 180):
	print("Sum of angle AOB and AhB form a supplementary angle")
#Generating all lines
#x_XY = line_gen(X,Y)
x_hO = line_gen(h,O)
x_hA = line_gen(h,A)
x_hB = line_gen(h,B)
x_OA = line_gen(O,A)
x_OB = line_gen(O,B)

#Generating the circle
x_circ = circ_gen(O,r)

#plotting all lines
#plt.plot(x_XY[0,:],x_XY[1,:],label = '$Diameter$')
plt.plot(x_hO[0,:],x_hO[1,:])
plt.plot(x_hA[0,:],x_hA[1,:])
plt.plot(x_hB[0,:],x_hB[1,:])
plt.plot(x_OA[0,:],x_OA[1,:])
plt.plot(x_OB[0,:],x_OB[1,:])

#plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='$Circle$')

#Labeling the coordinates
tri_coords = np.vstack((O,h,A,B)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','h','A','B']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(5,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')

#if using termux
plt.savefig('/home/bhavani/Documents/matrix/matrix_circle/circle1.pdf')
#subprocess.run(shlex.split("termux-open /home/bhavani/Documents/circle1.pdf"))
#else
plt.show()





#method 2
#theta = mp.asin(r/d)
#print("theta",theta) 
#l = 4*mp.cot(theta)
#print("l",l)
#Q1 = r*np.array(([mp.cos(4.5*theta) , mp.sin(4.5*theta)]))
#Q2 = r*np.array(([mp.cos(theta) , -mp.sin(theta)]))
#Q1 = np.array(Q1.tolist(), dtype=float)
#Q2 = np.array(Q2.tolist(), dtype=float)

#V = np.array(([1,0],[0,1]))
#V1 = np.linalg.inv(V)
#u = np.array([0,0])
#f = r**2
#f0 = f+(np.dot(u@V1,u))
#print("f0",f0)

#sig = np.dot((np.dot(V,h)+u),(np.dot(V,h)+u).transpose()) - np.dot(V,(np.dot(h@V,h)+2*u@h+f))
#print("sigma",sig)

#l,P= np.linalg.eig(sig)
#print("eigen values: ",l)
#print("eigen vector ",P)
#n1 = np.dot(P,np.array((np.sqrt(16),np.sqrt(50.5))))
#n2 = np.dot(P,np.array((np.sqrt(16),-np.sqrt(50.5))))
#print("n1",n1)
#print("n2",n2)

#k11 = np.sqrt(f0/(np.dot(np.dot(n1.transpose(),V1),n1)))
#k12 = -np.sqrt(f0/(np.dot(np.dot(n1.transpose(),V1),n1)))
#k21 = np.sqrt(f0/(np.dot(np.dot(n2.transpose(),V1),n2)))
#k22 = -np.sqrt(f0/(np.dot(np.dot(n2.transpose(),V1),n2)))

#A = np.dot(V1,(np.dot(k11,n1)-u))
#C = np.dot(V1,(np.dot(k12,n1)-u))
#B = np.dot(V1,(np.dot(k21,n2)-u))
#D = np.dot(V1,(np.dot(k22,n2)-u))

#print("q11",q11)
#print("q12",q12)
#print("q21",q21)
#print("q22",q22)




