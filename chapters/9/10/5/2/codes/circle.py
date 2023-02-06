import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sym
def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ

def line_gen(A,B):
  len =10
  dim = A.shape[0]
  x_AB = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
  return x_AB

#input parameters are
r=1
u=0
O=np.array(([0,0]))  
P=np.array(([1,0]))
theta=60*np.pi/180
alpha=130*np.pi/180
beta=-40*np.pi/180
#output parameters are
Q=np.array([np.cos(theta),np.sin(theta)])
R=np.array([np.cos(alpha),np.sin(alpha)])
S=np.array([np.cos(beta),np.sin(beta)])
m=math.dist(Q,R)*math.dist(P,R)
m1=np.linalg.norm(Q-R)**2*np.linalg.norm(P-R)**2
M=m/m1
angle1=np.arccos(M)
n=math.dist(Q,S)*math.dist(P,S)
n1=np.linalg.norm(Q-S)**2*np.linalg.norm(P-S)**2
N=n/n1
angle2=np.arccos(N)
print(Q)
print(R)
print(S)
print(m)
print(m1)
print(M)
print(n)
print(n1)
print(N)
print(angle1)
print(angle2)
##Generating the li

xOQ = line_gen(O,Q)
xOP = line_gen(O,P)
xQS = line_gen(Q,S)
xQR = line_gen(Q,R)
xPR = line_gen(P,R)
xQP = line_gen(Q,P)
xPS = line_gen(P,S)
##Generating the circle
x_circ= circ_gen(O,r)

#Plotting all lines
plt.plot(xQP[0,:],xQP[1,:],label='QP')
plt.plot(xPS[0,:],xPS[1,:],label='PS')
plt.plot(xOQ[0,:],xOQ[1,:],label='OQ')
plt.plot(xOP[0,:],xOP[1,:],label='OP')
plt.plot(xQS[0,:],xQS[1,:],label='QS')
plt.plot(xQR[0,:],xQR[1,:],label='QR')
plt.plot(xPR[0,:],xPR[1,:],label='PR')

#Plotting the circle
plt.plot(x_circ[0,:],x_circ[1,:],label='Circle')


#Labeling the coordinates
tri_coords = np.vstack((O,P,Q,R,S)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','P','Q','R','S']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
