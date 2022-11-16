import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def line_gen(A,B):
   len =10
   dim = A.shape[0]
   x_AB = np.zeros((dim,len))
   lam_1 = np.linspace(0,1,len)
   for i in range(len):
     temp1 = A + lam_1[i]*(B-A)
     x_AB[:,i]= temp1.T
   return x_AB

def dir_vec(A,B):
   return B-A
 
def norm_vec(A,B):
   return np.matmul(omat, dir_vec(A,B))

def circ_gen(O,r):
   len = 50
   theta = np.linspace(0,2*np.pi,len)
   x_circ = np.zeros((2,len))
   x_circ[0,:] = r*np.cos(theta)
   x_circ[1,:] = r*np.sin(theta)
   x_circ = (x_circ.T + O).T
   return x_circ

#if using termux
import subprocess
import shlex
#end if

omat=np.array([[0,-1],[1,0]])
#input parameters
r=3#radius
r1=4
alpha=120
lamada=-5.24/2.63
u=1
e1=np.array([1,0])
e2=np.array([0,1])
O = np.array([0,0])
A = r1*np.array([np.cos(alpha),np.sin(alpha)])
print(A)
At=A-O
a=r1**2
b = -2*(r**2)*(e1.T@At)
c = (r**2)*(r**2 - (e2.T@At)**2)
t1 = (-b + (np.sqrt(b**2 - 4*a*c)))/(2*a)
t2 = (-b - (np.sqrt(b**2 - 4*a*c)))/(2*a)
print(t1)
print(t2)
P_p = np.array([t1, np.sqrt(r**2 - t1**2)])
P_n = np.array([t1, -np.sqrt(r**2 - t1**2)])
if((At-P_p).T@P_p == 0):
    P = P_p + O
else:
    P = P_n + O
Cp = np.array([t2, np.sqrt(r**2 - t2**2)])
Cn = np.array([t2, -np.sqrt(r**2 - t2**2)])
if((At-Cn).T@Cn == 0):
    C = Cn + O
else:
    C = Cp + O
Q = 2*O-P
print(Q)
m = A-C
n = P-A
print(m)
print(n)
j=omat@n
k=omat@m
c1=j.T@Q
c2=k.T@A
B = LA.inv(np.vstack((j.T,k.T)))@np.array(([c1,c2]))
print(B)
X=P+u*n
Y=A-u*n
E=Q+u*n
F=B-u*n

##Generating all lines
x_OP = line_gen(O,P)
x_OQ = line_gen(O,Q)
x_OA = line_gen(O,A)
x_OB = line_gen(O,B)
x_QB = line_gen(Q,B)
x_AC = line_gen(A,C)
x_CB = line_gen(C,B)
x_PA = line_gen(P,A)
x_QB = line_gen(Q,B)
x_AY = line_gen(A,Y)
x_XP = line_gen(X,P)
x_EQ = line_gen(E,Q)
x_BF = line_gen(B,F)
x_OC = line_gen(O,C)
x_circ_1 = circ_gen(O,r)


#Plotting all lines
plt.plot(x_OP[0,:],x_OP[1,:])
plt.plot(x_OQ[0,:],x_OQ[1,:])
plt.plot(x_OA[0,:],x_OA[1,:])
plt.plot(x_OB[0,:],x_OB[1,:])
plt.plot(x_QB[0,:],x_QB[1,:])
plt.plot(x_AC[0,:],x_AC[1,:])
plt.plot(x_CB[0,:],x_CB[1,:])
plt.plot(x_PA[0,:],x_PA[1,:])
plt.plot(x_QB[0,:],x_QB[1,:])
plt.plot(x_AY[0,:],x_AY[1,:])
plt.plot(x_XP[0,:],x_XP[1,:])
plt.plot(x_EQ[0,:],x_EQ[1,:])
plt.plot(x_BF[0,:],x_BF[1,:])
plt.plot(x_OC[0,:],x_OC[1,:])
plt.plot(x_circ_1[0,:],x_circ_1[1,:])


#Labeling the coordinates
tri_coords = np.vstack((O,P,Q,A,B,C,X,Y,E,F)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','P','Q','A','B','C','X','Y','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
#plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
plt.savefig('/sdcard/Download/chinna/matrix/circle_assignment/c.pdf')
subprocess.run(shlex.split("termux-open '/sdcard/Dowload/chinna/matrix/circle_assignment/c.pdf'")) 
#else
#plt.show()
