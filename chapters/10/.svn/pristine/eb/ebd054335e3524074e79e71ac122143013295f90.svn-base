import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore",np.ComplexWarning)
import sys
sys.path.insert(0,'/sdcard/fwc/matrices/CoordGeo')

import subprocess
import shlex

r1=4
r2=6
d=6
e1=np.array(([1,0]))
O=d*e1
M1=np.array(([1,2/np.sqrt(5)]))
M2=np.array(([1,-2/np.sqrt(5)]))
P=np.array(([0,0]))
v=np.array([[1,0],[0,1]])
U=np.array([-6,0])
F=20

def conic_quad(q,V,u,f):
	return q@V@q + 2*u@q + f

def inter_pt(m,q,V,u,f):
    a = m@V@m
    b = m@(V@q+u)
    c = conic_quad(q,V,u,f)
    l1,l2 =np.roots([a,2*b,c])
    x1 = q+l1*m
    x2 = q+l2*m
    return x1,x2

def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ

def line_gen(X,Y):
  len =10
  dim = X.shape[0]
  x_XY = np.zeros((dim,len))
  lam_1 = np.linspace(0,1,len)
  for i in range(len):
    temp1 = X + lam_1[i]*(Y-X)
    x_XY[:,i]= temp1.T
  return x_XY

x_circ1= circ_gen(O,r1)
x_circ2= circ_gen(O,r2)

P1,Q1=inter_pt(M1,P,v,U,F)
P2,Q2=inter_pt(M2,P,v,U,F)
D=np.linalg.norm(P-Q1)
print('PQ1=',D)

x_OP=line_gen(O,P)
x_PQ1=line_gen(P,Q1)
x_PQ2=line_gen(P,Q2)
x_OQ1=line_gen(O,Q1)
x_OQ2=line_gen(O,Q2)

plt.plot(x_PQ1[0,:],x_PQ1[1,:],'g',label='$Tangent1$')
plt.plot(x_PQ2[0,:],x_PQ2[1,:],'g',label='$Tangent2$')
plt.plot(x_OQ1[0,:],x_OQ1[1,:],'y',label='$Radius$')
plt.plot(x_OQ2[0,:],x_OQ2[1,:],'y')
plt.plot(x_OP[0,:],x_OP[1,:],'-.',label='$Radius$')


plt.plot(x_circ1[0,:],x_circ1[1,:],'r',label='$Circle$')
plt.plot(x_circ2[0,:],x_circ2[1,:],'r')

tri_coords = np.vstack((O,P,Q1,Q2)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['O','P','Q1','Q2']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-2,4), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.axis('equal')
plt.legend(loc='best')

plt.savefig('/sdcard/fwc/matrices/circles/figs/main.pdf')
subprocess.run(shlex.split("termux-open /sdcard/fwc/matrices/circles/figs/main.pdf"))
