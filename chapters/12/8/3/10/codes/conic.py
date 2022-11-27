import sys
import numpy as np
import matplotlib.pyplot as plt
from line.funcs import *
sys.path.insert(0,'/home/root1/Downloads/CoordGeo') 
from numpy import linalg as LA
from conics.funcs import *
from line.funcs import *
from scipy.integrate import quad
#  plotting parabola
x = np.linspace(-3, 3, 100)
y = (x ** 2) 
plt.plot(x, y, label='Parabola')

#plt.fill_between(x,y, where= (-1.9  < x)&(x < 0))
def integrand1(x):
    return x+2
A1,err=quad(integrand1, -2,-1)
def integrand1(x):
    return x^2
A1,err=quad(integrand1, -1,0)
mu1=1
mu2=2
mu3=0
B=np.array([-2,0])+mu1*np.array([1,1])
O=np.array([-2,0])+mu2*np.array([1,1])
X=np.array([-2,0])+mu3*np.array([1,1])
print(B)
print(O)
print(x)
#B =  np.array(([0,2]))
A =  np.array(([2,4]))
#O =  np.array(([-2,0]))
xAB = line_gen(B,O)
xBC = line_gen(B,A)
xXO=line_gen(O,X)
plt.plot(xAB[0,:],xAB[1,:])
plt.plot(xBC[0,:],xBC[1,:])
plt.plot(xXO[0,:],xXO[1,:])

#plt.axvspan(2,4,ymin=1)
# Plotting the Lines
x_cor1 = [0, 0]
y_cor1 = [-5, 5]
plt.plot(x_cor1, y_cor1, 'r')
x_cor1 = [-6, 6]
y_cor1 = [0, 0]
plt.plot(x_cor1, y_cor1, 'r')

m=np.array([1,1]);#direction vector
q= np.array([0,2]);
V=np.array([[1,0],[0,0]]);
u=np.array([0,-0.5]);
f=0;
d = np.sqrt((m.T@(V@q + u)**2) - (q.T@V@q + 2*u.T@q + f)*(m.T@V@m))
print("d is =",d)
k1 = (d - m.T@(V@q + u))/(m.T@V@m)
k2 = (-d - m.T@(V@q + u))/(m.T@V@m)
print("k1 =",k1)
print("k2 =",k2)
a0 = q + k1*m
a1 = q + k2*m
p1,p2=inter_pt(m,q,V,u,f)
print("a0 =",a0)
print("a1 =",a1)

'''tri_coords = np.vstack((B,O)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['a1','a0']
for i, txt in enumerate(vert_labels):
      plt.annotate(txt,      # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points",   # how to position the text
                 xytext=(0,10),     # distance from text to points (x,y)
                 ha='center')     # horizontal alignm'''

plt.axis('equal')
plt.legend(loc='best')
plt.grid()
plt.show()
