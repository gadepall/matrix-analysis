import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/megha/Desktop/CoordGeo') 
from numpy import linalg as LA
from conics.funcs import *
from line.funcs import *
from scipy.integrate import quad
#  plotting parabola
y = np.linspace(-7, 7, 1000)
x = (y ** 2) / 9
plt.plot(x, y, label='Parabola')
plt.fill_between(x,y, where= (4.2 < y)&(y < 6))
def integrand1(x):
    return 3*np.sqrt(x)
A1,err=quad(integrand1, 2,4)

#Points of intersection of a conic section with a line
m2=np.array([0,1]);#direction vector
q2= np.array([2,0]);
V=np.array([[0,0],[0,1]]);
u=np.array([-4.5,0]);
f=0;
d = np.sqrt((m2.T@(V@q2 + u)) - (q2.T@V@q2 + 2*u.T@q2 + f)*(m2.T@V@m2))
k1 = (d - m2.T@(V@q2 + u))/(m2.T@V@m2)
k2 = (-d - m2.T@(V@q2 + u))/(m2.T@V@m2)
print("k1 =",k1)
print("k2 =",k2)
a0 = q2 + k1*m2
a1 = q2 + k2*m2
p1,p2=inter_pt(m2,q2,V,u,f)
m1=np.array([0,1]);
q1=np.array([4,0]);
p3,p4=inter_pt(m1,q1,V,u,f)
d = np.sqrt((m1.T@(V@q1 + u))*2 - (q1.T@V@q1 + 2*u.T@q1 + f)*(m1.T@V@m1))
k3 = (d - m1.T@(V@q1 + u))/(m1.T@V@m1)
k4 = (-d - m1.T@(V@q1 + u))/(m1.T@V@m1)
print("k3 =",k3)
print("k4 =",k4)
a3 = q1 + k3*m1
a2 = q1 + k4*m1
print("a0 =",a0)
print("a1 =",a1)
print("a3 =",a3)
print("a2 =",a2)
print('the area between x=2 and x=4 bounded by the parabola is ',A1);

#Generating  line
x_R1 = line_gen(p1,p2);
x_R2 = line_gen(p3,p4);

plt.plot(x_R1[0,:],x_R1[1,:],label='$line$')
plt.plot(x_R2[0,:],x_R2[1,:],label='$line$')

tri_coords = np.vstack((p1,p2,p3,p4)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['a1','a0','a2','a3']
for i, txt in enumerate(vert_labels):
      plt.annotate(txt,      # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points",   # how to position the text
                 xytext=(0,10),     # distance from text to points (x,y)
                 ha='center')     # horizontal alignment can be left, right or center
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
plt.axis('equal')
plt.legend(loc='best')
plt.grid(True)
plt.show()
